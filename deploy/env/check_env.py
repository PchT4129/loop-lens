"""阶段 0：环境自检。

这个脚本的产出不是"环境能跑"，而是**后面所有性能数字可信**。
每一项检查都对应一个会让测量静默失真的具体机制：

1. sm_120 支持     —— 缺了它 PyTorch 会走 JIT 或根本不支持，性能无从谈起
2. 显存溢出探测 ⭐ —— 显存不足时不会 OOM，而是静默溢出到系统内存。表现是
                      "不报错、只是慢"，极易被误读成"量化没有收益"。实测证实 WSL2
                      下靠控制面板设置关不掉，所以用带宽判定并改为按次记账防护。
3. 时钟/功耗基线   —— 笔记本 GPU 会降频。不记录这些，前后两天的数字就不可比
4. 工具链版本      —— TensorRT 与驱动/CUDA 的兼容性必须在动手前确认

用法：
    python deploy/env/check_env.py              # 打印报告
    python deploy/env/check_env.py --json PATH  # 另存机器可读结果
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {PASS: "[ PASS ]", WARN: "[ WARN ]", FAIL: "[ FAIL ]"}

results: list[dict] = []


def record(name: str, status: str, detail: str, **extra) -> None:
    results.append({"check": name, "status": status, "detail": detail, **extra})
    print(f"{_MARK[status]} {name}: {detail}")


def nvidia_smi(fields: str) -> list[str]:
    """查询 nvidia-smi 的指定字段。WSL2 下可用，但部分字段可能返回 [N/A]。"""
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "nvidia-smi failed")
    return [v.strip() for v in out.stdout.strip().split("\n")[0].split(",")]


# --------------------------------------------------------------- 1. 解释器与包
def check_interpreter() -> None:
    version = platform.python_version()
    major_minor = tuple(int(x) for x in version.split(".")[:2])
    # 3.13 上 onnxruntime-gpu / TensorRT 的 wheel 常常缺 cp313 轮子
    status = PASS if major_minor == (3, 11) else WARN
    prefix = sys.prefix.replace(str(Path.home()), "~")      # 结果会入库，不写本机主目录
    record("Python", status, f"{version} (prefix={prefix})", version=version)


def check_packages() -> None:
    import importlib.metadata as md

    # (包名, 是否必需)
    wanted = [
        ("torch", True), ("torchvision", True),
        ("onnx", True), ("onnxscript", False),
        ("onnxruntime-gpu", False), ("nvidia-modelopt", False),
        ("numpy", True), ("matplotlib", True), ("pillow", True),
    ]
    for name, required in wanted:
        try:
            record(f"pkg:{name}", PASS, md.version(name), version=md.version(name))
        except md.PackageNotFoundError:
            record(
                f"pkg:{name}",
                FAIL if required else WARN,
                "未安装" + ("" if required else "（可选，计划已按缺失设计降级路径）"),
            )


def check_tensorrt() -> None:
    """TensorRT 用 import 判定而非发行包名。

    pip 上的发行包名随 CUDA 线变化（tensorrt / tensorrt-cu12 / tensorrt-cu13），
    但可导入的模块名始终是 tensorrt，所以用 import 更稳。
    顺便探测平台对各精度的硬件支持——这是阶段 4 里 FP8/FP4 能否上的前置条件。
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        record("pkg:tensorrt", FAIL, f"无法导入: {exc}")
        return

    record("pkg:tensorrt", PASS, trt.__version__, version=trt.__version__)

    try:
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        fp16, int8 = builder.platform_has_fast_fp16, builder.platform_has_fast_int8
        record(
            "TRT 平台能力", PASS if (fp16 and int8) else WARN,
            f"fast FP16={fp16} | fast INT8={int8}",
            platform_has_fast_fp16=fp16, platform_has_fast_int8=int8,
        )
    except Exception as exc:                                  # noqa: BLE001
        record("TRT 平台能力", WARN, f"探测失败: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------- 2. GPU 与驱动
def check_gpu() -> "tuple[object, int] | tuple[None, None]":
    import torch

    record("torch.version", PASS, f"{torch.__version__} (built for CUDA {torch.version.cuda})")

    if not torch.cuda.is_available():
        record("CUDA 可用性", FAIL, "torch.cuda.is_available() 为 False")
        return None, None

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    record(
        "GPU", PASS,
        f"{props.name} | sm_{cap[0]}{cap[1]} | {props.multi_processor_count} SM "
        f"| {props.total_memory / 1024**3:.2f} GB",
        gpu_name=props.name, capability=f"sm_{cap[0]}{cap[1]}",
        sm_count=props.multi_processor_count,
        total_vram_gb=round(props.total_memory / 1024**3, 2),
    )

    # Blackwell(sm_120)必须在编译期 arch 列表里，否则要么不支持、要么走 PTX JIT
    arch = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    record(
        f"{arch} 原生支持", PASS if arch in arch_list else FAIL,
        f"arch_list={arch_list}" if arch in arch_list
        else f"{arch} 不在 arch_list 中，会退化为 PTX JIT，性能不可信。arch_list={arch_list}",
        arch_list=arch_list,
    )

    try:
        driver, = nvidia_smi("driver_version")
        record("驱动", PASS, driver, driver_version=driver)
    except Exception as exc:                                  # noqa: BLE001
        record("驱动", WARN, f"读取失败: {exc}")

    return props, props.total_memory


# ------------------------------------------------- 3. 显存溢出探测（核心检查）
def _write_bandwidth(nbytes: int, iters: int = 5) -> tuple[float, float]:
    """对一块显存反复整体写入，返回 (GB/s, 中位耗时秒)。"""
    import torch

    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buf.fill_(1)
    torch.cuda.synchronize()

    start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for i in range(iters):
        start_ev.record()
        buf.fill_(i % 256)
        end_ev.record()
        torch.cuda.synchronize()
        samples.append(start_ev.elapsed_time(end_ev) / 1e3)
    del buf
    torch.cuda.empty_cache()

    samples.sort()
    median = samples[len(samples) // 2]
    return nbytes / median / 1e9, median


def check_memory_fallback() -> None:
    """⭐ 探测超额显存申请会不会静默溢出到系统内存。

    为什么不能只看"申请是否成功"
    ----------------------------
    最初的设计是"申请超过可用显存，期望 OOM"。但实测发现在 WSL2 下超额申请
    **会成功**——CUDA 在 WSL 上走 WDDM 显存分页，驱动允许超订并把超出部分放到
    系统内存。NVIDIA 控制面板的"CUDA 系统内存回退策略"主要约束原生 Windows 应用，
    在 WSL2 里这条防线守不住，所以"申请成功"本身不能判定环境是否干净。

    真正能判定的是**带宽**：显存内写入是数百 GB/s，被分页到系统内存要走 PCIe，
    会掉一个数量级。本机实测 9.15 GB 时 508 GB/s，13.46 GB 时 35.1 GB/s。

    结论对项目的影响
    ----------------
    防护必须从"关掉开关"移到"每次测量记账"：每个测量点都记录峰值显存与当时可用
    显存，任何接近或超过物理容量的配置一律标记为不可信，而不是直接采信它的延迟。
    """
    import torch

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    record(
        "显存现状", PASS,
        f"可用 {free / 1024**3:.2f} GB / 共 {total / 1024**3:.2f} GB"
        f"（其余被 Windows 侧占用）",
        free_vram_gb=round(free / 1024**3, 2),
        total_vram_gb=round(total / 1024**3, 2),
    )

    # 基准：明显装得下的一块，代表真实显存带宽
    try:
        fit_bw, _ = _write_bandwidth(min(int(free * 0.5), 4 * 1024**3))
    except Exception as exc:                                  # noqa: BLE001
        record("显存溢出探测 ⭐", WARN, f"基准带宽测量失败: {type(exc).__name__}: {exc}")
        return

    # 探测：超过可用显存
    target = int(free * 1.25)
    try:
        spill_bw, _ = _write_bandwidth(target)
    except torch.OutOfMemoryError:
        record(
            "显存溢出探测 ⭐", PASS,
            f"超额申请 {target / 1024**3:.2f} GB 如期 OOM —— 不存在静默溢出，"
            f"显存内写入带宽 {fit_bw:.0f} GB/s",
            baseline_write_gbs=round(fit_bw, 1), oom_as_expected=True,
        )
        return
    except Exception as exc:                                  # noqa: BLE001
        record("显存溢出探测 ⭐", WARN, f"探测异常: {type(exc).__name__}: {exc}")
        return

    ratio = fit_bw / max(spill_bw, 1e-9)
    if ratio >= 3.0:
        record(
            "显存溢出探测 ⭐", WARN,
            f"超额申请 {target / 1024**3:.2f} GB 成功但带宽塌陷："
            f"{fit_bw:.0f} -> {spill_bw:.0f} GB/s（{ratio:.1f}x）。"
            f"这是 WSL2 的 WDDM 分页行为，靠控制面板设置关不掉。"
            f"防护改为：每个测量点记录峰值显存，超过物理容量的配置标记为不可信",
            baseline_write_gbs=round(fit_bw, 1), spill_write_gbs=round(spill_bw, 1),
            slowdown_x=round(ratio, 1), oom_as_expected=False,
        )
    else:
        record(
            "显存溢出探测 ⭐", FAIL,
            f"超额申请 {target / 1024**3:.2f} GB 成功且带宽未明显下降"
            f"（{fit_bw:.0f} -> {spill_bw:.0f} GB/s）——与硬件容量矛盾，需人工核查",
            baseline_write_gbs=round(fit_bw, 1), spill_write_gbs=round(spill_bw, 1),
        )


# --------------------------------------------------- 4. 时钟 / 功耗 / 温度基线
def check_clocks_and_power() -> None:
    """记录运行条件。笔记本 GPU 会因功耗墙和温度降频，不记录就无法解释前后差异。"""
    fields = ("clocks.sm,clocks.max.sm,clocks.mem,power.draw,"
              "power.limit,power.max_limit,temperature.gpu,utilization.gpu")
    try:
        values = nvidia_smi(fields)
        sm, sm_max, mem, draw, limit, limit_max, temp, util = values
        record(
            "时钟与功耗基线", PASS,
            f"SM {sm}/{sm_max} MHz | 显存 {mem} MHz | 功耗 {draw}/{limit} W "
            f"(上限 {limit_max} W) | {temp}°C | 占用 {util}%",
            clocks_sm_mhz=sm, clocks_sm_max_mhz=sm_max, clocks_mem_mhz=mem,
            power_draw_w=draw, power_limit_w=limit, power_max_limit_w=limit_max,
            temperature_c=temp, utilization_pct=util,
        )
    except Exception as exc:                                  # noqa: BLE001
        record("时钟与功耗基线", WARN, f"读取失败: {exc}")

    # WSL2 通常看不到电源状态；如实报告而不是猜
    supplies = glob.glob("/sys/class/power_supply/*")
    record(
        "电源状态", PASS if supplies else WARN,
        f"{[s.split('/')[-1] for s in supplies]}" if supplies
        else "WSL2 下不可见，按用户确认的「全程插电」记录",
        power_supply_visible=bool(supplies),
    )


# --------------------------------------------------------------- 5. WSL2 配置
def check_wsl() -> None:
    import os

    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    record(
        "WSL2 内存", PASS if total_gb >= 16 else WARN,
        f"{total_gb:.1f} GB 可见内存 / {os.cpu_count()} 核",
        wsl_memory_gb=round(total_gb, 1), cpu_count=os.cpu_count(),
    )

    hits = glob.glob("/mnt/c/Users/*/.wslconfig")
    if not hits:
        record(".wslconfig", WARN, "未找到（使用 WSL2 默认分配）")
        return
    try:
        body = open(hits[0], encoding="utf-8-sig", errors="replace").read().strip()
        shown = re.sub(r"/mnt/c/Users/[^/]+/", "/mnt/c/Users/<user>/", hits[0])   # 不写 Windows 用户名
        record(".wslconfig", PASS, f"{shown}: {body!r}", path=shown, content=body)
    except OSError as exc:
        record(".wslconfig", WARN, f"{hits[0]} 读取失败: {exc}")


# --------------------------------------------------------------- 6. 剖析工具
def check_profiling_tools() -> None:
    import shutil

    for tool, required in (("nsys", False), ("ncu", False), ("trtexec", False)):
        path = shutil.which(tool)
        if path:
            record(f"tool:{tool}", PASS, path)
        else:
            record(
                f"tool:{tool}", WARN,
                "未安装（阶段 3 用 torch.profiler + 墙钟/kernel 和判据替代）"
                if tool == "nsys" else
                "未安装（WSL2 支持有限，计划未依赖）" if tool == "ncu" else
                "未安装（pip 的 tensorrt wheel 不含它；本项目走 Python API 建引擎）",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="另存机器可读结果")
    args = parser.parse_args()

    print("=" * 78)
    print(f"阶段 0 环境自检  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 78)

    print("\n--- 1. 解释器与包 ---")
    check_interpreter()
    check_packages()
    check_tensorrt()

    print("\n--- 2. GPU 与驱动 ---")
    props, _ = check_gpu()

    if props is not None:
        print("\n--- 3. 显存溢出探测 ---")
        check_memory_fallback()

        print("\n--- 4. 运行条件基线 ---")
        check_clocks_and_power()

    print("\n--- 5. WSL2 配置 ---")
    check_wsl()

    print("\n--- 6. 剖析工具 ---")
    check_profiling_tools()

    failed = [r for r in results if r["status"] == FAIL]
    warned = [r for r in results if r["status"] == WARN]

    print("\n" + "=" * 78)
    print(f"结论：{len(results) - len(failed) - len(warned)} PASS / "
          f"{len(warned)} WARN / {len(failed)} FAIL")
    for r in failed:
        print(f"  FAIL  {r['check']}: {r['detail']}")
    print("=" * 78)

    if args.json:
        from pathlib import Path

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "checks": results,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        print(f"已写入 {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
