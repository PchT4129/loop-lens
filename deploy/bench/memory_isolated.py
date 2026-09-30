"""阶段 4a · R2 显存：每个变体在**独立进程**里测，得到它在共享显存预算里的真实占用。

为什么要独立进程
----------------
bench_compile.py 第一次测显存时，FP32 基座模型一直留在显存里、前面延迟测量留下的 CUDA Graph
内存池也可能没释放，峰值全部被污染（FP16 eager 测出 139 MB，而阶段 2 单独驻留时是 53 MB）。
这是阶段 2"同时驻留"错误的重演。新进程没有任何残留，是最干净的隔离。

测什么（按"对共享显存预算的意义"由小到大）
------------------------------------------
  peak_allocated   PyTorch 张量真正用到的峰值
  peak_reserved    PyTorch 缓存分配器向驱动要下的峰值（含碎片与 CUDA Graph 内存池）
  process_total    进程实际占用的全部显存 = 驱动报告的"已用"增量。包含 PyTorch 看不到的部分：
                   CUDA context（kernel 代码、库状态）、cuBLAS 工作区等。
                   **这才是它从机器人的共享显存预算里拿走的量**

process_total 的测法：子进程启动前记下整卡已用显存（包括 Windows 桌面等其他占用），
子进程在稳态时再报一次，二者之差。两端必须用同一个接口（torch.cuda.mem_get_info）。
其他程序在这期间的波动会带来几 MB 的噪声。

有效性自检：process_total 必须 ≥ peak_reserved（context 只会让它更大）。若不满足，说明该接口在
WSL2 下不是整卡口径，context / 进程总占用这两列无法测量，只报告 PyTorch 层的数字。

用法：
    python -m deploy.bench.memory_isolated            # 父进程：依次为每个变体启动子进程
    python -m deploy.bench.memory_isolated --only fp16:trt int8_entropy_day:trt --out memory_isolated_r5.json
                                                      # 只测指定变体，写到另一个文件（不覆盖 4b 的结果）
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "deploy" / "results"
VARIANTS = [("fp16", "eager"), ("fp16", "default"), ("fp16", "reduce-overhead"),
            ("fp16", "max-autotune"), ("fp32", "eager"), ("fp32", "reduce-overhead"),
            ("fp16", "trt"), ("fp32", "trt")]


def child(precision: str, mode: str) -> None:
    """子进程：只把一个变体放上 GPU，报告各层显存占用。"""
    import copy
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(REPO))
    import torch
    from deploy.eval.runtimes import DT, _base_fp32

    free0, total = torch.cuda.mem_get_info()           # 这一步已经建立了 CUDA context
    used_after_ctx = total - free0

    if mode == "trt":
        # TensorRT 用自己的显存分配器：它的权重与激活内存 PyTorch 统计不到，
        # 只有"进程总占用"能反映。PyTorch 在这里只负责输入/输出缓冲。
        # 逐步归因：第一版只报了进程总占用，FP32 引擎（83 MB 权重）和 FP16 引擎（44 MB）总占用却几乎相同
        # （108 vs 106 MB），说不清。这里在每一步之后读一次可用显存，看显存花在哪一步。
        import tensorrt as trt
        from deploy.export.trt_runner import TRTRunner
        eng_path = REPO / "deploy" / "results" / "engines" / f"vits14_mean_224_b1_{precision}.engine"
        f_a, _ = torch.cuda.mem_get_info()
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = rt.deserialize_cuda_engine(eng_path.read_bytes())
        torch.cuda.synchronize()
        f_b, _ = torch.cuda.mem_get_info()
        ctx = engine.create_execution_context()
        torch.cuda.synchronize()
        f_c, _ = torch.cuda.mem_get_info()
        del ctx, engine
        runner = TRTRunner(eng_path)                    # 正式运行用一个完整的执行器
        x = torch.randn(1, 3, 224, 224, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(50):
                runner(x)
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        step_attr = {"engine_load_mb": round((f_a - f_b) / 2**20, 1),
                     "context_create_mb": round((f_b - f_c) / 2**20, 1)}
        print(json.dumps({
            "precision": precision, "mode": mode, "weights_mb": None,
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "used_after_context_mb": round(used_after_ctx / 2**20, 1),
            "used_steady_mb": round((total - free1) / 2**20, 1),
            "trt_engine_device_memory_mb": round(runner.engine.device_memory_size_v2 / 2**20, 1),
            **step_attr,
        }))
        return

    base = _base_fp32().cpu()                          # 基座放 CPU，不占显存
    torch.cuda.empty_cache()
    m = copy.deepcopy(base).to(device="cuda", dtype=DT[precision]).eval()
    del base
    fn = m if mode == "eager" else torch.compile(
        m, mode=None if mode == "default" else mode, fullgraph=True, dynamic=False)
    x = torch.randn(1, 3, 224, 224, device="cuda", dtype=DT[precision])
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(50):
            fn(x)
        torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    print(json.dumps({
        "precision": precision, "mode": mode,
        "weights_mb": round(sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "used_after_context_mb": round(used_after_ctx / 2**20, 1),
        "used_steady_mb": round((total - free1) / 2**20, 1),
    }))


def gpu_used_mb() -> float:
    """整卡已用显存，口径与子进程的 torch.cuda.mem_get_info 一致。

    ✏️ 第一版用 nvidia-smi 的 memory.used 做基线，得出"进程占用 −3.9 GB"：WSL2 下 nvidia-smi 与
    CUDA 的 mem_get_info 对"已用"的口径本来就差约 4 GB（阶段 0 即可见：nvidia-smi 报已用 4.9 GB，
    CUDA 报已用 1.1 GB）。两种口径不能相减。改为两端都用 mem_get_info。
    父进程为此会建立自己的 CUDA context，它在子进程运行前后都存在，差值中被抵消。
    """
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def main(variants=VARIANTS, out_name: str = "memory_isolated_r2.json") -> None:
    rows = []
    print(f"{'变体':<22}{'权重':>8}{'allocated':>11}{'reserved':>10}{'CUDA context':>14}{'进程总占用':>12}")
    for precision, mode in variants:
        before = gpu_used_mb()                         # 子进程启动前的整卡已用（同一口径）
        out = subprocess.run([sys.executable, "-m", "deploy.bench.memory_isolated", "--child",
                              precision, mode], cwd=REPO, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(f"{precision} {mode}: 子进程失败\n{out.stderr[-800:]}")
            continue
        r = json.loads(line[-1])
        r["gpu_used_before_mb"] = before
        r["context_mb"] = round(r["used_after_context_mb"] - before, 1)
        r["process_total_mb"] = round(r["used_steady_mb"] - before, 1)
        r["process_total_valid"] = r["process_total_mb"] >= r["peak_reserved_mb"]
        rows.append(r)
        w = f"{r['weights_mb']:>7.1f}M" if r["weights_mb"] is not None else f"{'(TRT内)':>8}"
        print(f"{precision + ' ' + mode:<22}{w}{r['peak_allocated_mb']:>10.1f}M"
              f"{r['peak_reserved_mb']:>9.1f}M{r['context_mb']:>13.1f}M{r['process_total_mb']:>11.1f}M"
              f"  {'✓' if r['process_total_valid'] else '✗ 口径不可信'}"
              + (f"   [TRT 归因] 加载引擎 {r['engine_load_mb']} MB | 创建执行上下文 {r['context_create_mb']} MB"
                 f" | 引擎自报激活内存 {r['trt_engine_device_memory_mb']} MB" if mode == "trt" else ""))
    path = RESULTS / out_name
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {path.relative_to(REPO)}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--only":
        rest = sys.argv[2:]
        out_name = "memory_isolated_custom.json"
        if "--out" in rest:
            k = rest.index("--out")
            out_name, rest = rest[k + 1], rest[:k]
        main([tuple(v.split(":")) for v in rest], out_name)
    else:
        main()
