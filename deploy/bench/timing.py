"""GPU 计时的正确做法，以及为什么不能用 time.time()。

核心问题：CUDA 是**异步**的
------------------------------
`y = model(x)` 这一行返回时，GPU 上的计算通常**还没做完**。Python 只是把一串
kernel 提交到 CUDA stream 上就继续往下走了。所以：

    t0 = time.time(); y = model(x); t1 = time.time()   # ❌ 测到的是"提交耗时"

这样测出来的数可能只有真实耗时的几十分之一，而且 batch 越大错得越离谱。

两种正确做法：
1. `time.time()` 前后各加 `torch.cuda.synchronize()` —— 可行，但 synchronize
   本身有开销，且把 CPU 等待也算进去了。
2. **CUDA Event**（本模块采用）—— 让 GPU 自己在 stream 上打时间戳，测的是
   GPU 侧的真实执行区间，不含 Python/CPU 开销。

其余三件必须做对的事
--------------------
* **预热**：首次调用要付 kernel 加载、显存分配器冷启动、cuDNN/cuBLAS 算法选择
  等一次性开销。更重要的是——实测发现 GPU 空闲时处于低功耗档（显存频率
  9001 MHz 而非 14001 MHz），带宽会低 12–16%。所以预热同时是**把硬件拉到
  稳定高频**。见 EXPERIMENTS.md 发现 2。
* **足够多次重复 + 报分位数**：单次测量没有意义。报中位数（抗离群）、p90/p99
  （看抖动）。只报均值会被个别毛刺带偏。
* **记录条件**：峰值显存、当时可用显存、SM/显存频率、功耗、温度。不记录这些，
  隔天的数字不可比；而且峰值显存超过物理容量时会静默溢出到系统内存（发现 1），
  必须靠记账把这种测量标记为不可信。
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import torch

# 本机显存满频。低于这个值说明 GPU 还在低功耗档，测量会偏低。
TARGET_MEM_CLOCK_MHZ = 14001


def _smi(fields: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "nvidia-smi failed")
    return [v.strip() for v in out.stdout.strip().split("\n")[0].split(",")]


def mem_clock_mhz() -> float:
    try:
        return float(_smi("clocks.mem")[0])
    except Exception:                                         # noqa: BLE001
        return float("nan")


class _ConditionSampler:
    """在测量期间后台采样频率/功耗/温度。

    走 nvidia-smi 子进程，不占 GPU，也不会干扰被测 kernel。
    """

    def __init__(self, interval_s: float = 0.15):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sm, mem, power, temp = _smi("clocks.sm,clocks.mem,power.draw,temperature.gpu")
                self.samples.append((float(sm), float(mem), float(power), float(temp)))
            except Exception:                                 # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_ConditionSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        sm, mem, power, temp = zip(*self.samples)
        return {
            "sm_clock_mhz_median": statistics.median(sm),
            "sm_clock_mhz_max": max(sm),
            "mem_clock_mhz_median": statistics.median(mem),
            "power_w_median": statistics.median(power),
            "power_w_max": max(power),
            "temp_c_max": max(temp),
            "n_condition_samples": len(self.samples),
        }


@dataclass
class TimingResult:
    label: str
    median_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    n_iters: int
    n_warmup: int
    peak_reserved_gb: float
    peak_allocated_gb: float
    free_vram_before_gb: float
    total_vram_gb: float
    vram_headroom_ok: bool
    conditions: dict = field(default_factory=dict)

    @property
    def mem_clock_ok(self) -> bool:
        """测量期间显存是否保持满频。

        实测发现显存频率会在 11001/12501/14001 MHz 之间跳动，且**与温度和功耗墙
        都无关**（温度仅 57–62°C，功耗未触及 140 W 上限）。带宽与该频率近似成
        正比，所以低频下测得的访存受限数据偏低、不能与满频数据混用。
        见 EXPERIMENTS.md 发现 3。
        """
        clock = self.conditions.get("mem_clock_mhz_median")
        return clock is None or clock >= TARGET_MEM_CLOCK_MHZ

    @property
    def jitter_ratio(self) -> float:
        """p99 / 中位数。明显大于 1 说明有抖动，通常是降频或别的进程在抢 GPU。"""
        return self.p99_ms / self.median_ms if self.median_ms else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["jitter_ratio"] = round(self.jitter_ratio, 3)
        d["mem_clock_ok"] = self.mem_clock_ok
        return d

    def __str__(self) -> str:
        flag = "" if self.vram_headroom_ok else "  ⚠️显存超可用量,可能已溢出到系统内存,不可信"
        if not self.mem_clock_ok:
            flag += "  ⚠️显存未满频,访存受限数据偏低"
        return (
            f"{self.label:<34} 中位 {self.median_ms:8.3f} ms | "
            f"p90 {self.p90_ms:8.3f} | p99 {self.p99_ms:8.3f} "
            f"(抖动 {self.jitter_ratio:.2f}x) | 峰值显存 {self.peak_reserved_gb:5.2f} GB"
            f"{flag}"
        )


def ramp_clocks(max_seconds: float = 8.0, verbose: bool = True) -> float:
    """用持续负载把 GPU 从低功耗档拉到满频，返回达到的显存频率。

    为什么需要：见模块文档。空闲态测出的带宽会低 12–16%。
    """
    a = torch.randn(4096, 4096, device="cuda")
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        for _ in range(20):
            a = a @ a.clamp(-1, 1) * 0.001        # clamp 防数值爆掉
        torch.cuda.synchronize()
        if mem_clock_mhz() >= TARGET_MEM_CLOCK_MHZ:
            break
    del a
    torch.cuda.empty_cache()

    clock = mem_clock_mhz()
    if verbose:
        ok = "✓ 已满频" if clock >= TARGET_MEM_CLOCK_MHZ else "⚠️ 未达满频,测量可能偏低"
        print(f"  [预热] 显存频率 {clock:.0f} / {TARGET_MEM_CLOCK_MHZ} MHz  {ok}")
    return clock


def time_cuda(
    fn: Callable[[], object],
    label: str = "",
    warmup: int = 50,
    iters: int = 300,
    sample_conditions: bool = True,
) -> TimingResult:
    """用 CUDA Event 给 fn 计时，返回分位数与运行条件。

    每次迭代单独打一对 event，循环结束后统一 synchronize 再读数 —— 这样
    既能拿到逐次耗时（才能算分位数），又避免每次迭代都 synchronize 引入开销。
    """
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 预热之后再清峰值计数，这样测到的是稳态占用
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    sampler = _ConditionSampler() if sample_conditions else None
    if sampler is not None:
        sampler.__enter__()
    try:
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    peak_reserved = torch.cuda.max_memory_reserved()
    # allocated = 张量真正占用；reserved = 缓存分配器向驱动要下的总量（含空闲块）。
    # 部署预算看 reserved（别的进程拿不到这部分），分析模型本身看 allocated。
    peak_allocated = torch.cuda.max_memory_allocated()

    def q(p: float) -> float:
        # 分位数用最近秩法，样本已排序
        return times[min(len(times) - 1, int(p * len(times)))]

    return TimingResult(
        label=label,
        median_ms=statistics.median(times),
        p90_ms=q(0.90),
        p99_ms=q(0.99),
        mean_ms=statistics.fmean(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=times[0],
        n_iters=iters,
        n_warmup=warmup,
        peak_reserved_gb=round(peak_reserved / 1024**3, 3),
        peak_allocated_gb=round(peak_allocated / 1024**3, 3),
        free_vram_before_gb=round(free_before / 1024**3, 3),
        total_vram_gb=round(total / 1024**3, 3),
        # ⭐ 发现1 的防护：峰值占用必须小于测量前的可用显存，否则可能已分页到系统内存
        vram_headroom_ok=peak_reserved < free_before,
        conditions=sampler.summary() if sampler is not None else {},
    )
