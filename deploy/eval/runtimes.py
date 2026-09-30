"""阶段 4 的各种"执行方式"（runtime），统一注册在这里。

每个 runtime 是一个工厂：build() 返回 (fn, meta, info)
  fn   : 接收 FP32 已归一化的 [B,3,224,224] CUDA 张量，返回 [B,384] 描述子
  meta : 写进特征文件 meta["runtime"] 的记录
  info : 构建过程的附带信息（如编译耗时、图断点）

用法：
    python -m deploy.eval.runtimes extract torch-bf16          # 抽特征到 deploy/results/runtime/
    python -m deploy.eval.runtimes list
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.eval.extract_with_runtime import extract                 # noqa: E402
from deploy.export.wrapper import VPRDescriptor                      # noqa: E402
from src.backbones import DINOv2FeatureExtractor                     # noqa: E402

warnings.filterwarnings("ignore")
OUT_DIR = REPO / "deploy" / "results" / "runtime"
DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _base_fp32(res: int = 224) -> VPRDescriptor:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ext = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval()
    return VPRDescriptor(ext, res).eval()


# ------------------------------------------------------------- R0 / R1：eager
def torch_eager(precision: str):
    def build():
        m = copy.deepcopy(_base_fp32()).to(DT[precision])
        dt = DT[precision]
        return (lambda x: m(x.to(dt)),
                {"kind": "pytorch-eager", "precision": precision, "cast": "whole-model"}, {})
    return build


def torch_autocast_bf16():
    def build():
        m = _base_fp32()                      # 权重保持 FP32，运行时逐算子转换

        def fn(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m(x)
        return fn, {"kind": "pytorch-eager", "precision": "bf16", "cast": "autocast"}, {}
    return build


# ------------------------------------------------------------- R2：torch.compile
def torch_compiled(precision: str, mode: str):
    """mode: "default"（主要做算子融合）/ "reduce-overhead"（再加 CUDA Graph）/ "max-autotune"。

    fullgraph=True：要求整个模型被捕获成**一张**图。若有图断点会直接报错——
    我们想知道有没有断点，而不是让它静默地退回 Python 执行一部分。
    dynamic=False：形状固定，避免为"可变形状"生成更保守的代码。
    """
    def build():
        m = copy.deepcopy(_base_fp32()).to(DT[precision])
        dt = DT[precision]
        info: dict = {"mode": mode}
        torch._dynamo.reset()        # 同一进程里编译多个变体会撞上 recompile_limit=8，见 bench_compile.py
        compiled = torch.compile(m, mode=None if mode == "default" else mode,
                                 fullgraph=True, dynamic=False)
        x = torch.randn(1, 3, 224, 224, device="cuda", dtype=dt)
        t0 = time.perf_counter()
        with torch.inference_mode():
            compiled(x)
            torch.cuda.synchronize()
        info["first_call_s"] = round(time.perf_counter() - t0, 2)   # 含编译
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):                                       # CUDA Graph 需要几次热身才录制
                compiled(x)
            torch.cuda.synchronize()
        info["warm_calls_s"] = round(time.perf_counter() - t0, 3)
        return (lambda x: compiled(x.to(dt)).clone(),                # CUDA Graph 的输出会被下次重放覆盖
                {"kind": "torch-compile", "precision": precision, "mode": mode,
                 "fullgraph": True}, info)
    return build


# ------------------------------------------------------------- R3 / R4：TensorRT
def trt_engine(precision: str, batch: int = 1):
    """由 deploy/export/to_trt.py 构建的固定形状引擎。精度评估按部署形态用 batch 1。"""
    def build():
        import tensorrt as trt
        from deploy.export.trt_runner import TRTRunner
        eng = REPO / "deploy" / "results" / "engines" / f"vits14_mean_224_b{batch}_{precision}.engine"
        runner = TRTRunner(eng)
        return (lambda x: runner(x).clone(),                    # 输出缓冲会被下次调用覆盖
                {"kind": "tensorrt", "precision": precision, "trt_version": trt.__version__,
                 "engine": eng.name}, {})
    return build


REGISTRY = {
    "trt-fp32": trt_engine("fp32"),
    "trt-fp16": trt_engine("fp16"),
    # 阶段 4c：隐式量化 INT8（+FP16 回退），名字 = 校准算法 × 校准集
    **{f"trt-int8-{algo}-{split}": trt_engine(f"int8_{algo}_{split}")
       for algo in ("entropy", "minmax") for split in ("day", "night")},
    "torch-fp32": torch_eager("fp32"),
    "torch-fp16": torch_eager("fp16"),
    "torch-bf16": torch_eager("bf16"),
    "torch-autocast-bf16": torch_autocast_bf16(),
    **{f"compile-{m}-{p}": torch_compiled(p, m)
       for m in ("default", "reduce-overhead", "max-autotune") for p in ("fp32", "fp16", "bf16")},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("names", nargs="+")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "list":
        print("\n".join(REGISTRY))
        return
    for name in args.names:
        fn, meta, info = REGISTRY[name]()
        paths = extract(fn, meta, OUT_DIR / name.replace("-", "_"), batch_size=1)
        print(f"{name:<32} {info or ''} → {paths['query'].parent.relative_to(REPO)}/{paths['query'].stem[:-6]}_*.pt")
        del fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
