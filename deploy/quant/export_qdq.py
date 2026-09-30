"""阶段 5 ②：把一个伪量化配置导出成带 Q/DQ 的 ONNX，再建 TensorRT 显式 INT8 引擎。

显式量化的 ONNX 长什么样
------------------------
每个被量化的张量前面插一对节点：QuantizeLinear(x, scale) → DequantizeLinear(q, scale)。
数学上等于伪量化（取整到 INT8 格点再乘回 scale）。TensorRT 读到这对节点，就知道"这里要用 INT8，
缩放因子是这个"，并把 Q/DQ 与相邻的矩阵乘融合成真正的 INT8 kernel。没有 Q/DQ 的地方保持 FP16。
这就是"显式"：哪些层 INT8，由图里的 Q/DQ 决定，TensorRT 不再自己挑（对比 4c 的隐式量化）。

⚠️ 导出在 CPU 上做：在 GPU 上用 TorchScript 导出器追踪带量化器的模型会段错误（exit 139，崩在
torch.jit 的 _trace.py 里）；新的 dynamo 导出器则无法追踪 ModelOpt 的量化器（torch.export 失败）。
在 CPU 上追踪成功，而追踪只是记录计算图——导出的 Q/DQ 缩放因子与 GPU 上校准得到的完全相同。

NVFP4（阶段 6）：CPU 上导出会断言失败（动态块量化要求缩放因子在 GPU 上）；`--export-device cuda` 能跑完，
但导出的 ONNX 里**一个量化节点都没有**——量化器被静默丢掉。所以导出后会检查 Q/DQ 节点数：
启用了量化器却一个都没导出来，就直接报错，绝不继续建一个"标着低精度、其实是 FP16"的引擎。

用法：
    python -m deploy.quant.export_qdq --group qkv,proj,fc1 --algo smoothquant --tag lin_nofc2_sq --batch 1 32 --build
    python -m deploy.quant.export_qdq --fmt fp8 --group linear --tag fp8_lin --batch 1 32 --build
    python -m deploy.quant.export_qdq --fmt nvfp4 --group linear --tag nvfp4_lin --export-device cuda   # 复现静默丢失
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.quant.ptq import build_quantized                      # noqa: E402

warnings.filterwarnings("ignore")
ENG = REPO / "deploy" / "results" / "engines"


def export(group: str, algo: str, tag: str, batches: list[int], fmt: str = "int8",
           device: str = "cpu") -> dict:
    import onnx

    model, n_on = build_quantized(group, algo, fmt=fmt)
    model = model.to(device)                              # 默认 CPU：见模块文档，INT8 在 GPU 上追踪会段错误
    info: dict = {"format": fmt, "group": group, "algorithm": algo, "enabled_quantizers": n_on, "onnx": {}}
    for b in batches:
        path = ENG / f"vits14_mean_224_b{b}_qdq_{tag}.onnx"
        x = torch.randn(b, 3, 224, 224, device=device)
        with torch.inference_mode():
            torch.onnx.export(model, (x,), str(path), dynamo=False, opset_version=17,
                              input_names=["images"], output_names=["descriptors"])
        g = onnx.load(str(path))
        ops = collections.Counter(n.op_type for n in g.graph.node)
        q = sum(v for k, v in ops.items() if "QuantizeLinear" in k and "Dequantize" not in k)
        dq = sum(v for k, v in ops.items() if "DequantizeLinear" in k)
        info["onnx"][f"b{b}"] = {"path": str(path.relative_to(REPO)), "nodes": len(g.graph.node),
                                 "quantize": q, "dequantize": dq, "MatMul": ops["MatMul"],
                                 "qdq_op_types": sorted(k for k in ops if "uantize" in k)}
        print(f"b{b}: {len(g.graph.node)} 个节点 | Q {q} / DQ {dq} {sorted(k for k in ops if 'uantize' in k)} "
              f"| MatMul {ops['MatMul']} → {path.name}")
        if n_on and q == 0:
            raise SystemExit(f"❌ 启用了 {n_on} 个量化器，但导出的 ONNX 里没有任何量化节点——量化器被静默丢掉了。"
                             f"不继续建引擎（否则会得到一个标着 {fmt}、其实是 FP16 的引擎）。")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--algo", default="max", choices=["max", "smoothquant"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fmt", default="int8", choices=["int8", "fp8", "nvfp4"])
    ap.add_argument("--export-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--build", action="store_true", help="导出后接着建 TensorRT 显式量化引擎（INT8 / FP8 / FP4）")
    args = ap.parse_args()

    info = export(args.group, args.algo, args.tag, args.batch, args.fmt, args.export_device)
    if args.build:
        from deploy.export.to_trt import build
        info["engines"] = {}
        for b in args.batch:
            onnx_path = ENG / f"vits14_mean_224_b{b}_qdq_{args.tag}.onnx"
            eng = ENG / f"vits14_mean_224_b{b}_qdq_{args.tag}.engine"
            r = build(onnx_path, {"int8": "int8-explicit", "fp8": "fp8-explicit", "nvfp4": "fp4-explicit"}[args.fmt], eng)
            info["engines"][f"b{b}"] = r
            print(f"b{b} 引擎：构建 {r['build_s']} s | {r['engine_mb']} MB | 层 {r['engine_layers']} | "
                  f"INT8 输入的层 {r['layers_with_int8_input']} {r['int8_layers_by_type']} | "
                  f"GEMM {r['gemm_layers_by_input_type']} | MHA {r['mha_layers_by_input_type']}")
    out = REPO / "deploy" / "results" / f"qdq_{args.tag}.json"
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"已写入 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
