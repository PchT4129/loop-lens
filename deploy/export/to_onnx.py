"""阶段 4b · R3：把 VPRDescriptor 导出为 ONNX。

ONNX 是什么
-----------
一种与框架无关的**模型描述格式**：把"计算图 + 权重"存成一个文件。图里的每个节点是一个标准算子
（MatMul、Softmax、LayerNormalization……），算子的定义由 ONNX 标准统一规定，并按版本分组，
叫 **opset**（算子集版本）。PyTorch、TensorRT、ONNX Runtime 等工具都能读写它——
所以它是 PyTorch 与 TensorRT 之间的"通用语言"。

导出是怎么做的
--------------
torch 2.x 有两种导出器：
  * 新导出器（dynamo=True）：用 torch.export 捕获计算图（与 4a 中 torch.compile 用的是同一类捕获技术），
    再由 onnxscript 翻译成 ONNX。本脚本首选它
  * 旧导出器（TorchScript 追踪）：用一个样例输入把模型"跑一遍"，记录下经过的算子
新导出器失败时退回旧导出器，并如实记录用的是哪个。

导出时固定形状（batch、分辨率），与 4a 的 dynamic=False 同理：部署时形状不变，
固定形状能让 TensorRT 为确切的形状挑选 kernel。

用法：
    python -m deploy.export.to_onnx --batch 1
    python -m deploy.export.to_onnx --batch 32
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.eval.runtimes import _base_fp32                  # noqa: E402

warnings.filterwarnings("ignore")
ENGINE_DIR = REPO / "deploy" / "results" / "engines"          # 已被 .gitignore 忽略


def export(batch: int, res: int = 224) -> dict:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    path = ENGINE_DIR / f"vits14_mean_{res}_b{batch}_fp32.onnx"
    model = _base_fp32(res).eval()
    x = torch.randn(batch, 3, res, res, device="cuda")
    info: dict = {"batch": batch, "res": res, "path": str(path.relative_to(REPO))}

    t0 = time.perf_counter()
    try:
        prog = torch.onnx.export(model, (x,), dynamo=True, input_names=["images"],
                                 output_names=["descriptors"])
        prog.save(str(path))
        info["exporter"] = "dynamo"
    except Exception as exc:                                  # noqa: BLE001
        info["dynamo_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        torch.onnx.export(model, (x,), str(path), dynamo=False, input_names=["images"],
                          output_names=["descriptors"], opset_version=17)
        info["exporter"] = "torchscript"
    info["export_s"] = round(time.perf_counter() - t0, 1)

    import onnx
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    info["opset"] = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), None)
    info["n_nodes"] = len(m.graph.node)
    info["op_histogram"] = dict(ops.most_common())
    info["attention_kept_whole"] = "Attention" in ops
    info["size_mb"] = round(path.stat().st_size / 2**20, 1)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(m.graph.input) + list(m.graph.output)}
    info["io_shapes"] = shapes
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    allinfo = {}
    for b in args.batch:
        info = export(b)
        allinfo[f"b{b}"] = info
        print(f"batch {b}: 导出器 {info['exporter']} | opset {info['opset']} | {info['n_nodes']} 个节点 | "
              f"{info['size_mb']} MB | 耗时 {info['export_s']} s | 输入输出 {info['io_shapes']}")
        print(f"   注意力保留为整体算子: {info['attention_kept_whole']}")
        print(f"   算子分布: {info['op_histogram']}")
        if "dynamo_error" in info:
            print(f"   ⚠️ 新导出器失败，已退回旧导出器: {info['dynamo_error']}")
    out = REPO / "deploy" / "results" / "onnx_export.json"
    out.write_text(json.dumps(allinfo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
