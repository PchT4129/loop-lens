"""阶段 4b · R4：用 TensorRT 的 Python API 把 ONNX 构建成引擎（pip 版 TensorRT 不含 trtexec）。

TensorRT 的两个阶段
-------------------
  构建期（一次性，慢）：解析 ONNX → 层融合 → 为每一层把所有候选 kernel（tactic）在**这块卡上实际试跑**，
                        挑最快的 → 序列化成"引擎"文件
  运行期（每次推理，快）：反序列化引擎 → 创建执行上下文 → 绑定输入输出地址 → 发射执行

引擎绑定**具体的 GPU 型号与 TensorRT 版本**：它里面存的是为 sm_120、46 个 SM、这版 TRT 挑出的 kernel，
换一块卡或换一个版本就要重新构建。

精度怎么控制
------------
  fp32：关闭 TF32（TRT 默认允许矩阵乘用 TF32）——作为导出正确性的数值核对，必须是纯 FP32
  fp16：打开 FP16 标志。这是"弱类型"模式：TRT 被**允许**对每一层选 FP16 或 FP32 实现，挑最快的，
        对数值敏感的层（如归一化的累加）可能自动保留 FP32。所以"FP16 引擎"实际是混合精度
  int8：同时打开 INT8 与 FP16 标志，并给一个校准器（见 calibrate.py）。隐式量化：TRT 自己决定哪些层
        跑 INT8——只在 INT8 实现更快且该层支持时。所以"INT8 引擎"里 INT8 层有多少，要用检查器数出来

用法：
    python -m deploy.export.to_trt --precision fp32 fp16 --batch 1 32
    python -m deploy.export.to_trt --precision int8 --calib-algo entropy --calib-split day --batch 1 32 \
        --out deploy/results/trt_build_int8.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import tensorrt as trt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ENGINE_DIR = REPO / "deploy" / "results" / "engines"


def build(onnx_path: Path, precision: str, engine_path: Path, calibrator=None) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)                       # TRT 10 默认即显式 batch
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX 解析失败：\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # 让检查器能列出每一层的细节
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("int8 需要校准器")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)             # 不适合 INT8 的层退到 FP16 而不是 FP32
        config.int8_calibrator = calibrator
    else:
        raise ValueError(precision)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("引擎构建失败（见上方 TensorRT 日志）")
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    info = inspect(engine)
    info.update({"precision": precision, "onnx": onnx_path.name, "engine": engine_path.name,
                 "build_s": round(build_s, 1), "engine_mb": round(engine_path.stat().st_size / 2**20, 1),
                 "network_layers_before_opt": network.num_layers,
                 "trt_version": trt.__version__})
    return info


def inspect(engine) -> dict:
    """用引擎检查器列出优化后的每一层：数量、类型、是否被 Myelin 接管、注意力是否融合。"""
    insp = engine.create_engine_inspector()
    data = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = data.get("Layers", [])
    names = [l.get("Name", "") if isinstance(l, dict) else str(l) for l in layers]
    types = collections.Counter(l.get("LayerType", "?") for l in layers if isinstance(l, dict))
    tactics = collections.Counter((l.get("TacticName") or l.get("Tactic") or "?")
                                  for l in layers if isinstance(l, dict))
    # 每层按"输入里出现过的数据类型"归类：隐式量化下，这是看 INT8 到底落在哪些层的唯一办法
    def in_types(l):
        return {i.get("Format/Datatype", "?").split(" ")[0] for i in l.get("Inputs", [])}
    dicts = [l for l in layers if isinstance(l, dict)]
    int8_layers = [l for l in dicts if any("Int8" in t for t in in_types(l))]
    lower = [n.lower() for n in names]
    return {
        "engine_layers": len(layers),
        "layer_types": dict(types.most_common()),
        # ✏️ 第一版只匹配 "myelin"/"foreignnode"，报了 0 层；本版 TRT 的 Myelin 层名形如
        # "__myl_..." / "..._myl0_5"，要匹配 "myl"
        "myelin_layers": sum("myl" in n or "myelin" in n or "foreignnode" in n for n in lower),
        "attention_like_layers": sum(any(k in n for k in ("mha", "fmha", "attention", "flash"))
                                     for n in lower),
        "example_layer_names": names[:12],
        "top_tactics": dict(tactics.most_common(8)),
        "layers_with_int8_input": len(int8_layers),
        "int8_layers_by_type": dict(collections.Counter(l.get("LayerType", "?") for l in int8_layers)),
        "gemm_layers_by_input_type": dict(collections.Counter(
            "/".join(sorted(in_types(l))) for l in dicts if l.get("LayerType") == "gemm")),
        "mha_layers_by_input_type": dict(collections.Counter(
            "/".join(sorted(in_types(l))) for l in dicts if "mha" in l.get("Name", "").lower())),
        "int8_layer_examples": [{"name": l.get("Name"), "onnx": (l.get("Metadata") or "")[:160]}
                                for l in int8_layers[:6]],
        "device_memory_mb": round(getattr(engine, "device_memory_size_v2",
                                          getattr(engine, "device_memory_size", 0)) / 2**20, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--calib-algo", choices=["entropy", "minmax"], default="entropy")
    ap.add_argument("--calib-split", choices=["day", "night"], default="day",
                    help="校准图像：day=day_right 000-049（主方案），night=night_right 000-049（域匹配对照）")
    ap.add_argument("--out", default=str(REPO / "deploy" / "results" / "trt_build.json"))
    args = ap.parse_args()
    report = {}
    for b in args.batch:
        onnx_path = ENGINE_DIR / f"vits14_mean_224_b{b}_fp32.onnx"
        for p in args.precision:
            calib, tag = None, p
            if p == "int8":
                from deploy.export.calibrate import load_calibration_images, make_calibrator
                tag = f"int8_{args.calib_algo}_{args.calib_split}"
                cache = ENGINE_DIR / f"calib_{args.calib_algo}_{args.calib_split}.cache"
                # batch 1 真正喂数据校准并写缓存；其他 batch 只读这份缓存，保证缩放因子逐位相同
                images = load_calibration_images(args.calib_split) if b == 1 else None
                if images is None and not cache.exists():
                    raise SystemExit(f"batch {b} 需要先构建 batch 1 以生成校准缓存 {cache.name}")
                calib = make_calibrator(args.calib_algo, cache, images, (b, 3, 224, 224))
            eng = ENGINE_DIR / f"vits14_mean_224_b{b}_{tag}.engine"
            info = build(onnx_path, p, eng, calib)
            info["tag"] = tag
            report[f"{tag}_b{b}"] = info
            print(f"{p} b{b}: 构建 {info['build_s']} s | 引擎 {info['engine_mb']} MB | "
                  f"ONNX 层 {info['network_layers_before_opt']} → 引擎层 {info['engine_layers']} | "
                  f"Myelin 层 {info['myelin_layers']} | 注意力类层 {info['attention_like_layers']} | "
                  f"运行时显存 {info['device_memory_mb']} MB")
            print(f"   层类型: {info['layer_types']}")
            if p == "int8":
                print(f"   INT8 输入的层: {info['layers_with_int8_input']} {info['int8_layers_by_type']}"
                      f" | GEMM 按输入类型: {info['gemm_layers_by_input_type']}"
                      f" | MHA 按输入类型: {info['mha_layers_by_input_type']}")
            print(f"   前几层: {info['example_layer_names'][:6]}")
    out = Path(args.out).resolve()
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
