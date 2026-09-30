"""阶段 5 ①：逐组量化敏感度（PyTorch 伪量化，不建 TensorRT 引擎）。

每组：从 FP32 基座深拷贝 → 只启用这一组量化器 → 用 day_right 000–049 校准 → batch 1 抽 300 张图的特征 →
与 FP32 参考比（余弦、top-1 翻转、最大分数漂移、冻结门控上的决策翻转）。
`none` 组是 sanity check：量化器全关，输出必须与 FP32 逐位相同。

用法：
    python -m deploy.quant.sensitivity                      # 全部组
    python -m deploy.quant.sensitivity qkv fc2 linear+sq    # 指定组；组名加 "+sq" 表示用 SmoothQuant 校准
    python -m deploy.quant.sensitivity fp8:linear nvfp4:fc2 # 前缀 "格式:" 换数值格式（int8 / fp8 / nvfp4，默认 int8）
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.eval.compare_features import compare                  # noqa: E402
from deploy.eval.extract_with_runtime import extract              # noqa: E402
from deploy.quant.ptq import GROUPS, build_quantized              # noqa: E402

warnings.filterwarnings("ignore")
OUT = REPO / "deploy" / "results" / "runtime"
DEFAULT_RUN = ["none", "weights", "proj", "qkv", "fc1", "fc2", "conv", "ln_in", "linear", "default", "linear+sq"]


def run(name: str) -> dict:
    fmt, _, body = name.rpartition(":")
    fmt = fmt or "int8"
    group, algo = (body[:-3], "smoothquant") if body.endswith("+sq") else (body, "max")
    assert group == "default" or all(g.partition("@")[0] in GROUPS for g in group.split(",")), name
    t0 = time.perf_counter()
    model, n_on = build_quantized(group, algo, fmt=fmt)
    prefix = OUT / f"ptq_{name.replace('+', '_').replace(',', '-').replace('@', 'b').replace(':', '_')}"
    extract(lambda x: model(x), {"kind": "pytorch-fakequant", "format": fmt, "group": group, "algorithm": algo,
                                 "calibration": "day_right 000-049"}, prefix, batch_size=1)
    r = compare(prefix)
    r["enabled_quantizers"] = n_on
    r["seconds"] = round(time.perf_counter() - t0, 1)
    del model
    torch.cuda.empty_cache()
    return r


def main() -> None:
    names = sys.argv[1:] or DEFAULT_RUN
    path = REPO / "deploy" / "results" / "ptq_sensitivity.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    print(f"{'组':<11}{'量化器':>6}{'1−cos':>10}{'最小cos':>9}{'top-1翻转':>10}{'最大漂移':>10}{'决策翻转':>9}")
    for name in names:
        r = run(name)
        results[name] = r
        c, a, g = r["cosine"], r["retrieval_all"], r["frozen_gate_test"]
        print(f"{name:<11}{r['enabled_quantizers']:>6}{c['one_minus_mean']:>10.2e}{c['min']:>9.4f}"
              f"{a['top1_flips']:>7}/200{a['score_shift_max_abs']:>10.4f}{g['decision_flips']:>6}/50", flush=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {path.relative_to(REPO)}（决策余量 = {g['closest_margin_to_threshold']:.4f}）")


if __name__ == "__main__":
    main()
