"""特征级对比：某种执行方式抽出的特征，相对 FP32 参考偏了多少、会不会改变决策。

为什么需要它（在 src/evaluate.py 之外）
--------------------------------------
测试段只有 50 个查询，deployed F1 的 95% 置信区间宽达 18 个点（[0.747, 0.926]）。
小的精度损失在这个区间里根本分辨不出来。所以除了跑原有协议，还要看更灵敏的特征级指标：

  余弦相似度   每张图的新特征与 FP32 参考特征有多像。噪声底约 2e-7（阶段 2）
  top-1 翻转   用新特征检索，最相似的库图像是否换了一张
  分数漂移     top-1 的相似度分数变了多少——冻结门控是按分数卡阈值的
  ⭐决策翻转   在冻结门控的测试段上，同一个查询"接受/拒绝"的判定是否改变。
               这是与部署后果直接对应的量：一次翻转 = 一次多出来的误报或漏报

⚠️ 这里只是**诊断**，不是精度结论。精度结论一律以 src/evaluate.py 的 deployed F1 为准。
决策翻转用到的测试段切分（night_right 050-099，库排除 070-079）与协议第 4 步相同。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "deploy" / "results"
REF_PREFIX = RESULTS / "ref" / "fp32"
FROZEN_GATE = RESULTS / "protocol" / "ref_fp32" / "gate_similarity.json"


def _idx(path: str) -> int:
    return int(re.search(r"Image(\d+)\.jpg", path).group(1))


def _load(prefix: Path, split: str) -> dict:
    return torch.load(f"{prefix}_{split}.pt", map_location="cpu", weights_only=False)


def compare(cand_prefix: Path, ref_prefix: Path = REF_PREFIX) -> dict:
    out: dict = {}
    ref = {s: _load(ref_prefix, s) for s in ("database", "query")}
    cand = {s: _load(cand_prefix, s) for s in ("database", "query")}
    for s in ("database", "query"):
        assert ref[s]["paths"] == cand[s]["paths"], f"{s} 的路径顺序与参考不一致"

    # 1) 逐图余弦
    cos = torch.cat([F.cosine_similarity(ref[s]["features"], cand[s]["features"], dim=1)
                     for s in ("database", "query")])
    out["cosine"] = {"mean": float(cos.mean()), "min": float(cos.min()),
                     "p01": float(torch.quantile(cos, 0.01)),
                     "one_minus_mean": float(1 - cos.mean())}

    # 2) 全量检索：200 个查询 × 100 张库图
    def top1(q, d):
        sim = q @ d.T
        v, i = sim.max(dim=1)
        return v, i

    rv, ri = top1(ref["query"]["features"], ref["database"]["features"])
    cv, ci = top1(cand["query"]["features"], cand["database"]["features"])
    out["retrieval_all"] = {"queries": len(ri), "top1_flips": int((ri != ci).sum()),
                            "score_shift_mean_abs": float((rv - cv).abs().mean()),
                            "score_shift_max_abs": float((rv - cv).abs().max())}

    # 3) 冻结门控测试段上的决策翻转（与协议第 4 步同一切分）
    thr = json.loads(FROZEN_GATE.read_text(encoding="utf-8"))["sequence_threshold"]
    qp, dp = ref["query"]["paths"], ref["database"]["paths"]
    qmask = torch.tensor(["night_right" in p and 50 <= _idx(p) <= 99 for p in qp])
    dmask = torch.tensor([not (70 <= _idx(p) <= 79) for p in dp])
    rq, rd = ref["query"]["features"][qmask], ref["database"]["features"][dmask]
    cq, cd = cand["query"]["features"][qmask], cand["database"]["features"][dmask]
    rv2, ri2 = top1(rq, rd)
    cv2, ci2 = top1(cq, cd)
    ref_acc, cand_acc = rv2 >= thr, cv2 >= thr
    margin = (rv2 - thr).abs()
    out["frozen_gate_test"] = {
        "queries": int(qmask.sum()), "threshold": thr,
        "decision_flips": int((ref_acc != cand_acc).sum()),
        "top1_flips": int((ri2 != ci2).sum()),
        "score_shift_mean_abs": float((rv2 - cv2).abs().mean()),
        # 离阈值最近的那个查询有多近——量化带来的分数漂移若超过它，就可能翻转
        "closest_margin_to_threshold": float(margin.min()),
    }
    return out


def fmt(name: str, r: dict) -> str:
    c, a, g = r["cosine"], r["retrieval_all"], r["frozen_gate_test"]
    return (f"{name:<22} 余弦 均值 1-{c['one_minus_mean']:.1e} 最小 {c['min']:.6f} | "
            f"top-1 翻转 {a['top1_flips']}/{a['queries']} 分数漂移 均 {a['score_shift_mean_abs']:.1e} "
            f"最大 {a['score_shift_max_abs']:.1e} | 门控测试段 决策翻转 {g['decision_flips']}/{g['queries']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+", help="特征前缀，如 deploy/results/runtime/torch_bf16")
    ap.add_argument("--ref", default=str(REF_PREFIX))
    ap.add_argument("--out", default=None, help="把全部结果写成一个 JSON")
    args = ap.parse_args()
    allres = {}
    for c in args.candidates:
        r = compare(Path(c), Path(args.ref))
        allres[Path(c).name] = r
        print(fmt(Path(c).name, r))
    if args.out:
        Path(args.out).write_text(json.dumps(allres, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
