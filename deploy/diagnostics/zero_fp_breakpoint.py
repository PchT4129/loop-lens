"""阶段 5 · 零误报召回为什么从 0.900 掉到 0.100。

held-out（night_right 070–099，30 个查询，库不删）上，把每个查询的 Top-1 按分数从高到低排，
找出第一个错误出现在第几名——零误报召回（R@100%P）= 它前面有几个 / 30。
逐个列出错误的查询，看是"很多查询变错"还是"一个高分查询变错"。

结论（EXPERIMENTS.md 阶段 5）：显式 INT8 引擎只多错了一个查询（76 → 库 72，差 4 帧，恰好比 ±3 容差多 1 帧），
但它的分数排第 4，于是 R@100%P 在第 4 名就断了。

用法：
    python -m deploy.diagnostics.zero_fp_breakpoint [特征前缀 ...]
"""

import re
import sys

from deploy.diagnostics._common import REPO, save

import torch

TOL = 3


def _idx(p: str) -> int:
    return int(re.search(r"Image(\d+)\.jpg", p).group(1))


def ranked(prefix: str) -> list[dict]:
    d = {s: torch.load(REPO / f"{prefix}_{s}.pt", weights_only=False) for s in ("database", "query")}
    qp, dp = d["query"]["paths"], d["database"]["paths"]
    keep = [i for i, p in enumerate(qp) if "night_right" in p and 70 <= _idx(p) <= 99]
    v, j = (d["query"]["features"][keep] @ d["database"]["features"].T).max(1)
    rows = [{"score": round(float(v[k]), 4), "query": _idx(qp[i]), "db": _idx(dp[j[k]]),
             "correct": abs(_idx(qp[i]) - _idx(dp[j[k]])) <= TOL} for k, i in enumerate(keep)]
    return sorted(rows, key=lambda r: -r["score"])


def main() -> None:
    prefixes = sys.argv[1:] or ["deploy/results/ref/fp32", "deploy/results/runtime/trt_fp16",
                                "deploy/results/runtime/trt_qdq_lin_nofc2_sq"]
    out = {}
    for p in prefixes:
        rows = ranked(p)
        first = next((n for n, r in enumerate(rows) if not r["correct"]), len(rows))
        wrong = [dict(r, rank=n + 1) for n, r in enumerate(rows) if not r["correct"]]
        out[p] = {"recall_at_100p": round(first / len(rows), 3), "first_wrong_rank": first + 1, "wrong": wrong}
        print(f"{p:<48} R@100%P {first}/{len(rows)} = {first / len(rows):.3f} | 错误: " +
              ", ".join(f"q{w['query']}→db{w['db']}（{w['score']:.3f}，第{w['rank']}名）" for w in wrong))
    save("zero_fp_breakpoint", out)


if __name__ == "__main__":
    main()
