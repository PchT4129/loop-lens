"""诊断脚本的公共部分。

为什么单独一个 diagnostics/ 目录：这些脚本各自回答台账里的一个具体问题（"显存溢出慢多少"
"插值为什么慢"……），不是流水线的一部分。它们最初是草稿脚本或内联命令，结论写进了台账，
但代码不在仓库里——别人无法复现。补进来之后，台账里每一条结论都有对应的生成代码和产物文件。
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "deploy" / "results" / "diagnostics"


def save(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {path.relative_to(REPO)}")
    return path
