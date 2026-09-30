"""用任意"执行方式"（runtime）抽特征，输出与仓库原有 {features, paths, meta} 契约完全一致。

为什么要这一层
--------------
阶段 4 的每一档（PyTorch 半精度、torch.compile、TensorRT FP16/INT8……）都是"同一个模型，
换一种执行方式"。只要每种执行方式都写出和 src.extract_features **格式一致**的特征文件，
src/evaluate.py 就能一行不改地评测它们——这是整个方案的枢纽。

meta 与 src.extract_features 逐键一致，只多一个 meta["runtime"] 子字典，记录执行方式。
评测时用 --allow-runtime-mismatch 只放开这一个键（见 src/evaluate.py::pipelines_match）。

两个统一的约定
--------------
* **batch = 1 抽取**：部署场景是逐帧处理。而且 batch 大小会改变 kernel 选择（阶段 3：batch 1
  时会用 split-K / split-KV，求和顺序不同），数值会有微小差异。所以精度一律按部署的形态测。
* 预处理一律用 src.dataset.get_default_transform —— 与 FP32 参考逐位相同的输入。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dataset import ImageFolderDataset, get_default_transform   # noqa: E402

# 必须与 src.extract_features 为 dinov2_vits14 / mean / 224 写出的 meta 逐键一致
BASE_META = {"backbone": "dinov2_vits14", "pooling": "mean", "gem_p": None,
             "image_size": 224, "checkpoint": None, "feature_dim": 384}


def extract(runtime_fn: Callable[[torch.Tensor], torch.Tensor], runtime_meta: dict,
            out_prefix: str | Path, batch_size: int = 1) -> dict[str, Path]:
    """runtime_fn 接收 FP32、已归一化的 [B,3,224,224] CUDA 张量，返回 [B,384] 描述子。

    转成哪种精度、用哪个引擎，是 runtime_fn 自己的事；这里只负责数据和格式。
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ("database", "query"):
        # 相对路径 root 与生成 FP32 参考时一致，保证 paths 字符串逐字相同
        ds = ImageFolderDataset(f"data/gardens_point/{split}",
                                transform=get_default_transform(BASE_META["image_size"]))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        feats, paths = [], []
        with torch.inference_mode():
            for x, p in loader:
                feats.append(runtime_fn(x.cuda()).float().cpu())
                paths.extend(p)
        features = torch.cat(feats)
        if features.shape[1] != BASE_META["feature_dim"]:
            raise ValueError(f"特征维度 {features.shape[1]} ≠ {BASE_META['feature_dim']}")
        path = out_prefix.parent / f"{out_prefix.name}_{split}.pt"
        torch.save({"features": features, "paths": paths,
                    "meta": {**BASE_META, "runtime": runtime_meta}}, path)
        written[split] = path
    return written
