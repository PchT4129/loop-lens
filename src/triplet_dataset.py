import random
import re
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.dataset import IMAGE_EXTENSIONS, get_train_transform


def image_index_from_path(path: Path) -> int:
    match = re.search(r"Image(\d+)\.jpg$", path.name)
    if match is None:
        raise ValueError(f"Could not parse image index from path: {path}")

    return int(match.group(1))


def find_images(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)

    image_paths = [
        path
        for path in root_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def filter_by_index(
    paths: list[Path],
    min_index: int | None = None,
    max_index: int | None = None,
) -> list[Path]:
    """按图像帧号区间过滤。min/max 均为 None 时原样返回。"""
    return [
        path
        for path in paths
        if (min_index is None or image_index_from_path(path) >= min_index)
        and (max_index is None or image_index_from_path(path) <= max_index)
    ]


class TripletPlaceDataset(Dataset):
    def __init__(
        self,
        anchor_dir: str | Path,
        database_dir: str | Path,
        transform: Callable | None = None,
        positive_tolerance: int = 3,
        negative_gap: int = 20,
        min_index: int | None = None,
        max_index: int | None = None,
        db_min_index: int | None = None,
        db_max_index: int | None = None,
    ):
        self.anchor_paths = filter_by_index(
            find_images(anchor_dir), min_index, max_index
        )

        # database 也支持区间过滤。不过滤时（默认）训练会碰到全部 database 图像，
        # 包括测试区间的那些——它们会被当作负样本，边界处甚至会被当作正样本。
        # 传 db_max_index 可以留出一段 purged gap，避免这种边界泄漏。
        self.database_paths = filter_by_index(
            find_images(database_dir), db_min_index, db_max_index
        )

        self.transform = transform or get_train_transform()
        self.positive_tolerance = positive_tolerance
        self.negative_gap = negative_gap

        if len(self.anchor_paths) == 0:
            raise ValueError(f"No anchor images found in {anchor_dir}")
        if len(self.database_paths) == 0:
            raise ValueError(f"No database images found in {database_dir}")

        self.database_indices = {
            path: image_index_from_path(path)
            for path in self.database_paths
        }

    def __len__(self) -> int:
        return len(self.anchor_paths)

    def __getitem__(self, index: int):
        anchor_path = self.anchor_paths[index]
        anchor_index = image_index_from_path(anchor_path)

        positive_candidates = [
            path
            for path in self.database_paths
            if abs(self.database_indices[path] - anchor_index) <= self.positive_tolerance
        ]

        negative_candidates = [
            path
            for path in self.database_paths
            if abs(self.database_indices[path] - anchor_index) >= self.negative_gap
        ]

        if len(positive_candidates) == 0:
            raise ValueError(f"No positive candidates found for {anchor_path}")
        if len(negative_candidates) == 0:
            raise ValueError(f"No negative candidates found for {anchor_path}")

        positive_path = random.choice(positive_candidates)
        negative_path = random.choice(negative_candidates)

        anchor_image = self._load_image(anchor_path)
        positive_image = self._load_image(positive_path)
        negative_image = self._load_image(negative_path)

        return {
            "anchor": anchor_image,
            "positive": positive_image,
            "negative": negative_image,
            "anchor_path": str(anchor_path),
            "positive_path": str(positive_path),
            "negative_path": str(negative_path),
        }

    def _load_image(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)