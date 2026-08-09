from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform: Callable | None = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = self._find_images(self.root_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)

    @staticmethod
    def _find_images(root_dir: Path) -> list[Path]:
        image_paths = [
            path
            for path in root_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(image_paths)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transform(image_size: int = 224):
    """推理/建库用的确定性变换。抽特征时必须用它，不能带随机增强。"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transform(image_size: int = 224, jitter: float = 0.4):
    """训练用的增强变换。

    ColorJitter 是针对昼夜任务最对症的增强——它模拟的正是光照/色温变化。

    注意这里【刻意不加水平翻转】：VPR 里"左边有楼、右边有树"这种左右空间关系
    本身就是区分地点的线索，翻转会破坏它。分类任务可以随便翻，VPR 不行。
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(0.1, jitter / 4),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )