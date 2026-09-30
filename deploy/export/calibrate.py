"""阶段 4c · R5：TensorRT INT8 的校准器（隐式量化 + 训练后校准）。

INT8 为什么需要校准
-------------------
INT8 只能表示 -128..127 这 256 个整数。要用它表示一个浮点张量，得先选一个**缩放因子** s：
    x_int8 = round(x / s)，截断到 [-128, 127]；用的时候再乘回来 x ≈ x_int8 · s
s 取决于这个张量的数值范围：范围 [-6, 6] 就取 s = 6/127。
权重的范围建库时就知道；**激活**（每层的中间结果）的范围随输入变化，只能拿一批有代表性的图片
真实跑一遍、统计出来——这一步叫**校准**（calibration）。

两种校准算法（单变量对照）
--------------------------
  entropy  IInt8EntropyCalibrator2：选一个截断阈值，让量化前后的分布差异（KL 散度）最小。
           会**主动截掉**少数离群的大值，换取大多数值的分辨率。TensorRT 推荐的默认
  minmax   IInt8MinMaxCalibrator：直接用观察到的最大绝对值，不截断。离群值越大，分辨率越粗

校准集必须和测试集分开（否则就是用测试数据调了量化参数，一种泄漏）。本项目用 day_right 000–049，
与测试段 night_right 050–099 不同 traversal 且帧号不重叠；预处理与推理逐位一致（get_default_transform）。

校准缓存：TensorRT 把统计出的每个张量的缩放因子写成一段文本（calibration cache）。
batch 1 的引擎校准一次；batch 32 的引擎**直接读同一份缓存**——缩放因子与 batch 无关，
这样两个引擎的量化参数逐位相同，比较它们时只差 batch 一个变量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dataset import ImageFolderDataset, get_default_transform   # noqa: E402
from src.evaluate import image_index_from_path                     # noqa: E402

SPLITS = {"day": "data/gardens_point/query/day_right",
          "night": "data/gardens_point/query/night_right"}
BASES = {"entropy": trt.IInt8EntropyCalibrator2, "minmax": trt.IInt8MinMaxCalibrator}


def load_calibration_images(split: str, lo: int = 0, hi: int = 49, res: int = 224) -> list[torch.Tensor]:
    """校准图像：与推理完全相同的预处理，按帧号区间筛选，每张 [1, 3, res, res]。"""
    ds = ImageFolderDataset(REPO / SPLITS[split], transform=get_default_transform(res))
    keep = [i for i, p in enumerate(ds.image_paths) if lo <= image_index_from_path(str(p)) <= hi]
    return [ds[i][0].unsqueeze(0) for i in keep]


def make_calibrator(algo: str, cache_path: Path, images: list[torch.Tensor] | None = None,
                    input_shape: tuple[int, ...] = (1, 3, 224, 224)):
    """返回一个 TensorRT 校准器。

    images 为 None 时不再喂数据，只从 cache_path 读已有的缩放因子（用于 batch 32 复用 batch 1 的校准）。
    TensorRT 的校准器要求继承它的基类，而基类（算法）要到运行时才确定，所以在函数里定义类。
    """
    base = BASES[algo]

    class _Calibrator(base):
        def __init__(self):
            base.__init__(self)                      # pybind 基类必须显式初始化
            self.images = images or []
            self.i = 0
            self.buf = torch.empty(input_shape, device="cuda", dtype=torch.float32)
            self.cache_path = Path(cache_path)

        def get_batch_size(self) -> int:
            return input_shape[0]

        def get_batch(self, names):                  # noqa: ARG002 — 只有一个输入
            if self.i >= len(self.images):
                return None                          # 返回 None = 数据喂完了
            self.buf.copy_(self.images[self.i])
            torch.cuda.synchronize()                 # TensorRT 在自己的 stream 上读，先确保拷贝完成
            self.i += 1
            return [self.buf.data_ptr()]

        def read_calibration_cache(self):
            # 喂数据模式下故意不读旧缓存：每次都重新统计，结果才由本次的校准集决定
            if images is None and self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(bytes(cache))

    return _Calibrator()
