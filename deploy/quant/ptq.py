"""阶段 5：用 ModelOpt 做显式训练后量化（PTQ），按组启用量化器。

显式 vs 隐式（对照 4c）
-----------------------
4c 用 TensorRT 的隐式量化：打开 INT8 开关，TensorRT 自己决定哪些层跑 INT8——结果把所有矩阵乘和
融合注意力都量化了，R@1 从 0.933 崩到 0.633。显式量化反过来：**我们**决定哪些张量量化。
ModelOpt 的做法是给模型里的 Linear / Conv / LayerNorm 插入"量化器"（TensorQuantizer）：
  * 在 PyTorch 里它做**伪量化**——数值仍是浮点，但被取整到 INT8 能表示的格点上，
    所以可以直接用 PyTorch 测"这样量化会损失多少精度"
  * 导出 ONNX 时它变成 QuantizeLinear / DequantizeLinear（Q/DQ）节点，TensorRT 严格照着建 INT8 引擎

量化器的名字 = 模块名 + ".input_quantizer" / ".weight_quantizer"，例如 `blocks.3.mlp.fc1.input_quantizer`。
ModelOpt 的配置是一串**有序规则**（按名字通配符匹配，后面的规则覆盖前面的），这里据此拼出每组的配置。

分组（只启用这一组，其余全部 FP32）：见 GROUPS。权重逐输出通道（axis=0），激活逐张量（axis=None）——
与 ModelOpt 的 INT8 默认配方一致，也是 TensorRT INT8 GEMM 支持的形式。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import modelopt.torch.quantization as mtq                         # noqa: E402

from deploy.eval.runtimes import _base_fp32                       # noqa: E402
from deploy.export.calibrate import load_calibration_images       # noqa: E402

W = {"num_bits": 8, "axis": 0}        # 权重：逐输出通道
A = {"num_bits": 8, "axis": None}     # 激活：逐张量

# 阶段 6：同一套分组，只换数值格式。与 ModelOpt 的 FP8_DEFAULT_CFG / NVFP4_DEFAULT_CFG 一致
#   fp8   E4M3（4 位指数、3 位尾数），权重与激活都逐张量一个缩放因子
#   nvfp4 E2M1（2 位指数、1 位尾数），每 16 个数一个 FP8 缩放因子（按块、动态）
FORMATS = {
    "int8": {"W": W, "A": A},
    "fp8": {"W": {"num_bits": (4, 3), "axis": None}, "A": {"num_bits": (4, 3), "axis": None}},
    # 单变量对照：FP8 权重改成逐输出通道（与 INT8 组一致），激活仍逐张量
    "fp8pc": {"W": {"num_bits": (4, 3), "axis": 0}, "A": {"num_bits": (4, 3), "axis": None}},
    "nvfp4": {"W": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
              "A": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}},
}

_LINEAR = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

# 组名 -> [(量化器名通配符, 配置)]
GROUPS: dict[str, list[tuple[str, dict]]] = {
    "none": [],
    "weights": [(f"*{l}.weight_quantizer", W) for l in _LINEAR],
    "qkv": [("*attn.qkv.input_quantizer", A), ("*attn.qkv.weight_quantizer", W)],
    "proj": [("*attn.proj.input_quantizer", A), ("*attn.proj.weight_quantizer", W)],
    "fc1": [("*mlp.fc1.input_quantizer", A), ("*mlp.fc1.weight_quantizer", W)],
    "fc2": [("*mlp.fc2.input_quantizer", A), ("*mlp.fc2.weight_quantizer", W)],
    "linear": [(f"*{l}.{k}_quantizer", A if k == "input" else W)
               for l in _LINEAR for k in ("input", "weight")],
    "linear_in": [(f"*{l}.input_quantizer", A) for l in _LINEAR],     # 只量化 Linear 的输入（激活），权重保持全精度
    "ln_in": [("*norm*.input_quantizer", A)],
    "conv": [("patch_embed.proj.input_quantizer", A), ("patch_embed.proj.weight_quantizer", W)],
}


def group_patterns(group: str) -> list[tuple[str, dict]]:
    """组名语法：
      qkv            单个组
      qkv,proj,fc1   几个组的并集
      fc2@3          只作用于第 3 个 Block（把通配符开头的 `*` 换成 `blocks.3.`）
    """
    out: list[tuple[str, dict]] = []
    for part in group.split(","):
        name, _, block = part.partition("@")
        for pat, c in GROUPS[name]:
            out.append((f"blocks.{block}.{pat.lstrip('*')}" if block else pat, c))
    return out


def group_config(group: str, algorithm: str = "max", fmt: str = "int8") -> dict:
    """拼出 ModelOpt 配置：先全部关闭，再只打开这一组。`default` 直接用 ModelOpt 的 INT8 默认配方。

    fmt 只替换数值格式：GROUPS 里的 W / A 换成 FORMATS[fmt] 里对应的配置，分组本身不变。
    """
    if group == "default":
        cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        cfg["algorithm"] = algorithm
        return cfg
    swap = {id(W): FORMATS[fmt]["W"], id(A): FORMATS[fmt]["A"]}
    rules = [{"quantizer_name": "*", "enable": False}]
    rules += [{"quantizer_name": pat, "cfg": copy.deepcopy(swap.get(id(c), c))}
              for pat, c in group_patterns(group)]
    return {"quant_cfg": rules, "algorithm": algorithm}


def calibration_loop(split: str = "day"):
    """校准：把 50 张校准图逐张跑一遍前向，量化器在这期间记录每个张量的范围。"""
    images = load_calibration_images(split)

    def loop(model):
        with torch.no_grad():
            for x in images:
                model(x.cuda())
    return loop


def build_quantized(group: str, algorithm: str = "max", split: str = "day", fmt: str = "int8"):
    """返回 (伪量化后的 FP32 模型, 启用的量化器个数)。每次从一个新的 FP32 基座深拷贝，组与组之间互不影响。"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = copy.deepcopy(_base_fp32()).eval()
    model = mtq.quantize(model, group_config(group, algorithm, fmt), forward_loop=calibration_loop(split))
    n_on = sum(1 for n, m in model.named_modules()
               if n.endswith("quantizer") and getattr(m, "is_enabled", False))
    return model.eval(), n_on
