"""TensorRT 引擎的执行器：用 PyTorch 张量做输入输出缓冲，直接绑定显存地址，不经过 CPU 拷贝。

TRT 10 的执行接口是"按名字绑定地址"：
  context.set_tensor_address(名字, 显存地址)   告诉引擎输入/输出在显存的哪里
  context.execute_async_v3(cuda stream)        把整个引擎的所有 kernel 发射到这个 stream 上

这里用 PyTorch 当前的 stream，所以 TRT 的 kernel 与 PyTorch 的计时（CUDA Event）在同一条队列里，
计时方式与 4a 完全一致。
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

_TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16,
                 trt.bfloat16: torch.bfloat16, trt.int8: torch.int8, trt.int32: torch.int32}


class TRTRunner:
    def __init__(self, engine_path: str | Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"引擎反序列化失败：{engine_path}（TRT 版本或 GPU 不匹配？）")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.inp = next(n for n in names if mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.inp)]
        self.in_shape = tuple(self.engine.get_tensor_shape(self.inp))
        out_shape = tuple(self.engine.get_tensor_shape(self.out))
        self.out_buf = torch.empty(out_shape, device="cuda",
                                   dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype(self.out)])
        self.ctx.set_tensor_address(self.out, self.out_buf.data_ptr())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """注意：返回的是内部输出缓冲，下一次调用会覆盖它；需要保留时请 .clone()。"""
        if tuple(x.shape) != self.in_shape:
            raise ValueError(f"引擎按固定形状 {self.in_shape} 构建，收到 {tuple(x.shape)}")
        x = x.to(self.in_dtype).contiguous()
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 返回失败")
        self._keepalive = x          # 保证输入在 GPU 读完之前不被释放
        return self.out_buf
