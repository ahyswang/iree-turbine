import torch
import torch.nn as nn

from torch_mlir.dialects.torch import ops as torch_ops

import iree.runtime as rt
import logging
import numpy as np
import torch
import torch.nn as nn
import unittest

import iree.turbine.aot as aot

class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        # 量化
        scale = 0.1
        zero_point = 128
        dtype = torch.quint8
        x_quant = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        x_dequant = x_quant.dequantize()

        out = self.fc(x_dequant)

        out_quant = torch.quantize_per_tensor(out, scale=scale, zero_point=zero_point, dtype=dtype)
        out_dequant = out_quant.dequantize()

        return out_dequant

# 测试模型
model = QuantizedModel()
x = torch.randn(4, 4)
output = model(x)
print("Output:", output)

model = QuantizedModel()
example_x = torch.empty(4, 4, dtype=torch.float32)
exported = aot.export(model, example_x)
exported.print_readable()
compiled_binary = exported.compile(save_to=None)

