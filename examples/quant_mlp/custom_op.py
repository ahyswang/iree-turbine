
import iree.runtime as rt
import logging
import numpy as np
import torch
import torch.nn as nn
import unittest

import iree.turbine.aot as aot

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        out = self.fc(x)
        return out


class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        from iree.turbine.ops import _str_format_test_ops as test_ops
        out = self.fc(x)
        out = test_ops.test_add(out, out)
        return out
    
class CustomMLPDyn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        from iree.turbine.ops import _str_format_test_ops as test_ops
        out = self.fc(x)
        out = test_ops.test_add_dyn(out, out)
        return out
    
def test_model():
    # 测试模型
    model = MLP()
    x = torch.randn(4, 4)
    output = model(x)
    print("Output:", output)

    model = MLP()
    example_x = torch.empty(4, 4, dtype=torch.float32)
    exported = aot.export(model, example_x)
    exported.print_readable()
    compiled_binary = exported.compile(save_to=None)

def test_custom():
    from iree.turbine.ops import _str_format_test_ops as test_ops
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
    result = test_ops.test_add(t, t)
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], device="cpu")
    torch.testing.assert_close(result, expected)

def test_custom_model():
    # 测试模型
    model = CustomMLP()
    x = torch.randn(4, 4)
    output = model(x)
    print("Output:", output)

    print("Testing static export and compile...")
    model = CustomMLP()
    example_x = torch.empty(4, 4, dtype=torch.float32)
    exported = aot.export(model, example_x)
    exported.print_readable()
    compiled_binary = exported.compile(save_to=None)

def test_custom_model_dyn():
    # 测试模型
    model = CustomMLPDyn()
    x = torch.randn(4, 4)
    output = model(x)
    print("Output:", output)

    print("Testing dynamic export and compile...")
    example_x = torch.empty(4, 4, dtype=torch.float32)
    batch = torch.export.Dim("batch")
    #exported = aot.export(model, example_x)
    exported = aot.export(model, example_x , dynamic_shapes={'x':{0: batch}})
    exported.print_readable()
    compiled_binary = exported.compile(save_to=None)


if __name__ == "__main__":
    # test_model()
    # test_custom()
    # test_custom_model()
    test_custom_model_dyn()



