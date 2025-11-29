import torch
class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)


example_inputs = (torch.randn(1, 5),)
m = M().eval()

m = torch.export.export(m, example_inputs).module()
# we get a model with aten ops


# Step 2. quantization
from torchao.quantization.pt2e.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  get_symmetric_quantization_config,
  XNNPACKQuantizer,
)
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)

# calibration omitted

m2 = convert_pt2e(m)

quantized_ep = torch.export.export(m2, example_inputs)
torch.export.save(quantized_ep, '121.pth')
from torch_mlir import fx

module = fx.export_and_import(
    m2,
    torch.ones(1, 5),
    func_name=m2.__class__.__name__,
)

with open("x21.mlir", "w") as f:
    f.write(str(module))