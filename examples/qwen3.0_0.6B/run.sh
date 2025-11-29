source /workspace/yswang26/iree-turbine/.venv/bin/activate

save_dir=./data.ignore/
mkdir -p $save_dir
input_mlir=./data.ignore/qwen3-0.6b_dyn.mlir

# iree-compile $input_mlir \
# --iree-hal-target-device=local \
# --iree-hal-local-target-device-backends=llvm-cpu \
# --iree-llvmcpu-target-cpu=generic   \
# --iree-llvmcpu-link-embedded=false \
# --dump-compilation-phases-to=$save_dir \
# -o $save_dir/qwen3-0.6b_dyn.mlir.vmfb \
# --mlir-print-ir-after-all \
# --mlir-print-ir-after-change \
# --mlir-print-ir-before-all \
# > $save_dir/log.txt 2>&1

python qwen3-0.6b.py

iree-compile $input_mlir \
--iree-hal-target-device=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-cpu=generic   \
--iree-llvmcpu-link-embedded=false \
--dump-compilation-phases-to=$save_dir \
--iree-opt-export-parameter-minimum-size=1024 \
--iree-opt-export-parameters=./data.ignore/qwen3-0.6b_dyn.mlir.irpa \
-o ./data.ignore/qwen3-0.6b_dyn.mlir.vmfb

iree-import-onnx ./data.ignore/qwen3-0.6b.onnx -o ./data.ignore/qwen3-0.6b.onnx.mlir --large-model --externalize-params

iree-compile ./data.ignore/qwen3-0.6b.onnx.mlir \
--iree-hal-target-device=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-cpu=generic   \
--iree-llvmcpu-link-embedded=false \
--dump-compilation-phases-to=$save_dir \
--iree-opt-export-parameter-minimum-size=1024 \
--iree-opt-export-parameters=./data.ignore/qwen3-0.6b.onnx.mlir.irpa \
-o ./data.ignore/qwen3-0.6b.onnx.mlir.vmfb