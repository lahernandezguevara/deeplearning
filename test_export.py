import sys
print("Importing modules", flush=True)
import torch
import onnx
import onnxruntime as ort
from models.factory import create_model

print("Creating model", flush=True)
model = create_model("vgg11", num_classes=2)

print("Loading state dict", flush=True)
state_dict = torch.load("artifacts/best_model.tbh", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

print("Model loaded. Eval mode.", flush=True)
model.eval()
dummy_input = torch.randn(8, 3, 224, 224)

print("Exporting to ONNX", flush=True)
torch.onnx.export(model, dummy_input, "serving/model_repository/cnn/1/model.onnx", input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print("Loading ONNX", flush=True)
onnx_model = onnx.load("serving/model_repository/cnn/1/model.onnx")

print("Checking ONNX model", flush=True)
onnx.checker.check_model(onnx_model)

print("Done", flush=True)
