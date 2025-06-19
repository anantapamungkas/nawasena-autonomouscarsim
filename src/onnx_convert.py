import torch
from ultralytics import YOLO
import subprocess
import os

# 1. Load model
model_path = 'firayolov11n.pt'  # ganti sesuai lokasi best.pt kamu
model = YOLO(model_path)

# 2. Export ke ONNX (FP32)
model.export(format='onnx', dynamic=True, simplify=True, opset=12)

onnx_path = model_path.replace('.pt', '.onnx')
print(f"✅ ONNX model saved to {onnx_path}")

# 3. Konversi ke TensorRT menggunakan trtexec
tensorrt_engine = onnx_path.replace('.onnx', '_fp32.engine')

command = f"""
trtexec --onnx={onnx_path} --saveEngine={tensorrt_engine} --fp32 --workspace=4096
"""
print("🔄 Menjalankan TensorRT FP32 conversion...")
subprocess.run(command, shell=True, check=True)

print(f"✅ TensorRT engine disimpan di: {tensorrt_engine}")
