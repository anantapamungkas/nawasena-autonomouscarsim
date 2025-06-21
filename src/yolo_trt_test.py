'''
@ 2025, Copyright AVIS Engine
- Tes YOLOv11m TensorRT di AVIS Engine
'''

import avisengine
import config
import numpy as np
import cv2
import time
import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # otomatis inisialisasi CUDA

CONFIDENCE_THRESHOLD = 0.5
INPUT_SHAPE = (1, 3, 640, 640)

# Inisialisasi AVIS Engine
car = avisengine.Car()

try:
    car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)
    print(f"✅ Terhubung ke simulator di {config.SIMULATOR_IP}:{config.SIMULATOR_PORT}")
except Exception as e:
    print(f"❌ Gagal terhubung ke simulator: {e}")
    sys.exit(1)

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open("firayolov11m.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

# Alokasi memori CUDA
input_binding_idx = engine.get_binding_index("images")
output_binding_idx = 1  # sesuaikan jika output bukan di index ke-1

# Tentukan ukuran input dan output
input_shape = INPUT_SHAPE
output_shape = (1, 8400, 85)  # sesuaikan dengan output modelmu

# Alokasi memori host dan device
d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().nbytes)
bindings = [int(d_input), int(d_output)]

# Untuk hasil
host_output = np.empty(output_shape, dtype=np.float32)

# FPS
prev_time = time.time()
counter = 0

# Fungsi preprocessing
def preprocess(image):
    image = cv2.resize(image, (640, 640))
    img = image.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Tambah batch
    return img

# Fungsi postprocessing (YOLO decode sangat tergantung pada format output)
def postprocess(output):
    detections = []
    for det in output[0]:  # shape (8400, 85): [x, y, w, h, conf, cls...]
        conf = det[4]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        class_id = int(np.argmax(det[5:]))
        detections.append((det[:4], conf, class_id))
    return detections

try:
    while True:
        counter += 1
        car.getData()

        if counter < 5:
            continue

        image = car.getImage()
        if image is None or not image.any():
            continue

        sensors = car.getSensors()
        speed = car.getSpeed()

        print(f"Speed: {speed}")
        print(f"Sensors: {sensors}")

        # Preprocessing
        input_image = preprocess(image).astype(np.float32)
        cuda.memcpy_htod(d_input, input_image)

        # Inference
        context.execute_v2(bindings)

        # Ambil hasil
        cuda.memcpy_dtoh(host_output, d_output)

        # Postprocessing
        detections = postprocess(host_output)

        # Visualisasi
        annotated_frame = cv2.resize(image, (640, 640))
        for det, conf, class_id in detections:
            x, y, w, h = det
            x1 = int((x - w / 2) * 640)
            y1 = int((y - h / 2) * 640)
            x2 = int((x + w / 2) * 640)
            y2 = int((y + h / 2) * 640)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{class_id} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("YOLOv11m TensorRT", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    try:
        car.stop()
    except:
        pass
    cv2.destroyAllWindows()
    print("🛑 Program dihentikan dengan aman.")
