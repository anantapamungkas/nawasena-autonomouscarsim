from avisengine import Car  # Pastikan ini file `avisengine.py` kamu
from yolo_trt_inference import YoLov11mTRT
import config
import time
import cv2
import numpy as np
import sys

# Inisialisasi model TensorRT
model_trt = YoLov11mTRT("firayolov11m_fp32.trt")  # Ganti dengan path engine-mu

# Inisialisasi mobil AVIS Engine
car = Car()

# Coba koneksi ke simulator
try:
    car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)
    print(f"✅ Terhubung ke simulator di {config.SIMULATOR_IP}:{config.SIMULATOR_PORT}")
except Exception as e:
    print(f"❌ Gagal terhubung ke simulator: {e}")
    sys.exit(1)

counter = 0
debug_mode = True
prev_time = time.time()

try:
    while True:
        counter += 1
        current_time = time.time()

        # Ambil data dari simulator
        try:
            car.getData()
        except Exception as e:
            print(f"⚠️ Gagal mengambil data: {e}")
            break

        if counter > 4:
            sensors = car.getSensors()
            image = car.getImage()
            speed = car.getSpeed()

            if debug_mode:
                print(f"Speed : {speed}") 
                print(f"Sensors => L: {sensors[0]} | M: {sensors[1]} | R: {sensors[2]}")

            if image is not None and image.any():
                # Deteksi pakai TensorRT
                detections = model_trt.infer(image)

                # Postprocessing (contoh: threshold & box)
                boxes = []
                for det in detections[0]:
                    conf = det[4]
                    if conf > 0.6:
                        x, y, w, h = map(int, det[0:4])
                        boxes.append((x, y, w, h))

                # Visualisasi hasil
                annotated = image.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Hitung dan tampilkan FPS
                elapsed_time = current_time - prev_time
                prev_time = current_time
                fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
                cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow("YOLOv11m TensorRT", annotated)

            if cv2.waitKey(1) == ord('q'):
                break

finally:
    try:
        car.stop()
    except Exception as e:
        print(f"⚠️ Gagal menghentikan mobil: {e}")
    cv2.destroyAllWindows()
    print("🛑 Program dihentikan dengan aman.")
