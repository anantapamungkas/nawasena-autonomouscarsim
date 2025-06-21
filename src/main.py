# main.py

import cv2
import time
import torch
from ultralytics import YOLO
import config
from avisengine import Car
from steering import process_labels
from dummy_detector import DummyDetector, State as DummyState  # ⬅️ ganti nama untuk lebih jelas

# Threshold confidence YOLO
CONFIDENCE_THRESHOLD = 0.5

# Inisialisasi model YOLO
model = YOLO('firayolov11m.pt')  # ganti path sesuai file modelmu
# Deteksi apakah GPU tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'✅ CUDA tersedia' if torch.cuda.is_available() else '⚠️ CUDA tidak tersedia'}, menggunakan {device}.")

# Inisialisasi mobil dan koneksi
car = Car()
if not car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT):
    exit("❌ Gagal terhubung ke simulator.")

# Inisialisasi Dummy Detector
dummy_detector = DummyDetector()

# Tunggu koneksi stabil
time.sleep(2)

# Loop utama
try:
    prev_time = time.time()

    while True:
        # Ambil data dan gambar
        car.getData()
        sensors = car.getSensors()
        image = car.getImage()

        # Deteksi dummy (menggunakan state machine)
        dummy_state = dummy_detector.update(sensors, car)

        # Jika sedang dalam state NON-NORMAL, skip deteksi YOLO
        if dummy_state != DummyState.NORMAL:
            if image is not None:
                cv2.imshow("Dummy Navigation", image)
                if cv2.waitKey(1) == ord('q'):
                    break
            continue  # skip ke frame berikutnya

        # Jalankan YOLO jika tidak sedang menghindar dummy
        if image is not None and image.any():
            frame = cv2.resize(image, (640, 640))
            results = model(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)[0]

            # Ambil label
            labels = [model.names[int(cls)] for cls in results.boxes.cls]
            print("📦 Deteksi:", labels)

            # Kirim label ke steering logic
            process_labels(labels, car)

            # Tampilkan hasil YOLO + FPS
            annotated = results.plot()
            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()

            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("YOLOv11 Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    car.stop()
    cv2.destroyAllWindows()
    print("🛑 Program selesai dengan aman.")
