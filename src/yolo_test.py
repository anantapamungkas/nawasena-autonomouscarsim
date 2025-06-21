'''
@ 2023, Copyright AVIS Engine
- Compatible with AVIS Engine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

import avisengine
import config 
import time
import cv2
from ultralytics import YOLO
import torch
import sys
from steering import process_labels

# Threshold deteksi minimal (confidence)
CONFIDENCE_THRESHOLD = 0.5

# Periksa apakah CUDA tersedia
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ CUDA tersedia. Menggunakan GPU.")
else:
    device = torch.device('cpu')
    print("⚠️ CUDA tidak tersedia. Menggunakan CPU.")

# Inisialisasi model YOLOv11m
model = YOLO('firayolov11m.pt')  # Ganti dengan path ke modelmu

# Inisialisasi objek mobil dari AVIS Engine
car = avisengine.Car()

# Mencoba koneksi ke simulator
try:
    car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)
    print(f"✅ Terhubung ke simulator di {config.SIMULATOR_IP}:{config.SIMULATOR_PORT}")
except Exception as e:
    print(f"❌ Gagal terhubung ke simulator: {e}")
    sys.exit(1)

# Variabel loop
counter = 0
debug_mode = True

# Untuk menghitung FPS
prev_time = time.time()
fps = 0.0

# Tunggu agar koneksi stabil
time.sleep(3)

try:
    while True:
        counter += 1
        current_time = time.time()

        # Ambil data dari simulator (wajib setiap frame)
        try:
            car.getData()
        except Exception as e:
            print(f"⚠️ Gagal mengambil data dari simulator: {e}")
            break

        if counter > 4:
            sensors = car.getSensors()       # [left, middle, right]
            image = car.getImage()           # Citra kamera (OpenCV format)
            speed = car.getSpeed()           # Kecepatan mobil saat ini

            if debug_mode:
                print(f"Speed : {speed}") 
                print(f'Sensors => L: {sensors[0]} | M: {sensors[1]} | R: {sensors[2]}')

            if image is not None and image.any():
                # Resize ke 640x640 agar cocok dengan input YOLO
                frame = cv2.resize(image, (640, 640))

                results = model(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)[0]

                # Ambil label deteksi
                detected_labels = [model.names[int(cls)] for cls in results.boxes.cls]

                # Kirim label ke file steering untuk diproses
                process_labels(detected_labels, car)

                # Plot bounding box ke frame
                annotated_frame = results.plot()

                # Hitung FPS
                elapsed_time = current_time - prev_time
                prev_time = current_time
                fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

                # Tampilkan nilai FPS di pojok kiri atas
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Tampilkan hasil deteksi
                cv2.imshow("YOLOv11 Detection", annotated_frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(1) == ord('q'):
                break

finally:
    try:
        car.stop()
    except Exception as e:
        print(f"⚠️ Gagal menghentikan mobil dengan benar: {e}")
    cv2.destroyAllWindows()
    print("🛑 Program dihentikan dengan aman.")
