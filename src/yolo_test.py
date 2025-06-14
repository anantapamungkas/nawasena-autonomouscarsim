'''
@ 2023, Copyright AVIS Engine
- Compatible with AVIS Engine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

from engine import avisengine
import config 
import time
import cv2
from ultralytics import YOLO
import torch
import sys

# Periksa apakah CUDA tersedia
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ CUDA tersedia. Menggunakan GPU.")
else:
    device = torch.device('cpu')
    print("⚠️ CUDA tidak tersedia. Menggunakan CPU.")

# Inisialisasi model YOLOv11m
model = YOLO('firadetector.pt')  # Ganti dengan path ke modelmu

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

# Tunggu agar koneksi stabil
time.sleep(3)

try:
    while True:
        counter += 1

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

                # Deteksi objek menggunakan model (langsung ke GPU)
                results = model(frame, device=device, verbose=False)[0]

                # Plot bounding box ke frame
                annotated_frame = results.plot()

                # Tampilkan hasil deteksi
                cv2.imshow("YOLOv11m Detection", annotated_frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(10) == ord('q'):
                break

finally:
    try:
        car.stop()
    except Exception as e:
        print(f"⚠️ Gagal menghentikan mobil dengan benar: {e}")
    cv2.destroyAllWindows()
    print("🛑 Program dihentikan dengan aman.")