'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

from engine import avisengine
import config 
import time
import cv2
from ultralytics import YOLO  # pastikan sudah install: pip install ultralytics

# Inisialisasi model YOLOv11m
model = YOLO('firadetector.pt')  # Ganti dengan path ke modelmu

# Membuat instance dari kelas Car
car = avisengine.Car()

# Koneksi ke simulator
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Variabel penghitung
counter = 0

# Mode debug
debug_mode = False

# Delay 3 detik agar koneksi stabil ke simulator
time.sleep(3)

try:
    while True:
        # Hitung jumlah loop
        counter += 1

        # Mengambil data dari simulator (harus dilakukan setiap frame)
        car.getData()

        # Setelah beberapa loop, ambil data kamera dan sensor
        if counter > 4:
            sensors = car.getSensors()       # List: [left, middle, right]
            image = car.getImage()           # Gambar OpenCV
            carSpeed = car.getSpeed()        # Kecepatan mobil saat ini

            if debug_mode:
                print(f"Speed : {carSpeed}") 
                print(f'Left : {sensors[0]} | Middle : {sensors[1]} | Right : {sensors[2]}')

            if image is not None and image.any():
                # Resize gambar ke 640x640 untuk YOLO dan display
                frame = cv2.resize(image, (640, 640))

                # Deteksi objek menggunakan YOLOv11m
                results = model(frame, verbose=False)[0]

                # Gambar bounding box ke frame
                annotated_frame = results.plot()

                # Tampilkan frame dengan deteksi
                cv2.imshow('YOLOv11m Detection', annotated_frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(10) == ord('q'):
                break

finally:
    car.stop()
    cv2.destroyAllWindows()
