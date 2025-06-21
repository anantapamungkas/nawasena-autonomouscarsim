import socket
import os
import numpy as np
import cv2
from pupil_apriltags import Detector

# Konfigurasi
SERVER_IP = "127.0.0.1"
SERVER_PORT = 25001
WIDTH, HEIGHT, CHANNELS = 640, 640, 3
APRILTAG_FOLDER = "apriltag"

# Inisialisasi detektor
detector = Detector(families="tag36h11")

# =======================
# 🔍 1. Ambil ID tag dari folder apriltag
# =======================
valid_ids = set()
for filename in os.listdir(APRILTAG_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(APRILTAG_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        for r in results:
            valid_ids.add(r.tag_id)

print(f"[INFO] ID AprilTag valid dari folder: {valid_ids}")

# =======================
# 🚗 2. Hubungkan ke AVIS Engine
# =======================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
print("[INFO] Terhubung ke AVIS Engine")

while True:
    frame_data = b""
    while len(frame_data) < WIDTH * HEIGHT * CHANNELS:
        packet = client_socket.recv(WIDTH * HEIGHT * CHANNELS - len(frame_data))
        if not packet:
            break
        frame_data += packet

    if not frame_data:
        continue

    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = detector.detect(gray)
    print(f"[INFO] {len(results)} tag terdeteksi di AVIS")

    for r in results:
        if r.tag_id not in valid_ids:
            continue  # Hanya proses tag yang valid

        (ptA, ptB, ptC, ptD) = r.corners
        ptA, ptB, ptC, ptD = map(lambda p: (int(p[0]), int(p[1])), [ptA, ptB, ptC, ptD])
        (cX, cY) = (int(r.center[0]), int(r.center[1]))

        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(frame, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"[INFO] Ditemukan ID valid: {r.tag_id} ({tagFamily})")

    cv2.imshow("Deteksi AprilTag AVIS (Filtered)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()
