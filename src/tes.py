import socket
import os
import numpy as np
import cv2
from pupil_apriltags import Detector

# Konfigurasi koneksi ke AVIS
SERVER_IP = "127.0.0.1"
SERVER_PORT = 25001
WIDTH, HEIGHT, CHANNELS = 640, 640, 3
APRILTAG_FOLDER = "apriltag"

# Inisialisasi detektor AprilTag
detector = Detector(families="tag36h11")

# =======================
# 🔍 Ambil ID tag dari folder apriltag
# =======================
valid_ids = set()
for filename in os.listdir(APRILTAG_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(APRILTAG_FOLDER, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        for r in results:
            valid_ids.add(r.tag_id)

print(f"[INFO] ID AprilTag valid dari folder: {valid_ids}")

# =======================
# 🚗 Hubungkan ke AVIS Engine
# =======================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
print("[INFO] Terhubung ke AVIS Engine")

# =======================
# 🔁 Loop utama: terima frame dan deteksi
# =======================
while True:
    frame_data = b""
    expected_size = WIDTH * HEIGHT * CHANNELS

    while len(frame_data) < expected_size:
        packet = client_socket.recv(expected_size - len(frame_data))
        if not packet:
            break
        frame_data += packet

    if not frame_data:
        continue

    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    for r in results:
        if r.tag_id not in valid_ids:
            continue  # Skip tag yang tidak cocok

        corners = [tuple(map(int, pt)) for pt in r.corners]
        (ptA, ptB, ptC, ptD) = corners
        (cX, cY) = tuple(map(int, r.center))

        # Bounding box + label
        cv2.polylines(frame, [np.array(corners, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(frame, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"{tagFamily}", (ptA[0], ptA[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "DETECTED", (ptA[0], ptA[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"[INFO] Ditemukan ID valid: {r.tag_id} ({tagFamily})")

    # Tampilkan hasil deteksi
    cv2.imshow("Deteksi AprilTag dari AVIS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =======================
# 🧹 Bersihkan koneksi
# =======================
client_socket.close()
cv2.destroyAllWindows()
