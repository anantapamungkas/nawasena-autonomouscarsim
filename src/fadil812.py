'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

import time
import cv2
import numpy as np
import cv2.aruco as aruco # Import modul aruco dari OpenCV

import config # Asumsikan 'config.py' berisi SIMULATOR_IP, SIMULATOR_PORT, dan WHITE_SETTING
from engine import avisengine # Asumsikan 'engine.py' berisi kelas Car
from trackbar_utils import TrackbarManager # Asumsikan 'trackbar_utils.py' berisi kelas TrackbarManager

# --- Inisialisasi Trackbar ---
road_color_trackbar = TrackbarManager("road_color_trackbar")
road_color_trackbar.add_multiple(config.WHITE_SETTING)

# --- Koneksi ke Simulator ---
car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

time.sleep(3) # Beri waktu untuk koneksi stabil

# --- Konfigurasi Bird's-Eye View (BEV) ---
BEV_WIDTH = 640
BEV_HEIGHT = 480

tl = (int(BEV_WIDTH * 0.32), int(BEV_HEIGHT * 0.55))
bl = (int(0), int(BEV_HEIGHT * 0.94))
tr = (int(BEV_WIDTH * 0.68), int(BEV_HEIGHT * 0.55))
br = (int(BEV_WIDTH), int(BEV_HEIGHT * 0.94))
SRC_POINTS = np.float32([tl, bl, tr, br])
DST_POINTS = np.float32([[0, 0], [0, BEV_HEIGHT], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT]])

M_perspective = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
Minv_perspective = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)

def draw_perspective_points(frame_to_draw_on, tl_pt, bl_pt, tr_pt, br_pt):
    cv2.circle(frame_to_draw_on, (int(tl_pt[0]), int(tl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(bl_pt[0]), int(bl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(tr_pt[0]), int(tr_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(br_pt[0]), int(br_pt[1])), 5, (0, 0, 255), -1)

# --- Kalibrasi Piksel ke Meter (Dunia Nyata) ---
XM_PER_PIX = 3.7 / BEV_WIDTH
YM_PER_PIX = 30.0 / BEV_HEIGHT

# --- Parameter Kontrol Mobil ---
BASE_ENGINE_POWER = 5
MAX_ENGINE_POWER = 20
MIN_ENGINE_POWER = 5
ANGLE_THRESHOLD = 15
SPEED_REDUCTION_FACTOR = 1.25
STEERING_GAIN = -0.95
MAX_STEERING_ANGLE = 25

# --- Parameter Sensor ---
OBSTACLE_DISTANCE_THRESHOLD = 650 # Jarak rintangan depan (cm)
SPEED_REVERSE = -BASE_ENGINE_POWER # Menggunakan BASE_ENGINE_POWER sebagai nilai kecepatan mundur

# Parameter baru untuk kontrol kemudi berbasis sensor
SENSOR_STEERING_THRESHOLD = 500 # Jika sensor samping kurang dari ini, mulai koreksi kemudi
SENSOR_STEERING_GAIN = 0.3   # Seberapa kuat koreksi kemudi berdasarkan sensor
MIN_SENSOR_VALUE = 50        # Nilai sensor minimum yang masuk akal (hindari 0 / noise)

# --- Variabel State Mundur ---
reverse_start_time = 0
reverse_duration = 5 # Durasi mundur dalam detik
is_reversing = False

# --- Variabel State Belok Berdasarkan AprilTag ---
aruco_turn_start_time = 0
aruco_turn_duration = 8 # Durasi belok AprilTag dalam detik (default)
is_aruco_turning = False
aruco_turn_direction = 0 # -1 untuk kiri, 1 untuk kanan, 0 untuk tidak belok

# --- Variabel State Berhenti dan Maju Berdasarkan AprilTag ID 5 (BARU) ---
aruco_stop_go_start_time = 0
aruco_stop_duration_id5 = 4.0 # Durasi berhenti untuk ID 5
aruco_forward_duration_id5 = 1.0 # Durasi maju lurus untuk ID 5
is_aruco_stop_go = False
aruco_stop_go_phase = 0 # 0: idle, 1: stopping, 2: going_forward

# --- Variabel State Maju Lurus Berdasarkan AprilTag ID Selain 3 atau 5 (BARU) ---
aruco_other_id_go_start_time = 0
aruco_other_id_go_duration = 6.0 # Durasi maju lurus untuk ID selain 3 atau 5
is_aruco_other_id_go = False

# --- Inisialisasi ArUco Detector ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
parameters = aruco.DetectorParameters()

debug_mode = True

try:
    while True:
        car.getData()
        frame = car.getImage()
        sensors = car.getSensors() # [Sensor Kiri, Sensor Tengah, Sensor Kanan]
        current_car_speed_kmh = car.getSpeed()

        if frame is None or not frame.any():
            print("Gagal mendapatkan frame gambar dari simulator. Mencoba kembali...")
            time.sleep(0.1)
            car.setSpeed(1)
            car.setSteering(0)
            continue

        frame = cv2.resize(frame, (640, 640))
        result_original = frame.copy()

        # --- Logika Prioritas Tertinggi: Berhenti dan Maju ID 5 ---
        if is_aruco_stop_go:
            current_time = time.time()
            if aruco_stop_go_phase == 1: # Fase berhenti
                if current_time - aruco_stop_go_start_time < aruco_stop_duration_id5:
                    car.setSpeed(0) # Berhenti total
                    car.setSteering(0)
                    cv2.putText(result_original, "APRILTAG ID 5: BERHENTI!",
                                (frame.shape[1] // 2 - 320, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(result_original, f'Berhenti tersisa: {aruco_stop_duration_id5 - (current_time - aruco_stop_go_start_time):.1f}s',
                                (frame.shape[1] // 2 - 320, frame.shape[0] // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    # Selesai berhenti, lanjut ke fase maju
                    aruco_stop_go_phase = 2
                    aruco_stop_go_start_time = current_time # Reset timer untuk fase maju
                    print("AprilTag ID 5: Selesai berhenti, memulai maju lurus.")
            
            elif aruco_stop_go_phase == 2: # Fase maju lurus
                if current_time - aruco_stop_go_start_time < aruco_forward_duration_id5:
                    car.setSpeed(5) # Maju dengan percepatan 5
                    car.setSteering(0) # Tetap lurus
                    cv2.putText(result_original, "APRILTAG ID 5: MAJU LURUS!",
                                (frame.shape[1] // 2 - 320, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(result_original, f'Maju tersisa: {aruco_forward_duration_id5 - (current_time - aruco_stop_go_start_time):.1f}s',
                                (frame.shape[1] // 2 - 320, frame.shape[0] // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Selesai maju lurus, kembali ke mode normal
                    is_aruco_stop_go = False
                    aruco_stop_go_phase = 0
                    print("AprilTag ID 5: Selesai maju lurus, kembali ke deteksi jalur normal.")

            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
            
            if cv2.waitKey(10) == ord('q'):
                break
            continue # Lanjutkan ke iterasi berikutnya

        # --- Logika Maju Lurus untuk AprilTag ID Selain 3 atau 5 (Prioritas Kedua) ---
        if is_aruco_other_id_go:
            current_time = time.time()
            if current_time - aruco_other_id_go_start_time < aruco_other_id_go_duration:
                car.setSpeed(10) # Tetap maju lurus
                car.setSteering(0)
                cv2.putText(result_original, "APRILTAG LAIN: MAJU LURUS!",
                            (frame.shape[1] // 2 - 320, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(result_original, f'Maju tersisa: {aruco_other_id_go_duration - (current_time - aruco_other_id_go_start_time):.1f}s',
                            (frame.shape[1] // 2 - 320, frame.shape[0] // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                is_aruco_other_id_go = False
                print("AprilTag ID lain: Selesai maju lurus, kembali ke deteksi jalur normal.")
            
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
            
            if cv2.waitKey(10) == ord('q'):
                break
            continue # Lanjutkan ke iterasi berikutnya


        # --- Logika Belok Berdasarkan AprilTag (Prioritas Ketiga) ---
        if is_aruco_turning:
            if time.time() - aruco_turn_start_time < aruco_turn_duration:
                car.setSpeed(6)
                car.setSteering(aruco_turn_direction * 13)
                
                direction_text = "KIRI" if aruco_turn_direction == -1 else "KANAN"
                cv2.putText(result_original, f"APRILTAG BELOK {direction_text}!",
                            (frame.shape[1] // 2 - 280, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(result_original, f'Waktu belok tersisa: {aruco_turn_duration - (time.time() - aruco_turn_start_time):.1f}s',
                            (frame.shape[1] // 2 - 280, frame.shape[0] // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                is_aruco_turning = False
                aruco_turn_direction = 0
                print("Selesai belok karena AprilTag, kembali ke mode normal.")
            
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
            
            if cv2.waitKey(10) == ord('q'):
                break
            continue # Lanjutkan ke iterasi berikutnya

        # --- Logika Deteksi AprilTag (Jika tidak sedang dalam mode khusus) ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                current_id = ids[i][0] # Ambil ID marker saat ini
                current_corners = corners[i][0]

                # --- Mencari pojok Top-Left, Top-Right, Bottom-Left secara robust ---
                sorted_by_y = current_corners[current_corners[:, 1].argsort()]
                top_two_corners = sorted_by_y[:2]
                bottom_two_corners = sorted_by_y[2:]

                if top_two_corners[0, 0] < top_two_corners[1, 0]:
                    top_left = top_two_corners[0]
                    top_right = top_two_corners[1]
                else:
                    top_left = top_two_corners[1]
                    top_right = top_two_corners[0]

                if bottom_two_corners[0, 0] < bottom_two_corners[1, 0]:
                    bottom_left = bottom_two_corners[0]
                else:
                    bottom_left = bottom_two_corners[1]

                top_left_x = top_left[0]
                top_left_y = top_left[1]
                top_right_x = top_right[0]
                bottom_left_y = bottom_left[1]

                # Kondisi 1: Selisih sumbu X pojok kiri atas dan kanan atas lebih besar dari 50
                condition_x = abs(top_right_x - top_left_x) > 30

                # Kondisi 2: Selisih sumbu Y pojok kiri atas dan pojok kiri bawah lebih besar dari 50
                condition_y = abs(bottom_left_y - top_left_y) > 30

                # Hanya proses jika kedua kondisi terpenuhi DAN tidak sedang dalam mode khusus lainnya
                if condition_x and condition_y:
                    # Pastikan tidak ada mode AprilTag lain aktif
                    if not is_aruco_stop_go and not is_aruco_turning and not is_aruco_other_id_go: 
                        print(f"AprilTag Marker ID {current_id} Terdeteksi dengan Kondisi Ganda Terpenuhi!")
                        print(f"Selisih X (Top-Left ke Top-Right): {abs(top_right_x - top_left_x):.2f}")
                        print(f"Selisih Y (Top-Left ke Bottom-Left): {abs(bottom_left_y - top_left_y):.2f}")

                        # --- Menggunakan if/elif/else untuk ID marker ---
                        if current_id == 5:
                            is_aruco_stop_go = True
                            aruco_stop_go_phase = 1 # Mulai fase berhenti
                            aruco_stop_go_start_time = time.time()
                            car.setSpeed(0) # Langsung berhenti
                            car.setSteering(0)
                            print(f"Memicu berhenti dan maju karena AprilTag ID {current_id}.")
                            
                        elif current_id == 3:
                            is_aruco_turning = True
                            aruco_turn_start_time = time.time()
                            aruco_turn_direction = -1 # Belok kiri
                            aruco_turn_duration = 7.8 # Durasi khusus 7.5 detik untuk ID 3
                            car.setSpeed(BASE_ENGINE_POWER) 
                            car.setSteering(aruco_turn_direction * 13)
                            print(f"Memicu belok KIRI karena AprilTag ID {current_id} selama {aruco_turn_duration} detik.")
                        
                        else: # ID selain 3 atau 5 (kondisi 'selain itu')
                            is_aruco_other_id_go = True
                            aruco_other_id_go_start_time = time.time()
                            car.setSpeed(10) # Maju lurus dengan kecepatan 10
                            car.setSteering(0)
                            print(f"AprilTag ID {current_id} terdeteksi, memicu maju lurus selama {aruco_other_id_go_duration} detik.")
                            
                        # Visualisasi Marker di frame original
                        aruco.drawDetectedMarkers(result_original, corners)
                        cv2.putText(result_original, f"Marker ID: {current_id}", (int(top_left_x), int(top_left_y) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(result_original, f"APRILTAG ID {current_id} TERDETEKSI!",
                                            (frame.shape[1] // 2 - 380, frame.shape[0] // 2 + 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                        
                        cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
                        cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        cv2.imshow('Lanes on Original Frame', result_original)
                        cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
                        
                        if cv2.waitKey(10) == ord('q'):
                            break
                        
                        continue # Penting: Lanjutkan ke iterasi berikutnya setelah memicu aksi AprilTag

        # --- Logika Mundur (Prioritas Keempat) ---
        if is_reversing:
            if time.time() - reverse_start_time < reverse_duration:
                car.setSpeed(SPEED_REVERSE-10)
                car.setSteering(0)
                cv2.putText(result_original, "MUNDUR!", (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(result_original, f'Waktu mundur tersisa: {reverse_duration - (time.time() - reverse_start_time):.1f}s',
                            (frame.shape[1] // 2 - 180, frame.shape[0] // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                is_reversing = False
                car.setSpeed(0)
                print("Selesai mundur, kembali ke mode normal.")
            
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
            
            if cv2.waitKey(10) == ord('q'):
                break
            continue

        # --- Sensor-based Obstacle Detection (Memicu Mundur jika tidak sedang dalam mode khusus) ---
        if sensors[1] < OBSTACLE_DISTANCE_THRESHOLD:
            if not is_reversing:
                print(f"Rintangan terdeteksi di depan! Pembacaan sensor: {sensors[1]}. Memulai mundur.")
                is_reversing = True
                reverse_start_time = time.time()
                car.setSpeed(SPEED_REVERSE) 
                car.setSteering(0)
            
            cv2.putText(result_original, "RINTANGAN TERDETEKSI! Memulai Mundur...", (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR))
            
            if cv2.waitKey(10) == ord('q'):
                break
            continue

        # --- Jika tidak ada kondisi khusus aktif, lanjutkan dengan deteksi jalur ---
        roi_start_y = int(frame.shape[0] * 0.6)
        roi_end_y = frame.shape[0]
        roi_start_x = 0
        roi_end_x = frame.shape[1]
        frame_roi_display = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x].copy()

        warped_frame = cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR)

        road_rgb_value = road_color_trackbar.get_values_list()
        lower_road = np.array([road_rgb_value[0], road_rgb_value[1], road_rgb_value[2]])
        upper_road = np.array([road_rgb_value[3], road_rgb_value[4], road_rgb_value[5]])

        mask_road = cv2.inRange(warped_frame, lower_road, upper_road)
        binary_warped_for_lane = cv2.bitwise_not(mask_road)

        kernel = np.ones((3,3), np.uint8)
        binary_warped_for_lane = cv2.morphologyEx(binary_warped_for_lane, cv2.MORPH_OPEN, kernel)
        binary_warped_for_lane = cv2.morphologyEx(binary_warped_for_lane, cv2.MORPH_CLOSE, kernel)

        histogram = np.sum(binary_warped_for_lane[binary_warped_for_lane.shape[0]//2:,:], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 7
        margin = 100
        minpix = 30

        window_height = np.int32(binary_warped_for_lane.shape[0] // nwindows)

        nonzero = binary_warped_for_lane.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_warped_for_lane, binary_warped_for_lane, binary_warped_for_lane)) * 255

        for window in range(nwindows):
            win_y_low = binary_warped_for_lane.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped_for_lane.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_xright_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = None
        right_fit = None
        
        if len(leftx) > 50:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 50:
            right_fit = np.polyfit(righty, rightx, 2)

        curvature_m = 0.0
        lane_offset_m = 0.0
        steering_angle_command = 0.0
        current_engine_power = BASE_ENGINE_POWER

        result_bev = warped_frame.copy()
        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, binary_warped_for_lane.shape[0] - 1, binary_warped_for_lane.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            lane_area_bev = np.zeros_like(warped_frame).astype(np.uint8)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(lane_area_bev, np.int32([pts]), (0, 255, 0))

            result_bev = cv2.addWeighted(warped_frame, 1, lane_area_bev, 0.3, 0)

            unwarped_lane = cv2.warpPerspective(lane_area_bev, Minv_perspective, (frame.shape[1], frame.shape[0]))
            result_original = cv2.addWeighted(frame, 1, unwarped_lane, 0.3, 0)

            left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
            right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

            y_eval_m = (BEV_HEIGHT - 1) * YM_PER_PIX

            A_mid_cr = (left_fit_cr[0] + right_fit_cr[0]) / 2
            B_mid_cr = (left_fit_cr[1] + right_fit_cr[1]) / 2

            first_deriv = 2 * A_mid_cr * y_eval_m + B_mid_cr
            second_deriv = 2 * A_mid_cr

            if np.abs(second_deriv) < 1e-6:
                curvature_m = 10000.0
            else:
                curvature_m = ((1 + first_deriv**2)**1.5) / np.abs(second_deriv)

            car_center_x_m = (BEV_WIDTH / 2) * XM_PER_PIX
            lane_center_bottom_x_m = (left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2] +
                                        right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]) / 2

            lane_offset_m = car_center_x_m - lane_center_bottom_x_m
            
            p_term = -lane_offset_m * STEERING_GAIN * 5
            d_term = -A_mid_cr * STEERING_GAIN * 1000

            steering_angle_command = p_term + d_term
            
            # --- Koreksi Kemudi Berbasis Sensor Samping ---
            sensor_steering_correction = 0.0
            if sensors[0] < SENSOR_STEERING_THRESHOLD and sensors[0] > MIN_SENSOR_VALUE:
                sensor_steering_correction += (SENSOR_STEERING_THRESHOLD - sensors[0]) * SENSOR_STEERING_GAIN
            
            if sensors[2] < SENSOR_STEERING_THRESHOLD and sensors[2] > MIN_SENSOR_VALUE:
                sensor_steering_correction -= (SENSOR_STEERING_THRESHOLD - sensors[2]) * SENSOR_STEERING_GAIN

            steering_angle_command += sensor_steering_correction
            
            steering_angle_command = np.clip(steering_angle_command, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
            steering_angle_command *=10 # Skala ke rentang kemudi simulator
            
            abs_steering_angle = np.abs(steering_angle_command)
            
            if abs_steering_angle > ANGLE_THRESHOLD:
                speed_reduction = (abs_steering_angle - ANGLE_THRESHOLD) * SPEED_REDUCTION_FACTOR
                current_engine_power = MAX_ENGINE_POWER - speed_reduction
                current_engine_power = max(current_engine_power, MIN_ENGINE_POWER)
            else:
                current_engine_power = MAX_ENGINE_POWER
            
            current_engine_power = min(current_engine_power, MAX_ENGINE_POWER)

            car.setSpeed(current_engine_power)
            car.setSteering(steering_angle_command)

            line_start_x = frame.shape[1] // 2
            line_start_y = frame.shape[0]

            line_length = 150
            end_x = int(line_start_x + line_length * np.sin(np.radians(steering_angle_command)))
            end_y = int(line_start_y - line_length * np.cos(np.radians(steering_angle_command)))
            cv2.line(result_original, (line_start_x, line_start_y), (end_x, end_y), (255, 0, 0), 3)

            cv2.putText(result_original, f'Curvature: {curvature_m:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Offset: {lane_offset_m:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Steer Angle: {steering_angle_command:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Engine Power: {current_engine_power:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', result_bev)
        else:
            car.setSpeed(MIN_ENGINE_POWER)
            car.setSteering(0)
            print("Tidak ada garis jalur yang terdeteksi. Bergerak perlahan.")
            cv2.putText(result_original, f'Sensors: L:{sensors[0]} M:{sensors[1]} R:{sensors[2]}',
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', out_img)

        if debug_mode:
            cv2.imshow('Original Frame', frame)
            cv2.imshow('ROI (for reference)', frame_roi_display)
            cv2.imshow('Bird Eye View (Raw)', warped_frame)
            cv2.imshow('Mask Road (White is Road)', mask_road)
            cv2.imshow('Binary Mask (Inverted for Lane Detection)', binary_warped_for_lane)
            cv2.imshow('Sliding Window Debug', out_img)

        if cv2.waitKey(10) == ord('q'):
            break

finally:
    car.stop()
    cv2.destroyAllWindows()