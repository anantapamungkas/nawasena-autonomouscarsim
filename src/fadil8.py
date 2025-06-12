'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

import time
import cv2
import numpy as np

import config # Asumsikan 'config.py' berisi SIMULATOR_IP, SIMULATOR_PORT, dan WHITE_SETTING
from engine import avisengine # Asumsikan 'engine.py' berisi kelas Car
from trackbar_utils import TrackbarManager # Asumsikan 'trackbar_utils.py' berisi kelas TrackbarManager

# --- Inisialisasi Trackbar ---
# Trackbar untuk mengatur rentang warna jalan (putih/abu-abu)
road_color_trackbar = TrackbarManager("road_color_trackbar")
road_color_trackbar.add_multiple(config.WHITE_SETTING)

# --- Koneksi ke Simulator ---
car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Beri waktu untuk koneksi stabil
time.sleep(3)

# --- Konfigurasi Bird's-Eye View (BEV) ---
BEV_WIDTH = 640
BEV_HEIGHT = 480

# Titik-titik sumber untuk transformasi perspektif (dari frame asli)
# Diperkirakan secara manual, perlu disesuaikan dengan lingkungan simulator Anda
tl = (int(BEV_WIDTH * 0.33), int(BEV_HEIGHT * 0.5)) # Top-left
bl = (int(0), int(BEV_HEIGHT * 0.85))             # Bottom-left
tr = (int(BEV_WIDTH * 0.67), int(BEV_HEIGHT * 0.5)) # Top-right
br = (int(BEV_WIDTH), int(BEV_HEIGHT * 0.85))     # Bottom-right

SRC_POINTS = np.float32([tl, bl, tr, br])
# Titik-titik tujuan (di frame BEV)
DST_POINTS = np.float32([[0, 0], [0, BEV_HEIGHT], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT]])

# Matriks transformasi perspektif dan inversnya
M_perspective = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
Minv_perspective = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)

# Fungsi untuk menggambar titik-titik perspektif di frame asli (untuk debugging)
def draw_perspective_points(frame_to_draw_on, tl_pt, bl_pt, tr_pt, br_pt):
    cv2.circle(frame_to_draw_on, (int(tl_pt[0]), int(tl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(bl_pt[0]), int(bl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(tr_pt[0]), int(tr_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(br_pt[0]), int(br_pt[1])), 5, (0, 0, 255), -1)

# --- Kalibrasi Piksel ke Meter (Dunia Nyata) ---
# Nilai ini harus dikalibrasi secara akurat dengan lebar dan panjang jalur di simulator Anda
# Contoh: lebar jalur 3.7 meter, panjang yang terlihat di BEV 30 meter
XM_PER_PIX = 3.7 / BEV_WIDTH  # Meter per piksel di sumbu X (horizontal)
YM_PER_PIX = 30.0 / BEV_HEIGHT # Meter per piksel di sumbu Y (vertikal)

# --- Parameter Kontrol Mobil ---
# Kekuatan mesin (bukan kecepatan KMH langsung)
BASE_ENGINE_POWER = 30 
MAX_ENGINE_POWER = 50
MIN_ENGINE_POWER = 10 # Hindari 0 agar mobil tidak berhenti total saat kehilangan jalur
ANGLE_THRESHOLD = 5   # Sudut kemudi (derajat) di mana kecepatan mulai dikurangi
SPEED_REDUCTION_FACTOR = 1.5 # Faktor seberapa agresif pengurangan kekuatan mesin
STEERING_GAIN = 0.5   # Gain untuk mengonversi offset/kurva menjadi sudut kemudi
MAX_STEERING_ANGLE = 25 # Batas sudut kemudi maksimal (misal: +/- 25 derajat)

debug_mode = True # Aktifkan untuk menampilkan print dan banyak jendela debug

try:
    while True:
        car.getData()
        frame = car.getImage()

        if frame is None or not frame.any():
            print("Gagal mendapatkan frame gambar dari simulator. Mencoba kembali...")
            time.sleep(0.1)
            car.setSpeed(0)
            car.setSteering(0)
            continue

        # Resize frame agar konsisten (jika diperlukan)
        # Perhatikan: jika BEV_WIDTH/HEIGHT sudah sesuai, tidak perlu resize ini
        frame = cv2.resize(frame, (640, 640)) # Sesuaikan dengan BEV_WIDTH/HEIGHT jika berbeda

        # --- ROI (Hanya untuk Display) ---
        # Untuk demonstrasi, bagian ini tidak memengaruhi pemrosesan, hanya tampilan
        roi_start_y = int(frame.shape[0] * 0.6)
        roi_end_y = frame.shape[0]
        roi_start_x = 0
        roi_end_x = frame.shape[1]
        frame_roi_display = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x].copy()

        # --- 1. Terapkan Transformasi Bird's-Eye View (BEV) ---
        draw_perspective_points(frame, tl, bl, tr, br) # Gambar titik di frame asli
        warped_frame = cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR)

        # --- 2. Segmentasi Warna (Masking) pada BEV Frame - DETEKSI JALAN ---
        road_rgb_value = road_color_trackbar.get_values_list()
        lower_road = np.array([road_rgb_value[0], road_rgb_value[1], road_rgb_value[2]])
        upper_road = np.array([road_rgb_value[3], road_rgb_value[4], road_rgb_value[5]])

        # Masker yang mendeteksi area JALAN (piksel putih = jalan, piksel hitam = non-jalan)
        mask_road = cv2.inRange(warped_frame, lower_road, upper_road)

        # Invert mask_road untuk mendapatkan MARKAH JALAN (putih = marka, hitam = jalan/lainnya)
        # Ini penting agar sliding window mencari piksel putih (marka)
        binary_warped_for_lane = cv2.bitwise_not(mask_road)

        # Opsional: Terapkan morphological operations untuk menghaluskan mask marka
        kernel = np.ones((3,3), np.uint8)
        binary_warped_for_lane = cv2.morphologyEx(binary_warped_for_lane, cv2.MORPH_OPEN, kernel)
        binary_warped_for_lane = cv2.morphologyEx(binary_warped_for_lane, cv2.MORPH_CLOSE, kernel)

        # --- 3. Analisis Histogram untuk Titik Awal Jalur ---
        # Histogram di bagian bawah gambar BEV untuk menemukan puncak-puncak marka
        histogram = np.sum(binary_warped_for_lane[binary_warped_for_lane.shape[0]//2:,:], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)

        # Titik awal untuk pencarian jalur kiri dan kanan
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # --- 4. Pencarian Sliding Window ---
        nwindows = 9
        margin = 100 # Lebar jendela pencarian di sekitar jalur
        minpix = 50  # Jumlah piksel minimum yang diperlukan untuk memperbarui posisi jendela

        window_height = np.int32(binary_warped_for_lane.shape[0] // nwindows)

        # Identifikasi semua piksel non-nol (putih) pada gambar biner
        nonzero = binary_warped_for_lane.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # Gambar untuk debugging sliding window
        out_img = np.dstack((binary_warped_for_lane, binary_warped_for_lane, binary_warped_for_lane)) * 255

        for window in range(nwindows):
            win_y_low = binary_warped_for_lane.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped_for_lane.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Gambar jendela di out_img
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identifikasi piksel putih dalam jendela
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Jika cukup piksel ditemukan, perbarui posisi jendela berikutnya
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Gabungkan semua indeks piksel yang ditemukan
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # --- 5. Fit Polinomial ke Titik Jalur ---
        left_fit = None
        right_fit = None
        
        # Hanya lakukan fitting jika ada cukup titik
        if len(leftx) > 50: # Minimal 50 piksel untuk fitting yang stabil
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 50:
            right_fit = np.polyfit(righty, rightx, 2)

        # --- Inisialisasi Metrik Kontrol ---
        curvature_m = 0.0
        lane_offset_m = 0.0
        steering_angle_command = 0.0
        current_engine_power = BASE_ENGINE_POWER # Default engine power

        # Salin frame asli untuk menggambar overlay
        result_original = frame.copy()

        # --- 6. Overlay Jalur ke BEV dan Frame Asli ---
        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, binary_warped_for_lane.shape[0] - 1, binary_warped_for_lane.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Gambar titik jalur di gambar debug sliding window
            out_img[lefty, leftx] = [255, 0, 0]   # Biru untuk jalur kiri
            out_img[righty, rightx] = [0, 0, 255] # Merah untuk jalur kanan

            # Buat area jalur untuk overlay di BEV
            lane_area_bev = np.zeros_like(warped_frame).astype(np.uint8)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(lane_area_bev, np.int32([pts]), (0, 255, 0)) # Hijau untuk area jalur

            # Gabungkan BEV asli dengan area jalur overlay
            result_bev = cv2.addWeighted(warped_frame, 1, lane_area_bev, 0.3, 0)

            # Unwarp area jalur kembali ke perspektif asli
            unwarped_lane = cv2.warpPerspective(lane_area_bev, Minv_perspective, (frame.shape[1], frame.shape[0]))
            # Gabungkan frame asli dengan area jalur yang sudah di-unwarp
            result_original = cv2.addWeighted(frame, 1, unwarped_lane, 0.3, 0)

            # --- Perhitungan Kelengkungan dan Offset ---
            # Fit polinomial ke dunia nyata (meter)
            left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
            right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

            # Titik evaluasi di bagian bawah gambar (dekat mobil)
            y_eval_m = (BEV_HEIGHT - 1) * YM_PER_PIX

            # Kelengkungan jalan (rata-rata dari kiri dan kanan, atau pusat)
            # Menggunakan jalur tengah untuk perhitungan kelengkungan
            # y = Ax^2 + Bx + C -> x = Ay^2 + By + C (karena kita fit x sebagai fungsi y)
            # Kurva = (1 + (dx/dy)^2)^(3/2) / |d2x/dy2|
            # dx/dy = 2*A_cr*y + B_cr
            # d2x/dy2 = 2*A_cr

            # Koefisien untuk jalur tengah (dalam meter)
            A_mid_cr = (left_fit_cr[0] + right_fit_cr[0]) / 2
            B_mid_cr = (left_fit_cr[1] + right_fit_cr[1]) / 2

            first_deriv = 2 * A_mid_cr * y_eval_m + B_mid_cr
            second_deriv = 2 * A_mid_cr

            if np.abs(second_deriv) < 1e-6: # Hindari pembagian nol
                curvature_m = 10000.0 # Angka besar untuk mendekati garis lurus
            else:
                curvature_m = ((1 + first_deriv**2)**1.5) / np.abs(second_deriv)

            # Offset mobil dari pusat jalur
            car_center_x_m = (BEV_WIDTH / 2) * XM_PER_PIX
            lane_center_bottom_x_m = (left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2] +
                                     right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]) / 2

            lane_offset_m = car_center_x_m - lane_center_bottom_x_m
            
            # --- Perhitungan Sudut Kemudi (Steering Angle) ---
            # Menggunakan kombinasi P-controller untuk offset dan D-controller (atau turunan) untuk kelengkungan
            # Steering angle = Kp * offset + Kd * curvature (sederhana)
            # Atau, secara lebih intuitif, kita menginginkan mobil mengarah ke pusat jalur di depan.
            # Menggunakan arctan(offset / jarak_pandang) untuk steering (pendekatan PID yang lebih sederhana)
            
            # Steering proportional to offset from center (cross-track error)
            steering_from_offset = np.degrees(np.arctan(lane_offset_m / (BEV_HEIGHT * YM_PER_PIX / 2))) # Perkiraan jarak pandang ke depan
            
            # Steering proportional to curvature (heading error)
            # Anggap arah yang diinginkan adalah tangen kurva di depan.
            # Sudut target bisa dihitung dari turunan pertama polinomial di titik tertentu di depan.
            # Misalnya, di tengah gambar (y_eval_pixels = BEV_HEIGHT/2)
            y_lookahead_m = (BEV_HEIGHT / 2) * YM_PER_PIX
            first_deriv_lookahead = 2 * A_mid_cr * y_lookahead_m + B_mid_cr
            steering_from_curvature = np.degrees(np.arctan(first_deriv_lookahead))

            # Gabungkan kedua komponen (contoh sederhana)
            # Anda bisa menyesuaikan bobot (gain) untuk masing-masing
            # Semakin besar lane_offset_m, semakin banyak koreksi yang diperlukan.
            # Semakin tajam kelengkungan, semakin banyak steering yang diperlukan.
            
            # Pendekatan yang lebih umum: steering_angle_command = (offset_error + lookahead_error) * STEERING_GAIN
            # lookahead_error bisa dianggap sebagai deviasi dari jalur yang diharapkan pada jarak tertentu di depan.
            
            # Mari kita gunakan pendekatan yang mirip dengan PID:
            # P-term: proporsional terhadap offset jalur (mobil harus bergerak ke tengah)
            p_term = -lane_offset_m * STEERING_GAIN * 5 # Kalikan dengan konstanta untuk respons lebih kuat
            
            # D-term: proporsional terhadap sudut yang dibutuhkan untuk mengikuti kelengkungan (mobil harus berbelok sesuai kelengkungan)
            # Sumbu Y di gambar BEV terbalik, jadi perlu penyesuaian.
            # Jika A_mid_cr positif, kelengkungan ke kanan; jika negatif, ke kiri.
            # Kita bisa mengaitkannya langsung ke sudut yang dibutuhkan.
            d_term = -A_mid_cr * STEERING_GAIN * 1000 # Kalikan dengan konstanta besar karena A_mid_cr kecil

            # Gabungkan P-term dan D-term
            steering_angle_command = p_term + d_term
            
            # Batasi sudut kemudi agar tidak terlalu ekstrem
            steering_angle_command = np.clip(steering_angle_command, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
            steering_angle_command *=-10
            # --- Kontrol Kecepatan Adaptif ---
            # Mengurangi kekuatan mesin (setSpeed) saat mobil berbelok tajam
            abs_steering_angle = np.abs(steering_angle_command)
            
            if abs_steering_angle > ANGLE_THRESHOLD:
                speed_reduction = (abs_steering_angle - ANGLE_THRESHOLD) * SPEED_REDUCTION_FACTOR
                current_engine_power = MAX_ENGINE_POWER - speed_reduction
                current_engine_power = max(current_engine_power, MIN_ENGINE_POWER)
            else:
                current_engine_power = MAX_ENGINE_POWER
            
            current_engine_power = min(current_engine_power, MAX_ENGINE_POWER) # Batasi tidak melebihi MAX_ENGINE_POWER

            # --- Menggerakkan Mobil ---
            car.setSpeed(current_engine_power)
            car.setSteering(steering_angle_command)

            # --- Visualisasi Garis Kemudi di Frame Asli ---
            line_start_x = frame.shape[1] // 2
            line_start_y = frame.shape[0]

            line_length = 150 # Panjang garis indikator kemudi
            end_x = int(line_start_x + line_length * np.sin(np.radians(steering_angle_command)))
            end_y = int(line_start_y - line_length * np.cos(np.radians(steering_angle_command)))
            cv2.line(result_original, (line_start_x, line_start_y), (end_x, end_y), (255, 0, 0), 3) # Biru

            # --- Tampilan Metrik di Frame Asli ---
            cv2.putText(result_original, f'Curvature: {curvature_m:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Offset: {lane_offset_m:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Steer Angle: {steering_angle_command:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Engine Power: {current_engine_power:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            current_car_speed_kmh = car.getSpeed() # Dapatkan kecepatan nyata dari simulator
            cv2.putText(result_original, f'Car Speed: {current_car_speed_kmh:.2f} KMH', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', result_bev)
        else:
            # Jika jalur tidak terdeteksi, bergerak perlahan atau berhenti
            car.setSpeed(MIN_ENGINE_POWER)
            car.setSteering(0)
            print("Tidak ada garis jalur yang terdeteksi. Bergerak perlahan atau berhenti.")
            cv2.imshow('Lanes on Original Frame', result_original) # Tampilkan frame asli tanpa overlay
            cv2.imshow('Lanes on Bird Eye View', out_img) # Tampilkan debug BEV tanpa overlay

        # --- Tampilan Hasil Debugging (jika debug_mode True) ---
        if debug_mode:
            cv2.imshow('Original Frame', frame)
            cv2.imshow('ROI (for reference)', frame_roi_display)
            cv2.imshow('Bird Eye View (Raw)', warped_frame)
            cv2.imshow('Mask Road (White is Road)', mask_road)
            cv2.imshow('Binary Mask (Inverted for Lane Detection)', binary_warped_for_lane)
            cv2.imshow('Sliding Window Debug', out_img)

        # Kontrol keluar
        if cv2.waitKey(10) == ord('q'):
            break

finally:
    # Hentikan mobil dan tutup semua jendela saat program berakhir
    car.stop()
    cv2.destroyAllWindows()