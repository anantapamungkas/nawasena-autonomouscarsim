# '''
# @ 2023, Copyright AVIS Engine
# - An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
# '''

# import time
# import cv2
# import numpy as np
# # import matplotlib.pyplot as plt # Tidak digunakan, bisa di-komen atau dihapus

# import config
# from engine import avisengine
# from trackbar_utils import TrackbarManager


# # Inisialisasi Trackbar
# white_trackbar = TrackbarManager("white_trackbar")
# white_trackbar.add_multiple(config.WHITE_SETTING)

# # Dirt dan Yellow trackbar dipertahankan, meskipun tidak digunakan dalam pemrosesan saat ini.
# # Ini memberi Anda fleksibilitas jika ingin menambahkannya nanti.
# dirt_trackbar = TrackbarManager("dirt_trackbar")
# dirt_trackbar.add_multiple(config.DIRT_SETTING)

# yellow_trackbar = TrackbarManager("yellow_trackbar")
# yellow_trackbar.add_multiple(config.YELLOW_SETTING)


# # Koneksi ke Simulator
# car = avisengine.Car()
# car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# time.sleep(3) # Beri waktu untuk koneksi


# # --- Konfigurasi Bird's-Eye View (BEV) ---
# # Penting: Sesuaikan titik-titik ini berdasarkan perspektif kamera simulator Anda!
# # Ini adalah bagian yang paling sering membutuhkan kalibrasi manual.
# # Koordinat titik sumber (src_points) adalah titik-titik di gambar asli yang membentuk trapesium jalan.
# # Koordinat titik tujuan (dst_points) adalah titik-titik di gambar hasil BEV yang akan menjadi persegi panjang.

# # Ukuran target untuk gambar BEV
# BEV_WIDTH = 320
# BEV_HEIGHT = 480

# # Contoh titik sumber (relatif terhadap frame 640x640) - Sesuaikan ini!
# # Anda bisa menggunakan alat seperti GIMP atau mengklik gambar untuk menemukan koordinat yang tepat.
# # Contoh ini mencoba menutupi sebagian besar jalan di depan.
# # [X, Y]
# SRC_POINTS = np.float32([
#     [int(640 * 0.20), int(640 * 0.9)],  # bottom_left (dekat roda kiri)
#     [int(640 * 0.80), int(640 * 0.9)],  # bottom_right (dekat roda kanan)
#     [int(640 * 0.60), int(640 * 0.55)], # top_right (jauh di depan, kanan)
#     [int(640 * 0.40), int(640 * 0.55)]  # top_left (jauh di depan, kiri)
# ])

# # Titik tujuan yang akan membentuk persegi panjang di BEV
# DST_POINTS = np.float32([
#     [int(BEV_WIDTH * 0.1), BEV_HEIGHT],      # bottom_left
#     [int(BEV_WIDTH * 0.9), BEV_HEIGHT],      # bottom_right
#     [int(BEV_WIDTH * 0.9), 0],               # top_right
#     [int(BEV_WIDTH * 0.1), 0]                # top_left
# ])

# # Hitung matriks transformasi perspektif sekali di awal
# M_perspective = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
# # Minv_perspective = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS) # Disimpan jika perlu inverse mapping
# # --- Akhir Konfigurasi BEV ---


# try:
#     while(True):
#         car.getData()
#         # sensors = car.getSensors() # Sensors tidak digunakan dalam kode ini, bisa di-komen atau dihapus

#         frame = car.getImage()
#         if frame is None or not frame.any(): # Pastikan frame tidak kosong
#             print("Failed to get image frame from simulator.")
#             time.sleep(0.1) # Beri sedikit jeda sebelum mencoba lagi
#             continue

#         # Resize frame untuk konsistensi
#         frame = cv2.resize(frame, (640, 640))


#         # --- Pengolahan ROI (Region of Interest) ---
#         # Tentukan koordinat ROI (sesuaikan ini!)
#         # Misalnya, fokus pada bagian bawah gambar yang berisi jalan.
#         # Anda bisa mengatur ini melalui trackbar jika ingin dinamis.
#         roi_start_y = int(frame.shape[0] * 0.6) # Mulai dari 60% tinggi gambar
#         roi_end_y = frame.shape[0]             # Sampai bagian bawah gambar
#         roi_start_x = 0                        # Dari kiri
#         roi_end_x = frame.shape[1]             # Sampai kanan

#         # Potong frame untuk mendapatkan ROI
#         # Penting: Jika BEV dilakukan pada ROI, titik sumber BEV harus relatif terhadap frame_roi
#         # Untuk kesederhanaan, kita akan lakukan BEV pada frame penuh terlebih dahulu,
#         # dan ROI digunakan hanya untuk tampilan/pemotongan visual.
#         frame_roi_display = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x].copy()
#         # --- Akhir Pengolahan ROI ---


#         # --- Terapkan Transformasi Bird's-Eye View (BEV) ---
#         # Transformasikan frame asli ke tampilan BEV
#         warped_frame = cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR)
#         # --- Akhir BEV ---


#         # --- Pemrosesan Warna pada Bird's-Eye View Frame ---
#         # Sekarang semua pemrosesan (HSV, equalisasi, morfologi, inRange) diterapkan pada `warped_frame`
#         white_hsv_value = white_trackbar.get_values_list()

#         lower_white = np.array([white_hsv_value[0], white_hsv_value[1], white_hsv_value[2]])
#         upper_white = np.array([white_hsv_value[3], white_hsv_value[4], white_hsv_value[5]])

#         hsv_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)

#         # Pisahkan channel HSV dan terapkan histogram equalization pada Value
#         h, s, v = cv2.split(hsv_warped)
#         v_eq = cv2.equalizeHist(v)
#         hsv_eq = cv2.merge([h, s, v_eq])

#         # Kernel untuk operasi morfologi
#         kernel = np.ones((3,3), np.uint8)
#         # Morphological Opening: Menghilangkan noise kecil
#         hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_OPEN, kernel)
#         # Morphological Closing: Mengisi lubang kecil
#         hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_CLOSE, kernel)

#         # Buat mask putih berdasarkan rentang HSV
#         white_mask_bev = cv2.inRange(hsv_eq, np.array(lower_white), np.array(upper_white))
#         # --- Akhir Pemrosesan Warna ---


#         # --- Tampilan Hasil ---
#         cv2.imshow('Original Frame (640x640)', frame)
#         cv2.imshow('ROI (for reference)', frame_roi_display) # Menampilkan area ROI yang Anda tentukan
#         cv2.imshow('Bird Eye View', warped_frame)
#         cv2.imshow('White Mask (from BEV)', white_mask_bev)

#         # Kontrol keluar
#         if cv2.waitKey(10) == ord('q'):
#             break

# finally:
#     car.stop()
#     cv2.destroyAllWindows() # Pastikan semua jendela OpenCV ditutup
'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

import time
import cv2
import numpy as np

import config
from engine import avisengine
from trackbar_utils import TrackbarManager


# Inisialisasi Trackbar
# Hanya inisialisasi trackbar yang diperlukan (white_trackbar)
white_trackbar = TrackbarManager("white_trackbar")
white_trackbar.add_multiple(config.WHITE_SETTING)

# Trackbar untuk 'dirt_trackbar' dan 'yellow_trackbar' telah dihapus


# Koneksi ke Simulator
car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

time.sleep(3) # Beri waktu untuk koneksi


# --- Konfigurasi Bird's-Eye View (BEV) ---

# Menggunakan ukuran dan titik tujuan yang Anda berikan
BEV_WIDTH = 640
BEV_HEIGHT = 480

# Menggunakan titik-titik yang Anda berikan untuk transformasi perspektif
tl = (int(BEV_WIDTH * 0.35), int(BEV_HEIGHT * 0.5))
bl = (int(0), int(BEV_HEIGHT * 0.85))
tr = (int(BEV_WIDTH * 0.65), int(BEV_HEIGHT * 0.5))
br = (int(BEV_WIDTH), int(BEV_HEIGHT * 0.85))

# Titik sumber (pts1) - menggunakan titik-titik Anda
SRC_POINTS = np.float32([tl, bl, tr, br])

# Titik tujuan (pts2) - menggunakan definisi Anda untuk tampilan BEV
DST_POINTS = np.float32([[0, 0], [0, BEV_HEIGHT], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT]])

# Hitung matriks transformasi perspektif sekali di awal
M_perspective = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
Minv_perspective = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS) # Matriks invers untuk proyeksi balik

# Fungsi untuk memvisualisasikan titik-titik (opsional, untuk debugging)
def draw_perspective_points(frame_to_draw_on, tl_pt, bl_pt, tr_pt, br_pt):
    # Pastikan koordinat yang masuk ke cv2.circle juga integer
    cv2.circle(frame_to_draw_on, (int(tl_pt[0]), int(tl_pt[1])), 5, (0, 0, 255), -1) # Merah
    cv2.circle(frame_to_draw_on, (int(bl_pt[0]), int(bl_pt[1])), 5, (0, 0, 255), -1) # Merah
    cv2.circle(frame_to_draw_on, (int(tr_pt[0]), int(tr_pt[1])), 5, (0, 0, 255), -1) # Merah
    cv2.circle(frame_to_draw_on, (int(br_pt[0]), int(br_pt[1])), 5, (0, 0, 255), -1) # Merah
# --- Akhir Konfigurasi BEV ---


try:
    while(True):
        car.getData()
        frame = car.getImage()
        if frame is None or not frame.any():
            print("Failed to get image frame from simulator.")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 640))

        # --- ROI (Display Only) ---
        # Bagian ini tetap menggunakan persentase seperti sebelumnya,
        # karena ini adalah ROI terpisah dari transformasi perspektif.
        roi_start_y = int(frame.shape[0] * 0.6)
        roi_end_y = frame.shape[0]
        roi_start_x = 0
        roi_end_x = frame.shape[1]
        frame_roi_display = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x].copy()


        # --- 1. Terapkan Transformasi Bird's-Eye View (BEV) ---
        # Gambar titik-titik di frame asli sebelum di-warp
        draw_perspective_points(frame, tl, bl, tr, br)
        warped_frame = cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR)


        # --- 2. Segmentasi Warna (Masking) pada BEV Frame - HANYA SATU WARNA ---
        hsv_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_warped)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge([h, s, v_eq])

        kernel = np.ones((3,3), np.uint8)
        hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_OPEN, kernel)
        hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_CLOSE, kernel)

        white_hsv_value = white_trackbar.get_values_list()
        lower_white = np.array([white_hsv_value[0], white_hsv_value[1], white_hsv_value[2]])
        upper_white = np.array([white_hsv_value[3], white_hsv_value[4], white_hsv_value[5]])

        binary_warped = cv2.inRange(hsv_eq, lower_white, upper_white)
        # --- Akhir Segmentasi Warna ---


        # --- 3. Analisis Histogram ---
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        # --- 4. Pencarian Sliding Window ---
        nwindows = 9
        margin = 100
        minpix = 50

        window_height = np.int32(binary_warped.shape[0] // nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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

        # --- 5. Fit Polinomial ---
        left_fit = None
        right_fit = None
        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        # --- 6. Overlay Jalur ke BEV ---
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            lane_area_bev = np.zeros_like(out_img)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(lane_area_bev, np.int_([pts]), (0,255,0))

            result_bev = cv2.addWeighted(warped_frame, 1, lane_area_bev, 0.3, 0)

            # --- Proyeksi Kembali ke Original Frame (Opsional) ---
            new_warp = np.zeros_like(result_bev).astype(np.uint8)
            cv2.fillPoly(new_warp, np.int_([pts]), (0,255,0))
            unwarped_lane = cv2.warpPerspective(new_warp, Minv_perspective, (frame.shape[1], frame.shape[0]))
            result_original = cv2.addWeighted(frame, 1, unwarped_lane, 0.3, 0)

            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', result_bev)
        else:
            cv2.imshow('Lanes on Bird Eye View', out_img)
            print("No valid lane lines detected for fitting.")

        # --- Tampilan Hasil ---
        cv2.imshow('Original Frame (640x640)', frame) # Sekarang juga akan menampilkan titik-titik kalibrasi
        cv2.imshow('ROI (for reference)', frame_roi_display)
        cv2.imshow('Bird Eye View (Raw)', warped_frame)
        cv2.imshow('Binary Mask (from BEV)', binary_warped)
        cv2.imshow('Sliding Window Debug', out_img)


        # Kontrol keluar
        if cv2.waitKey(10) == ord('q'):
            break

finally:
    car.stop()
    cv2.destroyAllWindows()