'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''

import time
import cv2
import numpy as np

import config # Pastikan config.py memiliki WHITE_SETTING dan YELLOW_SETTING dalam format RGB
from engine import avisengine
from trackbar_utils import TrackbarManager


# Inisialisasi Trackbar
# PERHATIAN: Nilai-nilai di config.py sekarang harus dalam format RGB (B_low, G_low, R_low, B_high, G_high, R_high)
white_trackbar = TrackbarManager("white_trackbar")
white_trackbar.add_multiple(config.WHITE_SETTING)

yellow_trackbar = TrackbarManager("yellow_trackbar")
yellow_trackbar.add_multiple(config.YELLOW_SETTING)


# Koneksi ke Simulator
car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

time.sleep(3) # Beri waktu untuk koneksi


# --- Konfigurasi Bird's-Eye View (BEV) ---

BEV_WIDTH = 640
BEV_HEIGHT = 480

tl = (int(BEV_WIDTH * 0.33), int(BEV_HEIGHT * 0.4))
bl = (int(0), int(BEV_HEIGHT * 0.65))
tr = (int(BEV_WIDTH * 0.67), int(BEV_HEIGHT * 0.4))
br = (int(BEV_WIDTH), int(BEV_HEIGHT * 0.65))

SRC_POINTS = np.float32([tl, bl, tr, br])
DST_POINTS = np.float32([[0, 0], [0, BEV_HEIGHT], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT]])

M_perspective = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
Minv_perspective = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)

def draw_perspective_points(frame_to_draw_on, tl_pt, bl_pt, tr_pt, br_pt):
    cv2.circle(frame_to_draw_on, (int(tl_pt[0]), int(tl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(bl_pt[0]), int(bl_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(tr_pt[0]), int(tr_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame_to_draw_on, (int(br_pt[0]), int(br_pt[1])), 5, (0, 0, 255), -1)
# --- Akhir Konfigurasi BEV ---

# --- Konversi Piksel ke Meter (Kalibrasi Dunia Nyata) ---
XM_PER_PIX = 7.4 / BEV_WIDTH
YM_PER_PIX = 30.0 / BEV_HEIGHT
# --- Akhir Konversi ---

# --- Parameter Kontrol Mobil ---
BASE_SPEED = 30
MAX_SPEED = 50
MIN_SPEED = 10
ANGLE_THRESHOLD = 5
SPEED_REDUCTION_FACTOR = 1.5
# --- Akhir Parameter Kontrol ---

try:
    while(True):
        car.getData()
        frame = car.getImage()
        if frame is None or not frame.any():
            print("Failed to get image frame from simulator.")
            time.sleep(0.1)
            car.setSpeed(0)
            car.setSteering(0)
            continue

        frame = cv2.resize(frame, (640, 640))

        # --- ROI (Display Only) ---
        roi_start_y = int(frame.shape[0] * 0.6)
        roi_end_y = frame.shape[0]
        roi_start_x = 0
        roi_end_x = frame.shape[1]
        frame_roi_display = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x].copy()


        # --- 1. Terapkan Transformasi Bird's-Eye View (BEV) ---
        draw_perspective_points(frame, tl, bl, tr, br)
        warped_frame = cv2.warpPerspective(frame, M_perspective, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_LINEAR)


        # --- 2. Segmentasi Warna (Masking) pada BEV Frame - DUA WARNA (RGB) ---
        # warped_frame sudah dalam BGR, jadi tidak perlu konversi ruang warna

        # Segmentasi Warna Pertama (Putih)
        # Format trackbar: B_low, G_low, R_low, B_high, G_high, R_high
        white_rgb_value = white_trackbar.get_values_list()
        lower_white = np.array([white_rgb_value[0], white_rgb_value[1], white_rgb_value[2]])
        upper_white = np.array([white_rgb_value[3], white_rgb_value[4], white_rgb_value[5]])
        mask_white = cv2.inRange(warped_frame, lower_white, upper_white)

        # Segmentasi Warna Kedua (Kuning)
        yellow_rgb_value = yellow_trackbar.get_values_list()
        lower_yellow = np.array([yellow_rgb_value[0], yellow_rgb_value[1], yellow_rgb_value[2]])
        upper_yellow = np.array([yellow_rgb_value[3], yellow_rgb_value[4], yellow_rgb_value[5]])
        mask_yellow = cv2.inRange(warped_frame, lower_yellow, upper_yellow)

        # Gabungkan kedua masker dengan operasi Bitwise OR
        binary_warped = cv2.bitwise_or(mask_white, mask_yellow)

        # Opsional: Terapkan morphological operations setelah masking jika tujuannya menghaluskan mask
        kernel = np.ones((3,3), np.uint8)
        binary_warped = cv2.morphologyEx(binary_warped, cv2.MORPH_OPEN, kernel)
        binary_warped = cv2.morphologyEx(binary_warped, cv2.MORPH_CLOSE, kernel)
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

        curvature = 0.0
        lane_offset = 0.0
        steering_angle = 0.0
        current_speed = BASE_SPEED
        result_original = frame.copy()

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

            new_warp = np.zeros_like(result_bev).astype(np.uint8)
            cv2.fillPoly(new_warp, np.int_([pts]), (0,255,0))
            unwarped_lane = cv2.warpPerspective(new_warp, Minv_perspective, (frame.shape[1], frame.shape[0]))
            result_original = cv2.addWeighted(frame, 1, unwarped_lane, 0.3, 0)

            # --- Perhitungan Kelengkungan dan Offset ---
            center_fit = (left_fit + right_fit) / 2
            y_eval_pixels = BEV_HEIGHT - 1

            A_m = center_fit[0] * XM_PER_PIX / (YM_PER_PIX**2)
            B_m = center_fit[1] * XM_PER_PIX / YM_PER_PIX
            
            first_deriv = 2 * A_m * y_eval_pixels * YM_PER_PIX + B_m
            second_deriv = 2 * A_m

            if np.abs(second_deriv) < 1e-6:
                curvature = 10000.0
            else:
                curvature = ((1 + first_deriv**2)**1.5) / np.abs(second_deriv)

            car_center_x_pixels = BEV_WIDTH / 2
            lane_center_bottom_x_pixels = center_fit[0]*y_eval_pixels**2 + center_fit[1]*y_eval_pixels + center_fit[2]
            
            lane_offset_pixels = car_center_x_pixels - lane_center_bottom_x_pixels
            lane_offset = lane_offset_pixels * XM_PER_PIX

            # --- Perhitungan Sudut Kemudi ---
            if curvature == 0:
                steering_angle = np.arctan(lane_offset / 10000.0) * 180 / np.pi
            else:
                steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi
            
            steering_angle_command = steering_angle 

            # --- Kontrol Kecepatan Adaptif ---
            abs_steering_angle = np.abs(steering_angle_command)
            if abs_steering_angle > ANGLE_THRESHOLD:
                speed_reduction = (abs_steering_angle - ANGLE_THRESHOLD) * SPEED_REDUCTION_FACTOR
                current_speed = MAX_SPEED - speed_reduction
                current_speed = max(current_speed, MIN_SPEED)
            else:
                current_speed = MAX_SPEED
            
            current_speed = min(current_speed, MAX_SPEED)

            # --- Menggerakkan Mobil ---
            car.setSpeed(current_speed)
            car.setSteering(steering_angle_command)

            # --- Visualisasi Garis Kemudi di Frame Asli ---
            line_start_x = frame.shape[1] // 2
            line_start_y = frame.shape[0]

            line_length = 150
            end_x = int(line_start_x + line_length * np.sin(np.radians(steering_angle)))
            end_y = int(line_start_y - line_length * np.cos(np.radians(steering_angle)))

            cv2.line(result_original, (line_start_x, line_start_y), (end_x, end_y), (255, 0, 0), 3)

            # --- Tampilan Metrik di Frame Asli ---
            cv2.putText(result_original, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_original, f'Speed: {current_speed:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', result_bev)
        else:
            car.setSpeed(MIN_SPEED)
            car.setSteering(0)
            print("No valid lane lines detected for fitting. Moving slowly or stopping.")
            
            cv2.imshow('Lanes on Original Frame', result_original)
            cv2.imshow('Lanes on Bird Eye View', out_img)

        # --- Tampilan Hasil Debugging ---
        cv2.imshow('Original Frame (640x640)', frame)
        cv2.imshow('ROI (for reference)', frame_roi_display)
        cv2.imshow('Bird Eye View (Raw)', warped_frame)
        cv2.imshow('Binary Mask (from BEV) - Combined', binary_warped)
        cv2.imshow('Mask White', mask_white)
        cv2.imshow('Mask Yellow', mask_yellow)
        cv2.imshow('Sliding Window Debug', out_img)


        # Kontrol keluar
        if cv2.waitKey(10) == ord('q'):
            break

finally:
    car.stop()
    cv2.destroyAllWindows()