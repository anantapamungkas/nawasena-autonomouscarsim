'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''


import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config 
from engine import avisengine
from trackbar_utils import TrackbarManager


white_trackbar = TrackbarManager("white_trackbar")
white_trackbar.add_multiple(config.WHITE_SETTING)

dirt_trackbar = TrackbarManager("dirt_trackbar")
dirt_trackbar.add_multiple(config.DIRT_SETTING)

yellow_trackbar = TrackbarManager("yellow_trackbar")
yellow_trackbar.add_multiple(config.YELLOW_SETTING)

car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

time.sleep(3)

try:
    while(True):
        car.getData()
        sensors = car.getSensors() 
        
        frame = car.getImage()
        frame = cv2.resize(frame, (640, 640))


        white_hsv_value = white_trackbar.get_values_list()

        lower = np.array([white_hsv_value[0], white_hsv_value[1], white_hsv_value[2]])
        upper = np.array([white_hsv_value[3], white_hsv_value[4], white_hsv_value[5]])

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv_frame)

        # Apply histogram equalization to V
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge([h, s, v_eq])

        kernel = np.ones((3,3), np.uint8)
        hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_OPEN, kernel)  # Removes small noise
        hsv_eq = cv2.morphologyEx(hsv_eq, cv2.MORPH_CLOSE, kernel) # Fills small holes



        white_mask = cv2.inRange(hsv_eq, np.array(lower), np.array(upper))

        # ## --- Gradient Threshold (Sobel) ---
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Compute gradients
        # sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # # Magnitude
        # gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        # gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

        # # Threshold
        # _, sobel_thresh = cv2.threshold(gradient_magnitude, 25, 255, cv2.THRESH_BINARY)

        # l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        # l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        # u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        # lower = np.array([l_h,l_s,l_v])
        # upper = np.array([u_h,u_s,u_v])

        # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # white_mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))

        # combined = cv2.bitwise_or(sobel_thresh, white_mask)

        
        if frame is not None and frame.any():
            cv2.imshow('frames', frame)
        #     cv2.imshow('Sobel Gradient', sobel_thresh)
            cv2.imshow('white_mask', white_mask)
        #     cv2.imshow('combine_mask', combined)

        if cv2.waitKey(10) == ord('q'):
            break

finally:
    car.stop()