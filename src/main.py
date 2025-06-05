import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from engine import avisengine

# Constants
FRAME_SIZE = 640

# Initialize and connect to the simulated car
car = avisengine.Car()
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Allow time for the simulator to fully load
time.sleep(3)

try:
    while True:
        # Update car data: sensors and camera image
        car.getData()
        sensors = car.getSensors()
        raw_frame = car.getImage()

        # Resize frame to fixed size
        resized_frame = cv2.resize(raw_frame, (FRAME_SIZE, FRAME_SIZE))
        display_frame = resized_frame.copy()

        #* Perspective transform source and destination points
        src_points = np.float32([config.TL, config.BL, config.TR, config.BR])
        dst_points = np.float32([
            [0, 0],
            [0, FRAME_SIZE],
            [FRAME_SIZE, 0],
            [FRAME_SIZE, FRAME_SIZE]
        ])

        # Compute perspective transformation matrix and warp frame
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_frame = cv2.warpPerspective(resized_frame, perspective_matrix, (FRAME_SIZE, FRAME_SIZE))

        # Convert warped frame to HSV and Lab color spaces for color segmentation
        hsv_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        yuv_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2YUV)


        #* Morphological opening and closing to remove noise and fill gaps
        morph_kernel = np.ones((3, 3), np.uint8)
        opened_hsv = cv2.morphologyEx(hsv_frame, cv2.MORPH_OPEN, morph_kernel)
        cleaned_hsv = cv2.morphologyEx(hsv_frame, cv2.MORPH_CLOSE, morph_kernel)


        #* Split HSV channels
        h_channel, s_channel, v_channel = cv2.split(cleaned_hsv)

        # Histogram equalization on the V channel to enhance contrast
        equalized_v = cv2.equalizeHist(v_channel)
        hsv_equalized = cv2.merge([h_channel, s_channel, equalized_v])


        #* Create masks for white, dirt, and yellow colors based on thresholds in config
        white_mask = cv2.inRange(hsv_equalized, config.LOWER_WHITE, config.UPPER_WHITE)
        dirt_mask = cv2.inRange(hsv_equalized, config.LOWER_DIRT, config.UPPER_DIRT)
        yellow_mask = cv2.inRange(yuv_frame, config.LOWER_YELLOW, config.UPPER_YELLOW)

        # Combine the color masks with bitwise OR
        combined_color_mask = cv2.bitwise_or(white_mask, dirt_mask)
        combined_color_mask = cv2.bitwise_or(combined_color_mask, yellow_mask)

        # Convert warped frame to grayscale for gradient detection
        gray_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Sobel gradients in x and y directions
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Normalize and convert to uint8
        gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

        # Threshold gradient magnitude to create binary edge mask
        _, sobel_threshold = cv2.threshold(gradient_magnitude, 25, 255, cv2.THRESH_BINARY)

        # Combine gradient edges with color mask using bitwise AND
        combined_mask = cv2.bitwise_and(sobel_threshold, combined_color_mask)

        # Show frames if available
        if resized_frame is not None and resized_frame.any():
            cv2.imshow('Original Frame', display_frame)
            cv2.imshow('Warped Frame', warped_frame)
            cv2.imshow('Combined Mask', combined_mask)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    # Stop the car connection and clean up windows
    car.stop()
    cv2.destroyAllWindows()
