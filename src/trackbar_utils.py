import cv2

def nothing(x):
    pass

class TrackbarManager:
    def __init__(self, window_name="Trackbars"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.trackbar_configs = []

    def add_trackbar(self, name, default, max_val=255):
        cv2.createTrackbar(name, self.window_name, default, max_val, nothing)
        self.trackbar_configs.append(name)

    def add_multiple(self, config_list):
        """
        config_list: list of tuples (name, default_value, max_value)
        """
        for name, default, max_val in config_list:
            self.add_trackbar(name, default, max_val)

    def get_values(self):
        return {name: cv2.getTrackbarPos(name, self.window_name) for name in self.trackbar_configs}

    def get_values_list(self):
        return [cv2.getTrackbarPos(name, self.window_name) for name in self.trackbar_configs]