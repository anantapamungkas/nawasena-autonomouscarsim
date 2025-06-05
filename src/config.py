'''
@ 2022, Copyright AVIS Engine
'''

import numpy as np

SIMULATOR_IP = "127.0.0.1"
SIMULATOR_PORT = 25001


LOWER_WHITE = np.array([0,0,230])
UPPER_WHITE = np.array([29,69,255])

LOWER_DIRT = np.array([0,66,234])
UPPER_DIRT = np.array([28,109,255])

LOWER_YELLOW = np.array([87,0,149])
UPPER_YELLOW = np.array([109,137,187])

# region of interest settings 
TL = (213, 226)
BL = (0, 313)
TR = (396, 231)
BR = (640, 496)

WHITE_SETTING = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

DIRT_SETTING = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

YELLOW_SETTING = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

SETTING = [
    ("tl", 0, 640), ("bl", 0, 640), ("tr", 0, 640), ("br", 0, 640),
    ("tl_u", 0, 640), ("bl_u", 0, 640), ("tr_u", 0, 640), ("br_u", 0, 640)
]