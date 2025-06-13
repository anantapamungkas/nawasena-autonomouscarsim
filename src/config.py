'''
@ 2022, Copyright AVIS Engine
'''

SIMULATOR_IP = "127.0.0.1"
SIMULATOR_PORT = 25001


# WHITE_SETTING = [
#     ("l_h", 2, 255), ("l_s", 19, 255), ("l_v", 229, 255),
#     ("u_h", 44, 255), ("u_s", 74, 255), ("u_v", 255, 255)
# ]

# BROWN_SETTING = [
#     ("l_h", 14, 255), ("l_s", 51, 255), ("l_v", 200, 255),
#     ("u_h", 19, 255), ("u_s", 144, 255), ("u_v", 255, 255)
# ]

# YELLOW_SETTING = [
#     ("l_h", 9, 255), ("l_s", 97, 255), ("l_v", 232, 255),
#     ("u_h", 169, 255), ("u_s", 255, 255), ("u_v", 215, 255)
# ]
# config.py

# '''
# @ 2022, Copyright AVIS Engine
# '''

# SIMULATOR_IP = "127.0.0.1"
# SIMULATOR_PORT = 25001

# Setting untuk trackbar warna putih dalam format BGR
# Format: ("nama", nilai_default, nilai_maksimum)
WHITE_SETTING = [
    ("b_low", 0, 255), ("g_low", 0, 255), ("r_low", 0, 255),
    ("b_high", 65, 255), ("g_high", 106, 255), ("r_high", 255, 255)
]

# Setting untuk trackbar warna kuning dalam format BGR
YELLOW_SETTING = [
    ("b_low", 0, 255), ("g_low", 150, 255), ("r_low", 150, 255),
    ("b_high", 100, 255), ("g_high", 255, 255), ("r_high", 255, 255)
]

# Setting untuk trackbar warna tambahan (misal: Biru) dalam format BGR
ADDITIONAL_COLOR_SETTING = [
    ("b_low", 150, 255), ("g_low", 0, 255), ("r_low", 0, 255),
    ("b_high", 255, 255), ("g_high", 100, 255), ("r_high", 100, 255)
]