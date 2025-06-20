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
    ("b_low", 0, 255), ("g_low", 0, 255), ("r_low", 102, 255),
    ("b_high", 104, 255), ("g_high", 148, 255), ("r_high", 180, 255)
]

# Setting untuk trackbar warna kuning dalam format BGR
YELLOW_SETTING = [
    ("b_low", 0, 255), ("g_low", 53, 255), ("r_low", 91, 255),
    ("b_high", 98, 255), ("g_high", 183, 255), ("r_high", 206, 255)
]

# # Setting untuk trackbar warna tambahan (misal: Biru) dalam format BGR
# ADDITIONAL_COLOR_SETTING = [
#     ("b_low", 150, 255), ("g_low", 0, 255), ("r_low", 0, 255),
#     ("b_high", 255, 255), ("g_high", 100, 255), ("r_high", 100, 255)
# ]

#HSL
# WHITE_SETTING = [
#     ("h_low", 0, 180), ("l_low", 200, 255), ("s_low", 0, 255),
#     ("h_high", 180, 180), ("l_high", 255, 255), ("s_high", 255, 255)
# ]

# YELLOW_SETTING = [
#     ("h_low", 15, 180), ("l_low", 30, 255), ("s_low", 100, 255),
#     ("h_high", 35, 180), ("l_high", 204, 255), ("s_high", 255, 255)
# ]