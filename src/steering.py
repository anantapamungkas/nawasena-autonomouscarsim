# steering.py

from controller import traffic_lamp, road_sign
from controller import turn_the_car

def process_labels(labels, car, side_pix=None):
    """
    Memproses label hasil deteksi YOLO dan memberikan perintah ke mobil.
    
    Args:
    - labels: list str dari hasil deteksi (["left", "red lamp", ...])
    - car: objek mobil AVIS Engine
    - side_pix: posisi rata-rata garis sisi untuk membantu keputusan belok
    """

    # Prioritas 1: Lampu lalu lintas
    for label in labels:
        if label in ["red lamp", "green lamp"]:
            traffic_lamp(label, car)
            return

    # Prioritas 2: Rambu lalu lintas
    for label in labels:
        sign_state = label  # bisa saja "left", "right", "straight", "stop"

        if sign_state == 'left':
            if side_pix is not None and side_pix > 128:
                turn_the_car(car, -45, 13)
            else:
                turn_the_car(car, -50, 12)

        elif sign_state == 'straight':
            turn_the_car(car, 0, 11)

        elif sign_state == 'right':
            if side_pix is not None and side_pix > 128:
                turn_the_car(car, 65, 9.5)
            else:
                turn_the_car(car, 70, 11)

        elif sign_state == 'stop':
            car.setSpeed(0)
            print("🛑 Rambu Stop: Mobil berhenti.")

        break  # hanya proses satu sign
