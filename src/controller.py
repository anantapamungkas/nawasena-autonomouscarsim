# controller.py
import time

def traffic_lamp(label, car):
    if label == "red lamp":
        car.setSpeed(0)
        print("🚦 Lampu Merah: Mobil berhenti.")
    elif label == "green lamp":
        car.setSpeed(30)
        print("🚦 Lampu Hijau: Mobil jalan.")


def road_sign(label, car):
    if label == "straight":
        car.setSpeed(30)
        print("➡️ Lurus.")
    elif label == "left":
        car.setSteering(-25)
        print("⬅️ Belok kiri.")
    elif label == "right":
        car.setSteering(25)
        print("➡️ Belok kanan.")
    elif label == "stop":
        car.setSpeed(0)
        print("🛑 Rambu Stop: Mobil berhenti.")

def turn_the_car(car, steering_angle, duration):
    """
    Menggerakkan mobil dengan sudut steering tertentu selama durasi tertentu.

    Parameters:
    - car: objek mobil dari AVIS Engine
    - steering_angle: int, sudut kemudi (negatif: kiri, positif: kanan, 0: lurus)
    - duration: float, durasi belok dalam detik
    """
    print(f"🔁 Belok: Angle = {steering_angle}, Durasi = {duration}s")
    
    start_time = time.time()
    
    # Set steering dan mulai bergerak
    car.setSteering(steering_angle)
    car.setSpeed(15)  # atur sesuai kecepatan optimal kamu

    # Tunggu hingga durasi tercapai sambil terus update data
    while time.time() - start_time < duration:
        car.getData()
        time.sleep(0.01)  # agar tidak terlalu membebani CPU

    # Setelah selesai, kembalikan steering ke tengah dan hentikan
    car.setSteering(0)
    car.setSpeed(0)

def go_back(car, duration):
    """
    Memundurkan mobil selama durasi tertentu.
    """
    print(f"⬅️ Mundur selama {duration}s")
    start_time = time.time()
    while time.time() - start_time < duration:
        car.setSpeed(-15)
        car.getData()
        time.sleep(0.01)
    car.setSpeed(0)

def handle_dummy_obstacle(car, side_pix, utils):
    if car.getSensors()[1] < 700:
        print("🧱 Dummy terdeteksi! Berhenti dan lakukan manuver.")
        ret = utils.stop_the_car(car)
        print('📏 side_pix:', side_pix)
        time.sleep(3)

        if side_pix > 128:
            utils.turn_the_car(car, -100, 5.5)
            utils.turn_the_car(car, 100, 6.5)
            utils.turn_the_car(car, -100, 2.5)
        else:
            utils.turn_the_car(car, 100, 4)
        return True
    return False

