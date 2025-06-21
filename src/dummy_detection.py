'''
@ 2025, Copyright AVIS Engine
- 15 Juni 2025
'''
import avisengine
import config
import time
import cv2
import numpy as np

# Parameter kontrol
SAFE_DISTANCE = 500
FRONT_DISTANCE = 800
EMERGENCY_STOP_DIST = 400
STOP_TIMEOUT = 10
RETURN_THRESHOLD = 200
STEER_GAIN = 0.4
BACKUP_SPEED = -15
BACKUP_DURATION = 2.0  # detik

class State:
    NORMAL, AVOIDING, RETURNING, EMERGENCY, RECOVERING, BACKING = range(6)

def main():
    car = avisengine.Car()
    
    if not car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT):
        print("Gagal terhubung ke simulator!")
        return

    try:
        current_state = State.NORMAL
        path_history = []
        stop_timer = 0
        backup_timer = 0
        current_steering = 0  # Simpan nilai steering terakhir

        while True:
            try:
                # 1. Pembacaan sensor
                car.getData()
                sensors = car.getSensors()
                left, mid, right = sensors[0], sensors[1], sensors[2]
                image = car.getImage()
                
                # 2. State Management
                if current_state == State.NORMAL:
                    if mid < EMERGENCY_STOP_DIST:
                        current_state = State.EMERGENCY
                        stop_timer = time.time()
                    elif mid < FRONT_DISTANCE:
                        current_state = State.AVOIDING
                        path_history = []
                
                # 3. Emergency Stop dengan Timeout
                elif current_state == State.EMERGENCY:
                    if time.time() - stop_timer > STOP_TIMEOUT:
                        current_state = State.BACKING
                        backup_timer = time.time()
                    elif mid > EMERGENCY_STOP_DIST * 1.2:
                        current_state = State.NORMAL
                
                # 4. Proses Recovery
                elif current_state == State.BACKING:
                    if time.time() - backup_timer < BACKUP_DURATION:
                        car.setSpeed(BACKUP_SPEED)
                        current_steering = 0
                        car.setSteering(current_steering)
                    else:
                        current_state = State.RECOVERING
                
                # 5. Action Management
                if current_state == State.NORMAL:
                    # Update path history
                    if len(path_history) < 10:
                        path_history.append((left, right))
                    else:
                        path_history.pop(0)
                        path_history.append((left, right))
                    
                    # Kontrol steering
                    current_steering = np.clip((right - left) * STEER_GAIN, -25, 25)
                    car.setSteering(current_steering)
                    car.setSpeed(30)
                
                elif current_state == State.AVOIDING:
                    current_steering = -25 if left > right else 25
                    car.setSteering(current_steering)
                    car.setSpeed(20)
                    
                    if mid > FRONT_DISTANCE * 1.2 and abs(left - right) < RETURN_THRESHOLD:
                        current_state = State.RETURNING
                
                elif current_state == State.RETURNING:
                    if path_history:
                        avg_diff = np.mean([p[1]-p[0] for p in path_history[-3:]])
                        current_steering = np.clip(avg_diff * 0.5, -20, 20)
                        car.setSteering(current_steering)
                        car.setSpeed(15)
                        
                        if abs(left - right) < 150:
                            current_state = State.NORMAL
                
                elif current_state == State.RECOVERING:
                    current_steering = np.clip(current_steering * 0.9, -25, 25)
                    car.setSteering(current_steering)
                    car.setSpeed(15)
                    
                    if abs(left - right) < RETURN_THRESHOLD:
                        current_state = State.NORMAL
                
                # 6. Visualisasi
                if image is not None:
                    status_text = [
                        "NORMAL", "MENGHINDAR", "KEMBALI KE JALUR", 
                        "BERHENTI DARURAT", "RECOVERY", "MUNDUR"
                    ][current_state]
                    
                    color = [
                        (0, 255, 0), (0, 165, 255), (255, 255, 0),
                        (0, 0, 255), (255, 0, 255), (255, 0, 0)
                    ][current_state]
                    
                    cv2.putText(image, f"STATUS: {status_text}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    if current_state == State.EMERGENCY:
                        elapsed = int(time.time() - stop_timer)
                        cv2.putText(image, f"WAKTU: {elapsed}/{STOP_TIMEOUT}s", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.putText(image, f"Steering: {current_steering:.1f}°", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    cv2.imshow('AVIS Navigation', image)
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break
                
    finally:
        try:
            car.setSpeed(0)
            car.setSteering(0)
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()