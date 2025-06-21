# dummy_detector.py

import time
import numpy as np

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

class DummyDetector:
    def __init__(self):
        self.current_state = State.NORMAL
        self.path_history = []
        self.stop_timer = 0
        self.backup_timer = 0
        self.current_steering = 0

    def update(self, sensors, car):  # ✅ terima dua argumen
        left, mid, right = sensors[0], sensors[1], sensors[2]

        # --- logika state machine dari dummy detection sebelumnya
        if self.current_state == State.NORMAL:
            if mid < EMERGENCY_STOP_DIST:
                self.current_state = State.EMERGENCY
                self.stop_timer = time.time()
            elif mid < FRONT_DISTANCE:
                self.current_state = State.AVOIDING
                self.path_history = []

        elif self.current_state == State.EMERGENCY:
            if time.time() - self.stop_timer > STOP_TIMEOUT:
                self.current_state = State.BACKING
                self.backup_timer = time.time()
            elif mid > EMERGENCY_STOP_DIST * 1.2:
                self.current_state = State.NORMAL

        elif self.current_state == State.BACKING:
            if time.time() - self.backup_timer < BACKUP_DURATION:
                car.setSpeed(BACKUP_SPEED)
                self.current_steering = 0
                car.setSteering(self.current_steering)
            else:
                self.current_state = State.RECOVERING

        if self.current_state == State.NORMAL:
            if len(self.path_history) < 10:
                self.path_history.append((left, right))
            else:
                self.path_history.pop(0)
                self.path_history.append((left, right))

            self.current_steering = np.clip((right - left) * STEER_GAIN, -25, 25)
            car.setSteering(self.current_steering)
            car.setSpeed(30)

        elif self.current_state == State.AVOIDING:
            self.current_steering = -25 if left > right else 25
            car.setSteering(self.current_steering)
            car.setSpeed(20)
            if mid > FRONT_DISTANCE * 1.2 and abs(left - right) < RETURN_THRESHOLD:
                self.current_state = State.RETURNING

        elif self.current_state == State.RETURNING:
            if self.path_history:
                avg_diff = np.mean([p[1]-p[0] for p in self.path_history[-3:]])
                self.current_steering = np.clip(avg_diff * 0.5, -20, 20)
                car.setSteering(self.current_steering)
                car.setSpeed(15)
                if abs(left - right) < 150:
                    self.current_state = State.NORMAL

        elif self.current_state == State.RECOVERING:
            self.current_steering = np.clip(self.current_steering * 0.9, -25, 25)
            car.setSteering(self.current_steering)
            car.setSpeed(15)
            if abs(left - right) < RETURN_THRESHOLD:
                self.current_state = State.NORMAL

        return self.current_state
