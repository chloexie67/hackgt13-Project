"""Kalman filter for the ball's position and velocity on the pitch."""
import numpy as np

class BallKalman:
    def __init__(self, accel_std=25.0, meas_std=0.15, maneuver=False, kick_nis=13.8, kick_vel_std=10.0):
        self.accel_std = accel_std
        self.maneuver, self.kick_nis, self.kick_vel_std = maneuver, kick_nis, kick_vel_std
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.eye(2) * meas_std ** 2
        self.x = None
        self.P = None
        self.age = 0.0

    def reset(self):
        self.x = self.P = None
        self.age = 0.0

    def predict(self, dt):
        if self.x is None:
            return None
        self.age += dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.accel_std ** 2
        return self.x

    def gate_distance2(self, z):
        y = np.asarray(z, float) - self.x[:2]
        S = self.P[:2, :2] + self.R
        return float(y @ np.linalg.solve(S, y))

    def update(self, z):
        z = np.asarray(z, float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([1.0, 1.0, 100.0, 100.0])
            return self.x
        y = z - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        if self.maneuver and float(y @ np.linalg.solve(S, y)) > self.kick_nis:
            self.P[2:, 2:] += np.eye(2) * self.kick_vel_std ** 2
            S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.Hm) @ self.P
        return self.x
