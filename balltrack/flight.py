"""Detecting when the ball is in the air."""
import numpy as np

PERSON_HEIGHT_M = 1.8


def vertical_scale(people, x, y, k=3):
    if not people:
        return None
    d = [np.hypot((a + c) / 2 - x, b2 - y) for a, _, c, b2 in people]
    near = [people[i] for i in np.argsort(d)[:k]]
    return float(np.median([b2 - b1 for _, b1, _, b2 in near])) / PERSON_HEIGHT_M


def at_feet(people, x, y):
    for x1, y1, x2, y2 in people:
        h, w = y2 - y1, x2 - x1
        if x1 - 0.3 * w <= x <= x2 + 0.3 * w and y2 - 0.2 * h <= y <= y2 + 0.1 * h:
            return True
    return False


class AirDetector:
    def __init__(self, window_s=0.4, min_samples=8, air_accel=(6.0, 25.0), land_accel=3.0,
                 max_rms_m=0.15, max_gap_s=0.12, max_air_s=3.0, min_rise=2.0):
        self.window_s, self.min_samples = window_s, min_samples
        self.air_accel, self.land_accel, self.max_rms_m = air_accel, land_accel, max_rms_m
        self.max_gap_s, self.max_air_s, self.min_rise = max_gap_s, max_air_s, min_rise
        self.reset()

    def reset(self):
        self.hist = []
        self.airborne = False
        self.since = None
        self.falling = False
        self.accel = None

    def break_track(self):
        self.hist = []

    def update(self, t, y_px, px_per_m, on_ground=False):
        if self.hist and t - self.hist[-1][0] > self.max_gap_s:
            self.hist = []
        self.hist.append((t, y_px, px_per_m))
        self.hist = [h for h in self.hist if t - h[0] <= self.window_s]

        self.accel = vy = v_first = None
        scales = [h[2] for h in self.hist if h[2]]
        if len(self.hist) >= self.min_samples and scales:
            ts = np.array([h[0] for h in self.hist]) - t
            ys = np.array([h[1] for h in self.hist])
            coef = np.polyfit(ts, ys, 2)
            ppm = float(np.median(scales))
            rms = float(np.sqrt(np.mean((np.polyval(coef, ts) - ys) ** 2))) / ppm
            if rms <= self.max_rms_m:
                self.accel = 2 * coef[0] / ppm
                vy = coef[1] / ppm
                v_first = (coef[1] + 2 * coef[0] * ts[0]) / ppm

        if not self.airborne:
            if (not on_ground and self.accel is not None and self.air_accel[0] < self.accel < self.air_accel[1]
                    and v_first < -self.min_rise):
                self.airborne, self.since, self.falling = True, t, False
        else:
            if vy is not None and vy > 1.0:
                self.falling = True
            bounced = self.falling and vy is not None and vy < 0
            flat = self.accel is not None and abs(self.accel) < self.land_accel
            if bounced or flat or on_ground or t - self.since > self.max_air_s:
                self.airborne = False
                self.hist = self.hist[-2:]
        return self.airborne
