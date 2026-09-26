"""Motor speed signal: a jump at each touch, then only level or falling until the next touch.

Speed comes from a straight-line fit of the raw ground positions over +-half_window, so each
value describes the ball `delay` seconds earlier. A touch (tracking restart, a speed jump of
rise_touch m/s, or a turn over turn_touch_deg) starts a new pass; for onset_s after it the
speed may still rise while the kick is measured, after that it can only fall.
"""
from collections import deque

import numpy as np


class MotorSpeed:
    def __init__(self, half_window=0.12, delay=0.15, rise_touch=3.0, turn_touch_deg=35.0,
                 onset_s=0.25, min_speed=1.0, max_gap=0.3):
        self.half_window, self.delay = half_window, max(delay, half_window)
        self.rise_touch, self.turn_cos = rise_touch, np.cos(np.radians(turn_touch_deg))
        self.onset_s, self.min_speed, self.max_gap = onset_s, min_speed, max_gap
        self.samples = deque()        # (t, xy or None, restarted)
        self.queued_times = deque()   # times whose output isn't ready yet
        self.speed = None
        self.direction = None
        self.touch_t = -1e9
        self.last_valid = -1e9

    def _fit_velocity(self, t_out):
        """Velocity at t_out from raw points within +-half_window, not crossing a tracking restart."""
        window = [(ts, xy, restarted) for ts, xy, restarted in self.samples
                  if abs(ts - t_out) <= self.half_window and xy is not None]
        restart_times = [ts for ts, _, restarted in window if restarted and ts > t_out - self.half_window]
        restarts_before = [ts for ts in restart_times if ts <= t_out]
        if restarts_before:
            window = [s for s in window if s[0] >= max(restarts_before)]
        elif restart_times:
            window = [s for s in window if s[0] < min(restart_times)]
        if len(window) < 4 or window[-1][0] - window[0][0] < self.half_window:
            return None
        ts = np.array([s[0] for s in window]) - t_out
        xy = np.array([s[1] for s in window])
        return np.array([np.polyfit(ts, xy[:, k], 1)[0] for k in range(2)])

    def push(self, t, xy, restarted=False):
        """Feed one frame; returns the outputs that became ready, each dict(t, speed, vx, vy, touch)."""
        self.samples.append((t, None if xy is None else np.asarray(xy, float), bool(restarted)))
        self.queued_times.append((t, bool(restarted)))
        while self.samples and self.samples[0][0] < t - self.delay - 2 * self.half_window - 0.5:
            self.samples.popleft()
        ready = []
        while self.queued_times and self.queued_times[0][0] <= t - self.delay:
            t_out, restarted_then = self.queued_times.popleft()
            ready.append(self._step(t_out, restarted_then))
        return ready

    def _step(self, t_out, restarted):
        velocity = self._fit_velocity(t_out)
        touch = False
        if velocity is None:
            if t_out - self.last_valid > self.max_gap:
                self.speed = self.direction = None
            return dict(t=t_out, speed=self.speed, vx=self._vx(), vy=self._vy(), touch=False)
        self.last_valid = t_out
        speed_now = float(np.hypot(*velocity))
        heading = velocity / speed_now if speed_now > 1e-6 else None
        if self.speed is None or restarted:
            touch = True
        elif speed_now > self.speed + self.rise_touch:
            touch = True
        elif heading is not None and self.direction is not None and speed_now > self.min_speed and \
                self.speed > self.min_speed and float(heading @ self.direction) < self.turn_cos:
            touch = True
        if touch:
            self.speed, self.touch_t = speed_now, t_out
        elif t_out - self.touch_t <= self.onset_s:
            self.speed = max(self.speed, speed_now)   # still measuring the kick
        else:
            self.speed = min(self.speed, speed_now)   # rolling: never speeds up mid-pass
        if heading is not None and speed_now > self.min_speed:
            self.direction = heading if touch or self.direction is None else self._blend(self.direction, heading)
        return dict(t=t_out, speed=self.speed, vx=self._vx(), vy=self._vy(), touch=touch)

    @staticmethod
    def _blend(a, b, w=0.3):
        mixed = (1 - w) * a + w * b
        return mixed / (np.linalg.norm(mixed) or 1)

    def _vx(self):
        return None if self.speed is None or self.direction is None else float(self.speed * self.direction[0])

    def _vy(self):
        return None if self.speed is None or self.direction is None else float(self.speed * self.direction[1])
