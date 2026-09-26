"""Speed signal for the haptic motors: one clean speed profile per pass.

The tracker's Kalman velocity is fine for positions but not for motors: it can ramp up after a
kick and wobble up and down during a pass, which would feel like the ball speeding up and
slowing down several times. A real pass is a jump in speed at the touch followed by a steady
slow-down, so this signal enforces exactly that:

  * speed comes from a straight-line fit of the raw ground positions over +-half_window seconds,
    so it needs `delay` seconds of look-ahead: the value for time T is ready at T + delay;
  * a *touch* starts a new pass: tracking (re)starting, the speed jumping by rise_touch m/s
    above the current pass speed, or the direction turning by more than turn_touch_deg;
  * for onset_s after a touch the speed may still rise (the fit window straddles the kick);
    after that it can only stay level or fall until the next touch.

Usage (per frame, live or offline):
    ms = MotorSpeed()
    for t, raw_xy, restarted in frames:        # raw_xy None when there is no ground position
        for out in ms.push(t, raw_xy, restarted):
            ...  # out = dict(t, speed, vx, vy, touch) for time out["t"] (= t - delay)
"""
from collections import deque

import numpy as np


class MotorSpeed:
    def __init__(self, half_window=0.12, delay=0.15, rise_touch=3.0, turn_touch_deg=35.0,
                 onset_s=0.25, min_speed=1.0, max_gap=0.3):
        self.h, self.delay = half_window, max(delay, half_window)
        self.rise, self.turn = rise_touch, np.cos(np.radians(turn_touch_deg))
        self.onset, self.min_speed, self.max_gap = onset_s, min_speed, max_gap
        self.buf = deque()          # (t, xy or None, restarted)
        self.pending = deque()      # output times not yet emitted
        self.speed = None           # current pass speed (m/s)
        self.dir = None             # current pass direction (unit vector)
        self.touch_t = -1e9
        self.last_valid = -1e9

    def _fit(self, T):
        """Velocity at T from raw points within +-h, not crossing a tracking restart."""
        pts = [(t, xy, r) for t, xy, r in self.buf if abs(t - T) <= self.h and xy is not None]
        restarts = [t for t, xy, r in pts if r and t > T - self.h]
        if restarts:  # use only points after the most recent restart at or before T+h
            cut = max(t for t in restarts if t <= T) if any(t <= T for t in restarts) else None
            pts = [p for p in pts if cut is not None and p[0] >= cut] if cut is not None else \
                [p for p in pts if p[0] < min(restarts)]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < self.h:
            return None
        ts = np.array([p[0] for p in pts]) - T
        xy = np.array([p[1] for p in pts])
        return np.array([np.polyfit(ts, xy[:, k], 1)[0] for k in range(2)])

    def push(self, t, xy, restarted=False):
        self.buf.append((t, None if xy is None else np.asarray(xy, float), bool(restarted)))
        self.pending.append((t, bool(restarted)))
        while self.buf and self.buf[0][0] < t - self.delay - 2 * self.h - 0.5:
            self.buf.popleft()
        out = []
        while self.pending and self.pending[0][0] <= t - self.delay:
            T, restarted_T = self.pending.popleft()
            out.append(self._step(T, restarted_T))
        return out

    def _step(self, T, restarted):
        v = self._fit(T)
        touch = False
        if v is None:
            if T - self.last_valid > self.max_gap:
                self.speed = self.dir = None   # lost: no speed until the next clean sighting
            return dict(t=T, speed=self.speed, vx=self._vx(), vy=self._vy(), touch=False)
        self.last_valid = T
        s = float(np.hypot(*v))
        u = v / s if s > 1e-6 else None
        if self.speed is None or restarted:
            touch = True
        elif s > self.speed + self.rise:
            touch = True
        elif u is not None and self.dir is not None and s > self.min_speed and \
                self.speed > self.min_speed and float(u @ self.dir) < self.turn:
            touch = True
        if touch:
            self.speed, self.touch_t = s, T
        elif T - self.touch_t <= self.onset:
            self.speed = max(self.speed, s)          # still measuring the kick
        else:
            self.speed = min(self.speed, s)          # rolling: never speeds up mid-pass
        if u is not None and s > self.min_speed:
            self.dir = u if touch or self.dir is None else self._blend(self.dir, u)
        return dict(t=T, speed=self.speed, vx=self._vx(), vy=self._vy(), touch=touch)

    @staticmethod
    def _blend(a, b, w=0.3):
        c = (1 - w) * a + w * b
        return c / (np.linalg.norm(c) or 1)

    def _vx(self):
        return None if self.speed is None or self.dir is None else float(self.speed * self.dir[0])

    def _vy(self):
        return None if self.speed is None or self.dir is None else float(self.speed * self.dir[1])
