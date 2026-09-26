"""Live pass profiles for the motors, decided as frames arrive.

Each pass is announced ~0.2 s after the touch as a start time t0, start position, start
velocity v0 and slow-down rate, so the motors can follow
    speed(t) = max(0, |v0| - decel * (t - t0))
which never rises within a pass. mode="constant" gives one fixed speed per pass instead
(|v0| * avg_factor, the typical average/start ratio of real passes).
"""
from collections import deque

import numpy as np


class LivePasses:
    def __init__(self, measure_s=0.12, confirm_frames=2, dev_m=0.7, decel=7.0, min_speed=1.0,
                 max_gap_s=0.3, mode="profile", avg_factor=0.63, same_dir_deg=30.0, kick_rise=2.0):
        self.measure_s, self.confirm_frames, self.dev_m = measure_s, confirm_frames, dev_m
        self.decel, self.min_speed, self.max_gap = decel, min_speed, max_gap_s
        self.mode, self.avg_factor = mode, avg_factor
        self.same_dir_deg, self.kick_rise = same_dir_deg, kick_rise
        self.recent = deque(maxlen=60)   # (t, xy)
        self.current_pass = None         # dict(t0, p0, v0, announced)
        self.touch_pending = None        # dict(t_touch) while the new pass is being measured
        self.frames_off_path = 0
        self.last_t = None

    def _speed_at(self, pass_info, t):
        start_speed = np.hypot(*pass_info["v0"])
        if self.mode == "constant":
            return start_speed * self.avg_factor
        return max(0.0, start_speed - self.decel * (t - pass_info["t0"]))

    def _position_at(self, pass_info, t):
        start_speed = np.hypot(*pass_info["v0"])
        if start_speed < 1e-6:
            return pass_info["p0"].copy()
        direction = pass_info["v0"] / start_speed
        t_stop = start_speed / self.decel if self.decel > 0 else np.inf
        dt = min(t - pass_info["t0"], t_stop)
        return pass_info["p0"] + direction * (start_speed * dt - 0.5 * self.decel * dt * dt)

    def velocity(self, t):
        """Motor velocity (vx, vy) at time t, or None when no pass is known."""
        pass_info = self.current_pass
        if pass_info is None:
            return None
        start_speed = np.hypot(*pass_info["v0"])
        if start_speed < 1e-6:
            return (0.0, 0.0)
        direction = pass_info["v0"] / start_speed
        speed = self._speed_at(pass_info, t)
        return (float(direction[0] * speed), float(direction[1] * speed))

    def push(self, t, xy, restarted=False):
        """Feed one frame's raw ground position (None if none). Returns the new pass as a
        dict when one is announced, else None."""
        if self.last_t is not None and t - self.last_t > self.max_gap:
            self.current_pass = self.touch_pending = None
            self.recent.clear()
        if restarted:
            self.recent.clear()
        self.last_t = t if xy is not None else self.last_t
        if xy is None:
            return None
        xy = np.asarray(xy, float)
        self.recent.append((t, xy))

        if self.touch_pending is None:
            touch = restarted or self.current_pass is None
            if not touch:
                off_path = np.linalg.norm(xy - self._position_at(self.current_pass, t)) > self.dev_m
                self.frames_off_path = self.frames_off_path + 1 if off_path else 0
                touch = self.frames_off_path >= self.confirm_frames
            if touch:
                # date the touch to when the ball first left the predicted path
                back = 0 if restarted or self.current_pass is None else self.frames_off_path - 1
                t_touch = self.recent[-1 - back][0] if len(self.recent) > back else t
                self.touch_pending = dict(t_touch=t_touch)
                self.frames_off_path = 0

        if self.touch_pending is not None and t - self.touch_pending["t_touch"] >= self.measure_s:
            t_touch = self.touch_pending["t_touch"]
            since_touch = [(ts, pos) for ts, pos in self.recent if ts >= t_touch - 1e-6]
            if len(since_touch) >= 3:
                ts = np.array([s[0] for s in since_touch]) - t_touch
                positions = np.array([s[1] for s in since_touch])
                fit_x, fit_y = (np.polyfit(ts, positions[:, k], 1) for k in range(2))
                v0 = np.array([fit_x[0], fit_y[0]])
                if np.hypot(*v0) < self.min_speed:
                    v0 = np.zeros(2)
                previous = self.current_pass
                if previous is not None and np.hypot(*v0) > 0 and np.hypot(*previous["v0"]) > 0:
                    expected = self._speed_at(previous, t_touch)
                    new_speed = np.hypot(*v0)
                    same_dir = ((v0 @ previous["v0"]) / (new_speed * np.hypot(*previous["v0"]))
                                > np.cos(np.radians(self.same_dir_deg)))
                    if same_dir and new_speed <= expected + self.kick_rise:
                        # same pass drifting from the standard slow-down, not a new kick:
                        # re-anchor the path but never let the motors speed up
                        v0 = v0 / new_speed * min(new_speed, expected)
                p0 = np.array([fit_x[1], fit_y[1]])
                self.current_pass = dict(t0=t_touch, p0=p0, v0=v0, announced=t)
                self.touch_pending = None
                return dict(t0=t_touch, announced=t, x0=float(p0[0]), y0=float(p0[1]),
                            vx0=float(v0[0]), vy0=float(v0[1]), speed0=float(np.hypot(*v0)),
                            decel=self.decel if self.mode == "profile" else 0.0,
                            speed_const=float(np.hypot(*v0) * self.avg_factor))
            if t - t_touch > self.measure_s + 0.2:
                self.touch_pending = None
        return None
