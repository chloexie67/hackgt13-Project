"""Live pass detection for the motors: one clean speed profile per pass, decided as frames arrive.

A pass is announced shortly after the touch, then described completely by:
    start time t0, start position (x0, y0), start velocity (vx0, vy0), slowdown rate `decel`
so at any later time t the motors can use
    speed(t) = max(0, |v0| - decel * (t - t0))   along the direction of v0
which never rises within a pass. A new touch replaces the pass.

Timing (how late the announcement is, after the touch on screen):
    touch recognition  ~confirm_frames frames (ball leaves the predicted path, or tracking restarts)
    speed measurement   measure_s of positions after the touch
    + the tracker's own per-frame processing time.
With the defaults at 25 fps that is ~0.08 + 0.12 = 0.2 s before processing, keeping the total
under 0.3 s. During the pass there is no lag: the profile is known, so motors follow it live.

`mode="constant"` instead gives one constant speed per pass: |v0| * avg_factor (the typical
average/start ratio of real passes), for devices that can only hold one speed.

Usage (per frame):
    lp = LivePasses()
    for t, raw_xy, restarted in frames:
        ev = lp.push(t, raw_xy, restarted)   # a dict when a new pass is announced, else None
        v = lp.velocity(t)                   # (vx, vy) for the motors right now, or None
"""
from collections import deque

import numpy as np


class LivePasses:
    def __init__(self, measure_s=0.12, confirm_frames=2, dev_m=0.7, decel=7.0, min_speed=1.0,
                 max_gap_s=0.3, mode="profile", avg_factor=0.63, same_dir_deg=30.0, kick_rise=2.0):
        self.measure_s, self.confirm, self.dev = measure_s, confirm_frames, dev_m
        self.decel, self.min_speed, self.max_gap = decel, min_speed, max_gap_s
        self.mode, self.avg_factor = mode, avg_factor
        self.same_dir_deg, self.kick_rise = same_dir_deg, kick_rise
        self.buf = deque(maxlen=60)   # recent (t, xy)
        self.pass_ = None             # current pass: dict(t0, p0, v0, ...)
        self.pending = None           # touch seen, measuring: dict(t_touch)
        self.off_path = 0             # consecutive frames off the current pass's path
        self.last_t = None
        self.passes = []              # every announced pass (for logs / tests)

    # ---- the current pass's predicted motion ----
    def _speed_at(self, p, t):
        s0 = np.hypot(*p["v0"])
        if self.mode == "constant":
            return s0 * self.avg_factor
        return max(0.0, s0 - self.decel * (t - p["t0"]))

    def _pos_at(self, p, t):
        s0 = np.hypot(*p["v0"])
        if s0 < 1e-6:
            return p["p0"].copy()
        u = p["v0"] / s0
        dt = t - p["t0"]
        t_stop = s0 / self.decel if self.decel > 0 else np.inf
        dt = min(dt, t_stop)
        return p["p0"] + u * (s0 * dt - 0.5 * self.decel * dt * dt)

    def velocity(self, t):
        """Motor velocity (vx, vy) at time t, or None when no pass is known."""
        p = self.pass_
        if p is None:
            return None
        s0 = np.hypot(*p["v0"])
        if s0 < 1e-6:
            return (0.0, 0.0)
        u = p["v0"] / s0
        s = self._speed_at(p, t)
        return (float(u[0] * s), float(u[1] * s))

    # ---- feeding frames ----
    def push(self, t, xy, restarted=False):
        if self.last_t is not None and t - self.last_t > self.max_gap:
            self.pass_ = self.pending = None            # long gap: the old pass says nothing now
            self.buf.clear()
        if restarted:
            self.buf.clear()
        self.last_t = t if xy is not None else self.last_t
        if xy is None:
            return None
        xy = np.asarray(xy, float)
        self.buf.append((t, xy))

        if self.pending is None:
            touch = restarted or self.pass_ is None
            if not touch:
                off = np.linalg.norm(xy - self._pos_at(self.pass_, t)) > self.dev
                self.off_path = self.off_path + 1 if off else 0
                touch = self.off_path >= self.confirm
            if touch:
                # the touch happened about when the ball first left the path
                back = 0 if restarted or self.pass_ is None else self.off_path - 1
                t_touch = self.buf[-1 - back][0] if len(self.buf) > back else t
                self.pending = dict(t_touch=t_touch)
                self.off_path = 0

        if self.pending is not None and t - self.pending["t_touch"] >= self.measure_s:
            pts = [(tt, p) for tt, p in self.buf if tt >= self.pending["t_touch"] - 1e-6]
            if len(pts) >= 3:
                ts = np.array([q[0] for q in pts]) - self.pending["t_touch"]
                P = np.array([q[1] for q in pts])
                cx, cy = (np.polyfit(ts, P[:, k], 1) for k in range(2))
                v0 = np.array([cx[0], cy[0]])
                if np.hypot(*v0) < self.min_speed:
                    v0 = np.zeros(2)                    # ball essentially still / at feet
                old = self.pass_
                if old is not None and np.hypot(*v0) > 0 and np.hypot(*old["v0"]) > 0:
                    cur = self._speed_at(old, self.pending["t_touch"])
                    s_new = np.hypot(*v0)
                    same_dir = (v0 @ old["v0"]) / (s_new * np.hypot(*old["v0"])) > np.cos(np.radians(self.same_dir_deg))
                    if same_dir and s_new <= cur + self.kick_rise:
                        # the same pass drifting from the standard slow-down, not a new kick:
                        # re-anchor the path but never let the motors speed up
                        v0 = v0 / s_new * min(s_new, cur)
                p0 = np.array([cx[1], cy[1]])
                self.pass_ = dict(t0=self.pending["t_touch"], p0=p0, v0=v0, announced=t)
                self.passes.append(self.pass_)
                self.pending = None
                return dict(t0=self.pass_["t0"], announced=t, x0=float(p0[0]), y0=float(p0[1]),
                            vx0=float(v0[0]), vy0=float(v0[1]), speed0=float(np.hypot(*v0)),
                            decel=self.decel if self.mode == "profile" else 0.0,
                            speed_const=float(np.hypot(*v0) * self.avg_factor))
            if t - self.pending["t_touch"] > self.measure_s + 0.2:
                self.pending = None                     # couldn't measure: wait for the next touch
        return None
