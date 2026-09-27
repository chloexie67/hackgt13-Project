"""Choosing which detected candidate is the ball."""
import numpy as np

from .pitch import to_pitch


class BallSelector:

    def __init__(self, width, max_jump=0.6, gate_chi2=9.21, confirm_n=3, min_conf=0.25,
                 switch_margin=0.2, forget_s=3.0):
        self.max_jump_px_s = max_jump * width
        self.gate_chi2, self.confirm_n, self.min_conf = gate_chi2, confirm_n, min_conf
        self.switch_margin, self.forget_s = switch_margin, forget_s
        self.forget()

    def forget(self):
        self.confirmed_track = []
        self.last_px = self.last_t = None
        self.conf_ema = 0.0
        self.challenger = None

    def _max_jump_px(self, dt):
        return self.max_jump_px_s * max(dt, 1 / 30)

    def choose(self, cands, t, kf, H):
        if self.last_t is not None and t - self.last_t > self.forget_s:
            self.forget()

        in_gate, out_gate = [], []
        for cand in cands:
            rel_dist = np.inf
            if self.last_px is not None:
                rel_dist = (np.hypot(cand[0] - self.last_px[0], cand[1] - self.last_px[1])
                            / self._max_jump_px(t - self.last_t))
            if kf.x is not None and H is not None:
                gx, gy = to_pitch(H, cand[0], cand[2])
                rel_dist = min(rel_dist, np.sqrt(kf.gate_distance2((gx, gy)) / self.gate_chi2))
            score = cand[3] - 0.5 * min(rel_dist, 1)
            (in_gate if rel_dist <= 1 else out_gate).append((score, cand))

        best = max(in_gate, key=lambda scored: scored[0])[1] if in_gate else None
        switched = False

        strong = [cand for _, cand in out_gate if cand[3] >= self.min_conf]
        challenger = self.challenger
        if challenger is not None and t - challenger["t"] > 0.15:
            challenger = None
        if strong:
            cand = max(strong, key=lambda c: c[3])
            continues = challenger is not None and np.hypot(
                cand[0] - challenger["px"][0], cand[1] - challenger["px"][1]) <= self._max_jump_px(t - challenger["t"])
            if continues:
                challenger = {"px": cand[:2], "t": t, "n": challenger["n"] + 1,
                              "conf": challenger["conf"] + cand[3], "cand": cand,
                              "seen": challenger["seen"] + [(t, cand)]}
            else:
                challenger = {"px": cand[:2], "t": t, "n": 1, "conf": cand[3], "cand": cand, "seen": [(t, cand)]}
        self.challenger = challenger

        if challenger is not None and challenger["t"] == t and challenger["n"] >= self.confirm_n:
            mean_conf = challenger["conf"] / challenger["n"]
            if best is None or mean_conf >= self.conf_ema + self.switch_margin:
                best, switched = challenger["cand"], True
                self.confirmed_track = challenger["seen"][:-1]
                self.challenger = None

        if best is not None:
            self.conf_ema = best[3] if switched or self.last_px is None else 0.8 * self.conf_ema + 0.2 * best[3]
            self.last_px, self.last_t = best[:2], t
        return best, switched
