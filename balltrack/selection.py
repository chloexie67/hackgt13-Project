"""Choosing which detected candidate is the ball."""
import numpy as np

from .pitch import to_pitch

class BallSelector:
    """Chooses which detection is the ball, and refuses one-frame jumps.

    A candidate is accepted straight away only if it is where the ball can plausibly be:
      * within the Kalman filter's uncertainty around the predicted ground position
        (the area grows while the ball is unseen and shrinks when tracking is steady), or
      * within a screen distance of the last sighting that grows with the time since it
        (covers a ball kicked into the air, whose ground position is meaningless).
    Anything else must earn it: it becomes a "challenger" and replaces the current ball
    only after it has been seen confirm_n frames in a row moving plausibly, and, if the
    current ball is still being seen, only if it is clearly more confident.
    With nothing tracked (start, after a cut or a long loss), every new ball needs that
    confirmation, so a single false hit anywhere on screen can never win.
    """

    def __init__(self, width, max_jump=0.6, gate_chi2=9.21, confirm_n=3, min_conf=0.25,
                 switch_margin=0.2, forget_s=3.0):
        self.max_jump_px_s = max_jump * width
        self.gate_chi2, self.confirm_n, self.min_conf = gate_chi2, confirm_n, min_conf
        self.switch_margin, self.forget_s = switch_margin, forget_s
        self.forget()

    def forget(self):
        """Nothing tracked any more (cut, close-up): the next ball must be confirmed."""
        self.confirmed_track = []
        self.last_px = self.last_t = None
        self.conf_ema = 0.0
        self.challenger = None   # {"px", "t", "n", "conf"}

    def _limit(self, dt):
        return self.max_jump_px_s * max(dt, 1 / 30)

    def choose(self, cands, t, kf, H):
        """Returns (candidate or None, switched). switched=True means the ball was
        re-acquired somewhere new, so the caller should restart its filters."""
        if self.last_t is not None and t - self.last_t > self.forget_s:
            self.forget()

        in_gate, out_gate = [], []
        for c in cands:
            dn = np.inf  # distance as a fraction of the allowed distance
            if self.last_px is not None:
                dn = np.hypot(c[0] - self.last_px[0], c[1] - self.last_px[1]) / self._limit(t - self.last_t)
            if kf.x is not None and H is not None:
                gx, gy = to_pitch(H, c[0], c[2])
                dn = min(dn, np.sqrt(kf.gate_distance2((gx, gy)) / self.gate_chi2))
            (in_gate if dn <= 1 else out_gate).append((c[3] - 0.5 * min(dn, 1), c))

        best = max(in_gate, key=lambda sc: sc[0])[1] if in_gate else None
        switched = False

        # challengers: the strongest candidate outside the gate, followed frame to frame
        strong = [c for _, c in out_gate if c[3] >= self.min_conf]
        ch = self.challenger
        if ch is not None and t - ch["t"] > 0.15:
            ch = None  # not seen again: it was a one-off
        if strong:
            c = max(strong, key=lambda c: c[3])
            if ch is not None and np.hypot(c[0] - ch["px"][0], c[1] - ch["px"][1]) <= self._limit(t - ch["t"]):
                ch = {"px": c[:2], "t": t, "n": ch["n"] + 1, "conf": ch["conf"] + c[3], "cand": c,
                      "seen": ch["seen"] + [(t, c)]}
            else:
                ch = {"px": c[:2], "t": t, "n": 1, "conf": c[3], "cand": c, "seen": [(t, c)]}
        self.challenger = ch

        if ch is not None and ch["t"] == t and ch["n"] >= self.confirm_n:
            avg = ch["conf"] / ch["n"]
            if best is None or avg >= self.conf_ema + self.switch_margin:
                best, switched = ch["cand"], True
                self.confirmed_track = ch["seen"][:-1]  # earlier sightings, e.g. for the air test
                self.challenger = None

        if best is not None:
            self.conf_ema = best[3] if switched or self.last_px is None else 0.8 * self.conf_ema + 0.2 * best[3]
            self.last_px, self.last_t = best[:2], t
        return best, switched
