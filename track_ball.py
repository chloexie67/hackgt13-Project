"""Locate the soccer ball on the pitch in broadcast footage (recorded or live).

Pipeline per frame:
    1. Scene filter   - pause on replays / close-ups / crowd shots (little grass, or a
                        player filling the frame).
    2. Ball detection - YOLO; picks the candidate most consistent with the last position.
    3. Pitch mapping  - PnLCalib (fixed landmarks + visible lines) calibrates the camera about
                        once a second; camera motion is tracked in between. The ball's
                        ground point is mapped to metres on a 105 x 68 pitch (homography).
    3b. In the air    - a lofted ball's ground mapping is wrong, so while it's airborne
                        (gravity-shaped path on screen) no position is reported; tracking
                        restarts from where it lands.
    4. Smoothing      - constant-velocity Kalman filter in pitch coordinates; coasts through
                        short occlusions and gives velocity.
    5. Output         - CSV per frame, optional annotated video with a radar minimap,
                        optional UDP JSON stream for the haptic device. Positions are
                        relative to the centre spot (0, 0).

Setup: pip install -r requirements.txt && ./setup_pnlcalib.sh

Examples:
    python track_ball.py videos/OFbyNU6UQQs.mp4 --start 600 --duration 60 --out-video annotated.mp4
    python track_ball.py "https://www.youtube.com/watch?v=OFbyNU6UQQs"     # downloads first
    python track_ball.py screen --region 0,100,1280,720 --udp 192.168.1.50:5005   # live
"""
import argparse
import csv
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pitch model (metres). x runs goal line to goal line, y touchline to touchline.
# The 32 vertices are in the same order as the keypoints of the roboflow/sports
# pitch keypoint model, so keypoint i maps to PITCH_VERTICES[i].
# ---------------------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
_PB_W, _PB_L = 40.32, 16.5    # penalty box
_GB_W, _GB_L = 18.32, 5.5     # goal box
_CIRCLE_R = 9.15
_SPOT = 11.0
_L, _W = PITCH_LENGTH, PITCH_WIDTH

PITCH_VERTICES = np.array([
    (0, 0),
    (0, (_W - _PB_W) / 2),
    (0, (_W - _GB_W) / 2),
    (0, (_W + _GB_W) / 2),
    (0, (_W + _PB_W) / 2),
    (0, _W),
    (_GB_L, (_W - _GB_W) / 2),
    (_GB_L, (_W + _GB_W) / 2),
    (_SPOT, _W / 2),
    (_PB_L, (_W - _PB_W) / 2),
    (_PB_L, (_W - _GB_W) / 2),
    (_PB_L, (_W + _GB_W) / 2),
    (_PB_L, (_W + _PB_W) / 2),
    (_L / 2, 0),
    (_L / 2, _W / 2 - _CIRCLE_R),
    (_L / 2, _W / 2 + _CIRCLE_R),
    (_L / 2, _W),
    (_L - _PB_L, (_W - _PB_W) / 2),
    (_L - _PB_L, (_W - _GB_W) / 2),
    (_L - _PB_L, (_W + _GB_W) / 2),
    (_L - _PB_L, (_W + _PB_W) / 2),
    (_L - _SPOT, _W / 2),
    (_L - _GB_L, (_W - _GB_W) / 2),
    (_L - _GB_L, (_W + _GB_W) / 2),
    (_L, 0),
    (_L, (_W - _PB_W) / 2),
    (_L, (_W - _GB_W) / 2),
    (_L, (_W + _GB_W) / 2),
    (_L, (_W + _PB_W) / 2),
    (_L, _W),
    (_L / 2 - _CIRCLE_R, _W / 2),
    (_L / 2 + _CIRCLE_R, _W / 2),
], dtype=np.float32)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Scene filter
# ---------------------------------------------------------------------------
def grass_ratio(frame: np.ndarray) -> float:
    """Fraction of green pixels in the lower 2/3 of the frame (wide gameplay shots are mostly grass)."""
    h = frame.shape[0]
    small = cv2.resize(frame[h // 3:], (160, 60))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    return float(np.count_nonzero(mask)) / mask.size


def promo_frame(frame: np.ndarray) -> bool:
    """True when the match is shown shrunk inside the FIFA archive promo frame (flat blue panel
    down the left edge). Mapping and detection are unreliable there, so it's treated like a replay."""
    hsv = cv2.cvtColor(frame[:, :100], cv2.COLOR_BGR2HSV)
    blue = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 30)
    return float(blue.mean()) > 0.5


# ---------------------------------------------------------------------------
# 2. Ball detection
# ---------------------------------------------------------------------------
class BallDetector:
    def __init__(self, weights: str, device: str, imgsz: int, conf: float, precision: int = 32):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device, self.imgsz, self.conf, self.precision = device, imgsz, conf, precision
        names = self.model.names
        # COCO calls it "sports ball"; soccer-specific models usually call it "ball".
        self.class_ids = [i for i, n in names.items() if "ball" in n.lower()]
        if not self.class_ids:
            raise ValueError(f"No ball class in {weights}: {names}")
        # COCO models also detect people in the same pass, which gives a free close-up check.
        self.person_ids = [i for i, n in names.items() if n.lower() == "person"]

    def detect(self, frame):
        """Return (ball candidates as (cx, cy, bottom_y, conf, w, h), people boxes as (x1, y1, x2, y2)).

        People are empty when the model has no person class.
        """
        from field_calib import GPU_LOCK

        with GPU_LOCK:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                   classes=self.class_ids + self.person_ids,
                                   device=self.device, quantize=self.precision, verbose=False)[0]
        balls, people = [], []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy()):
            if int(k) in self.class_ids:
                balls.append((float(x1 + x2) / 2, float(y1 + y2) / 2, float(y2), float(c),
                              float(x2 - x1), float(y2 - y1)))
            elif c >= 0.4:
                people.append((float(x1), float(y1), float(x2), float(y2)))
        return balls, people


BALL_DIAMETER_M = 0.22
MAX_ASPECT = 1.6          # a ball's box is roughly square (motion blur stretches it a little)
SIZE_RANGE = (0.5, 4.0)   # box size vs the expected size of a ball on the grass at that spot;
                          # generous above: a lofted ball is closer to the camera, and
                          # detector boxes run ~1.5x the ball itself


MAX_BALL_SATURATION = 85  # the ball's bright pixels are white; neon boots are strongly coloured


def bright_saturation(frame, c):
    """Median saturation (0-255) of the brightest quarter of pixels in a candidate's box."""
    cx, cy, w, h = c[0], c[1], c[4], c[5]
    x1, y1 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
    crop = frame[y1:int(cy + h / 2) + 1, x1:int(cx + w / 2) + 1]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    top = hsv[hsv[:, 2] >= np.percentile(hsv[:, 2], 75)]
    return float(np.median(top[:, 1]))


def plausible_balls(cands, H, frame=None, allow_off_pitch=False, off_pitch_margin=0.5):
    """Drop candidates that can't be the ball: elongated boxes (boots), strongly coloured
    ones (neon boots; the ball is white), and, where the pitch mapping is known, boxes far
    too small or large for a 22 cm ball at that spot, or lying outside the pitch lines
    (spare balls behind the goal, ad boards). allow_off_pitch while the ball is in the air,
    where its ground projection can fall outside the pitch."""
    out = []
    Hinv = np.linalg.inv(H) if H is not None else None
    for c in cands:
        w, h = c[4], c[5]
        if max(w, h) > MAX_ASPECT * max(min(w, h), 1.0):
            continue
        if frame is not None and bright_saturation(frame, c) > MAX_BALL_SATURATION:
            continue
        if Hinv is not None:
            gx, gy = to_pitch(H, c[0], c[2])
            if not allow_off_pitch and not on_pitch(gx, gy, margin=off_pitch_margin):
                continue
            if on_pitch(gx, gy):
                r = BALL_DIAMETER_M / 2
                pts = cv2.perspectiveTransform(np.float32([[[gx - r, gy], [gx + r, gy], [gx, gy - r], [gx, gy + r]]]), Hinv)[0]
                expected = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[2]))
                if not SIZE_RANGE[0] * expected <= max(w, h) <= SIZE_RANGE[1] * expected:
                    continue
        out.append(c)
    return out


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


# ---------------------------------------------------------------------------
# 3. Pitch mapping (image -> metres)
# ---------------------------------------------------------------------------
def _pitch_line_samples(step=0.75):
    """Points along every painted pitch line (metres) with their unit tangent and a line id."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    segs = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)),
            ((L / 2, 0), (L / 2, W))]
    for x0, sgn in ((0, 1), (L, -1)):
        for bw, bl in ((_PB_W, _PB_L), (_GB_W, _GB_L)):
            y0, y1, x1 = (W - bw) / 2, (W + bw) / 2, x0 + sgn * bl
            segs += [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x1, y0), (x1, y1))]
    pts, tans, ids = [], [], []
    for i, (a, b) in enumerate(segs):
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step))
        t = (b - a) / np.linalg.norm(b - a)
        for s in np.linspace(0, 1, n):
            pts.append(a + s * (b - a)); tans.append(t); ids.append(i)
    n = int(2 * np.pi * _CIRCLE_R / step)
    for k, ang in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        pts.append((L / 2 + _CIRCLE_R * np.cos(ang), W / 2 + _CIRCLE_R * np.sin(ang)))
        tans.append((-np.sin(ang), np.cos(ang)))
        ids.append(len(segs) + k * 8 // n)  # split the circle into 8 arcs for the spread check
    return np.array(pts, np.float32), np.array(tans, np.float32), np.array(ids)


LINE_PTS, LINE_TANS, LINE_IDS = _pitch_line_samples()


def line_mask(frame):
    """Painted pitch lines: thin, bright, low-saturation structures surrounded by grass.

    White kits pass the colour test too, so anything thicker than a line (a shirt, shorts)
    is removed, along with everything off the grass (crowd, ad boards).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    bright = ((tophat > 15) & (hsv[:, :, 1] < 130)).astype(np.uint8)
    thick = cv2.erode(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return (bright > 0) & (thick == 0) & (grass > 0)


def line_score(H, dist, min_visible=40, tol_px=3.0):
    """How well a homography's projected pitch lines sit on the detected lines.

    Fraction of in-frame projected line samples within tol_px of a detected line pixel;
    higher is better (lines hidden behind players just lower it a little). None if too
    little of the pitch lands in frame.
    """
    h, w = dist.shape
    try:
        img = cv2.perspectiveTransform(LINE_PTS[None], np.linalg.inv(H))[0]
    except np.linalg.LinAlgError:
        return None
    ok = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
    if ok.sum() < min_visible:
        return None
    d = dist[img[ok, 1].astype(int), img[ok, 0].astype(int)]
    return float((d <= tol_px).mean())


def line_coverage(H, mask, tol_px=3):
    """Share of detected line pixels that lie on some projected pitch line (the reverse of
    line_score): a wrong fit leaves most of the real lines unexplained."""
    if not mask.any():
        return 0.0
    try:
        img = cv2.perspectiveTransform(LINE_PTS[None], np.linalg.inv(H))[0]
    except np.linalg.LinAlgError:
        return 0.0
    canvas = np.zeros(mask.shape, np.uint8)
    for i in np.unique(LINE_IDS):
        pts = img[LINE_IDS == i]
        if np.abs(pts).max() < 1e5:
            cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 255, 2 * tol_px + 1)
    return float((canvas[mask] > 0).mean())


class PitchMapper:
    """Fallback pitch mapping from the roboflow/sports landmark model alone (--pitch keypoints).

    Less accurate than PnLCalib: the landmark positions are often 20-40 px off, i.e.
    several metres. Fits are blended over time to reduce jitter.
    """

    MIN_INLIERS = 5
    MAX_DISAGREE_M = 4.0  # a fit this far from the previous one is taken as a new view, not blended

    def __init__(self, weights: str, device: str, kp_conf: float, hold_frames: int, precision: int = 32):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device, self.kp_conf, self.hold_frames = device, kp_conf, hold_frames
        self.precision = precision
        self.H = None
        self.age = 0  # frames since H was last refreshed
        self.grid = None

    def reset(self):
        self.H, self.age = None, 0

    def update(self, frame):
        H = self._fit(frame)
        if H is not None and self.H is not None:
            if self.grid is None:
                h, w = frame.shape[:2]
                gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 4) * w, np.linspace(0.35, 0.95, 3) * h)
                self.grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)[None]
            disagree = np.median(np.linalg.norm(cv2.perspectiveTransform(self.grid, H)[0]
                                                - cv2.perspectiveTransform(self.grid, self.H)[0], axis=1))
            if disagree <= self.MAX_DISAGREE_M:
                H = 0.5 * H / H[2, 2] + 0.5 * self.H / self.H[2, 2]
        if H is not None:
            self.H, self.age = H, 0
        else:
            self.age += 1
            if self.age > self.hold_frames:  # camera has moved too much to trust the old one
                self.H = None
        return self.H

    def hold(self):
        """Reuse the previous homography without refitting (for --pitch-every > 1)."""
        self.age += 1
        if self.age > self.hold_frames:
            self.H = None
        return self.H

    def _fit(self, frame):
        r = self.model.predict(frame, conf=0.3, device=self.device, quantize=self.precision,
                               verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        xy = r.keypoints.xy[0].cpu().numpy()
        conf = (r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None
                else np.ones(len(xy)))
        n = min(len(xy), len(PITCH_VERTICES))
        mask = (conf[:n] > self.kp_conf) & (xy[:n, 0] > 1) & (xy[:n, 1] > 1)
        if mask.sum() < self.MIN_INLIERS:
            return None
        H, inliers = cv2.findHomography(xy[:n][mask], PITCH_VERTICES[:n][mask], cv2.RANSAC, 1.5)
        if H is None or inliers is None or inliers.sum() < self.MIN_INLIERS:
            return None
        return H


def to_pitch(H, px, py):
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0, 0]
    return float(pt[0]), float(pt[1])


def on_pitch(x, y, margin=3.0):
    return -margin <= x <= PITCH_LENGTH + margin and -margin <= y <= PITCH_WIDTH + margin


# ---------------------------------------------------------------------------
# 3b. Ball in the air
# ---------------------------------------------------------------------------
PERSON_HEIGHT_M = 1.8


def vertical_scale(people, x, y, k=3):
    """Pixels per metre of height near (x, y), from the heights of the k nearest people."""
    if not people:
        return None
    d = [np.hypot((a + c) / 2 - x, b2 - y) for a, _, c, b2 in people]
    near = [people[i] for i in np.argsort(d)[:k]]
    return float(np.median([b2 - b1 for _, b1, _, b2 in near])) / PERSON_HEIGHT_M


def at_feet(people, x, y):
    """True if (x, y) is at some player's feet, i.e. the ball is on the ground being played."""
    for x1, y1, x2, y2 in people:
        h, w = y2 - y1, x2 - x1
        if x1 - 0.3 * w <= x <= x2 + 0.3 * w and y2 - 0.2 * h <= y <= y2 + 0.1 * h:
            return True
    return False


class AirDetector:
    """Decides whether the ball is in the air.

    Ground geometry only holds for a ball on the grass; a lofted ball maps metres away
    from where it really is. In camera-stabilised image coordinates an airborne ball
    accelerates downward at ~9.8 m/s^2 (pixels converted with nearby players' heights),
    which ground balls and detector noise don't: the test needs a clean, rising-then-
    curving path from an unbroken run of detections. It counts as landed at the bounce
    (stops falling), when its motion is flat again, when it reaches a player's feet, or
    after max_air_s.
    """

    def __init__(self, window_s=0.4, min_samples=8, air_accel=(6.0, 25.0), land_accel=3.0,
                 max_rms_m=0.15, max_gap_s=0.12, max_air_s=3.0, min_rise=2.0):
        self.window_s, self.min_samples = window_s, min_samples
        self.air_accel, self.land_accel, self.max_rms_m = air_accel, land_accel, max_rms_m
        self.max_gap_s, self.max_air_s, self.min_rise = max_gap_s, max_air_s, min_rise
        self.reset()

    def reset(self):
        self.hist = []          # (t, y_px_stabilised, px_per_m) of an unbroken run of detections
        self.airborne = False
        self.since = None       # take-off time
        self.falling = False
        self.accel = None

    def break_track(self):
        """The ball jumped (re-acquired elsewhere): earlier samples aren't the same trajectory."""
        self.hist = []

    def update(self, t, y_px, px_per_m, on_ground=False):
        """Feed one detection; returns True while the ball is in the air.
        on_ground: independent evidence it's on the grass (at a player's feet)."""
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
            if rms <= self.max_rms_m:   # only trust a clean curve
                self.accel = 2 * coef[0] / ppm   # m/s^2, positive = accelerating downward on screen
                vy = coef[1] / ppm               # m/s now, positive = moving down the screen
                v_first = (coef[1] + 2 * coef[0] * ts[0]) / ppm  # at the oldest sample

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
                self.hist = self.hist[-2:]  # the next flight is judged on fresh samples
        return self.airborne


# ---------------------------------------------------------------------------
# 4. Smoothing
# ---------------------------------------------------------------------------
class BallKalman:
    """Constant-velocity Kalman filter over pitch coordinates. State = [x, y, vx, vy] (m, m/s)."""

    def __init__(self, accel_std=25.0, meas_std=0.15, maneuver=False, kick_nis=13.8, kick_vel_std=10.0):
        """meas_std: error of one mapped ground position (measured ~0.1 m with PnLCalib).
        maneuver: kick detection. A measurement far outside what the prediction expects
        (squared Mahalanobis distance > kick_nis, 99.9% for 2 dof) means the ball was struck;
        the velocity uncertainty is then widened by kick_vel_std so the estimate follows the
        new speed at once instead of ramping up over several frames."""
        self.accel_std = accel_std
        self.maneuver, self.kick_nis, self.kick_vel_std = maneuver, kick_nis, kick_vel_std
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.eye(2) * meas_std ** 2
        self.x = None
        self.P = None
        self.age = 0.0

    def reset(self):
        self.x = self.P = None
        self.age = 0.0  # seconds followed since the last restart; velocity needs a little history

    def predict(self, dt):
        """Advance the state by dt seconds (frames can be unevenly spaced when running live)."""
        if self.x is None:
            return None
        self.age += dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.accel_std ** 2
        return self.x

    def gate_distance2(self, z):
        """Squared Mahalanobis distance of a measurement from the prediction (chi-square, 2 dof)."""
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


# ---------------------------------------------------------------------------
# 5. Output helpers
# ---------------------------------------------------------------------------
RADAR_SCALE = 3  # px per metre


def draw_radar(frame, ball_xy, state):
    """Draw a top-down pitch with the ball in the bottom-right corner of the frame."""
    s = RADAR_SCALE
    w, h = int(PITCH_LENGTH * s), int(PITCH_WIDTH * s)
    pad = 10
    radar = np.full((h + 2 * pad, w + 2 * pad, 3), (40, 110, 40), np.uint8)

    def P(x, y):
        return int(pad + x * s), int(pad + y * s)

    white = (255, 255, 255)
    cv2.rectangle(radar, P(0, 0), P(_L, _W), white, 1)
    cv2.line(radar, P(_L / 2, 0), P(_L / 2, _W), white, 1)
    cv2.circle(radar, P(_L / 2, _W / 2), int(_CIRCLE_R * s), white, 1)
    for x0, x1 in ((0, _PB_L), (_L, _L - _PB_L)):
        cv2.rectangle(radar, P(x0, (_W - _PB_W) / 2), P(x1, (_W + _PB_W) / 2), white, 1)
    for x0, x1 in ((0, _GB_L), (_L, _L - _GB_L)):
        cv2.rectangle(radar, P(x0, (_W - _GB_W) / 2), P(x1, (_W + _GB_W) / 2), white, 1)
    if ball_xy is not None:
        color = (0, 255, 255) if state == "live" else (0, 165, 255)
        cv2.circle(radar, P(*ball_xy), 5, color, -1)
        cv2.circle(radar, P(*ball_xy), 5, (0, 0, 0), 1)

    fh, fw = frame.shape[:2]
    rh, rw = radar.shape[:2]
    if rw < fw and rh < fh:
        roi = frame[fh - rh - 10:fh - 10, fw - rw - 10:fw - 10]
        cv2.addWeighted(radar, 0.85, roi, 0.15, 0, dst=roi)


def annotate(frame, rec, px, ball_xy):
    if px is not None:
        # yellow = position sent to the device; grey = followed but not sent (uncertain, air)
        sent = rec["field_x_m"] != ""
        cv2.circle(frame, (int(px[0]), int(px[1])), 14, (0, 255, 255) if sent else (150, 150, 150), 2)
    label = f"{rec['state']}"
    if rec["field_x_m"] != "":
        label += f"  x={rec['field_x_m']:.1f}m y={rec['field_y_m']:.1f}m"
    cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    draw_radar(frame, ball_xy, rec["state"])


CSV_FIELDS = ["frame", "time_s", "state", "confidence",
              "ball_px_x", "ball_px_y", "screen_x", "screen_y",
              # field position relative to the centre spot (0, 0), metres: +x toward the goal on
              # the right of the main camera view, +y toward the camera-side touchline.
              # *_norm: the same scaled to -1..1 (goal line / touchline = +-1).
              "field_x_m", "field_y_m", "field_vx_ms", "field_vy_ms",
              "field_x_norm", "field_y_norm", "has_homography",
              "air_accel_ms2",   # the in-the-air test's measured acceleration, for tuning
              "track_conf",      # recent confidence of the tracked ball (reporting threshold)
              "n_detected", "n_plausible",  # ball candidates from the detector / after the filters
              "raw_x_m", "raw_y_m", "kf_restart",  # unsmoothed ground position (centre origin); 1 = filter restarted
              # motor signal (motor_speed.py): one clean speed profile per pass, for the haptics.
              # It lags by --motor-delay: these values describe the ball at time motor_t.
              "motor_t", "motor_speed_ms", "motor_vx_ms", "motor_vy_ms", "motor_touch",
              # live pass profile (live_passes.py): velocity the motors should have NOW, and the
              # pass being followed (start time, start velocity, slow-down); new_pass=1 when announced
              "live_vx_ms", "live_vy_ms", "live_speed_ms",
              "pass_t0", "pass_vx0_ms", "pass_vy0_ms", "pass_decel_ms2", "new_pass"]


# ---------------------------------------------------------------------------
# Frame sources. Each yields (frame_number, time_s, frame).
# ---------------------------------------------------------------------------
class FileSource:
    """A video file. With realtime=True it behaves like a live feed: frames are skipped
    whenever processing falls behind the video clock, exactly as a live stream would."""

    def __init__(self, path, start=0.0, duration=None, stride=1, realtime=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.start_idx = int(start * self.fps)
        self.end_idx = self.start_idx + int(duration * self.fps) if duration else None
        self.stride, self.realtime = stride, realtime
        self.dropped = 0

    def __iter__(self):
        idx = self.start_idx
        if idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        t0 = time.time()
        next_idx = idx
        while self.end_idx is None or next_idx < self.end_idx:
            if self.realtime:
                # the frame "on air" now; never process ahead of it, skip anything behind it
                live_idx = self.start_idx + int((time.time() - t0) * self.fps)
                if next_idx > live_idx:
                    time.sleep((next_idx - live_idx) / self.fps)
                else:
                    next_idx = max(next_idx, live_idx)
            while idx < next_idx:  # skip without the cost of converting the frame
                if not self.cap.grab():
                    return
                idx += 1
                if self.realtime and idx % self.stride == 0:
                    self.dropped += 1
            ok, frame = self.cap.read()
            if not ok:
                return
            yield idx, idx / self.fps, frame
            idx += 1
            next_idx = (idx + self.stride - 1) // self.stride * self.stride

    def release(self):
        self.cap.release()


class LiveSource:
    """Screen region or camera (e.g. OBS Virtual Camera), captured on a background thread
    so capture never waits on inference; the tracker always gets the newest frame."""

    def __init__(self, spec, region=None, width=1280):
        import threading

        self.spec, self.region, self.width = spec, region, width
        self.latest = None
        self.cond = threading.Condition()
        self.running = True
        self.fps = 30.0
        self.total = 0
        self.dropped = 0
        self.captured = 0
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        with self.cond:  # wait for the first frame so the size is known
            self.cond.wait_for(lambda: self.latest is not None or self.error, timeout=10)
        if self.error or self.latest is None:
            raise SystemExit(f"Could not capture from {spec}: {self.error or 'no frames within 10 s'}")
        h, w = self.latest[2].shape[:2]
        self.size = (w, h)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self):
        try:
            if self.spec == "screen":
                import mss

                with mss.mss() as sct:
                    mon = sct.monitors[1]
                    if self.region:
                        x, y, w, h = self.region
                        mon = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
                    while self.running:
                        frame = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
                        self._publish(self._resize(np.ascontiguousarray(frame)))
            else:
                cap = cv2.VideoCapture(int(self.spec))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("camera returned no frame")
                    self._publish(self._resize(frame))
                cap.release()
        except Exception as e:  # surfaced to the main thread
            self.error = e
            with self.cond:
                self.cond.notify_all()

    def _publish(self, frame):
        with self.cond:
            if self.latest is not None and not self.latest[3]:
                self.dropped += 1  # previous frame was never processed
            self.captured += 1
            self.latest = [self.captured, time.time(), frame, False]
            self.cond.notify_all()

    def __iter__(self):
        t0 = None
        while True:
            with self.cond:
                self.cond.wait_for(lambda: (self.latest is not None and not self.latest[3]) or self.error)
                if self.error:
                    raise SystemExit(f"Capture failed: {self.error}")
                self.latest[3] = True
                n, t, frame, _ = self.latest
            t0 = t if t0 is None else t0
            yield n, t - t0, frame

    def release(self):
        self.running = False


def open_source(args):
    src = args.source
    if src == "screen" or src.isdigit():
        region = tuple(int(v) for v in args.region.split(",")) if args.region else None
        return LiveSource(src, region), src
    if src.startswith("http"):
        from download import download
        src = str(download(src))
    return FileSource(src, args.start, args.duration, args.stride, args.realtime), src


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args):
    source, name = open_source(args)
    width, height = source.size
    live = isinstance(source, LiveSource)

    device = pick_device(args.device)
    precision = 32 if (args.fp32 or device == "cpu") else 16
    print(f"{name}: {width}x{height} @ {source.fps:.1f} fps, device={device}, fp{precision}")

    detector = BallDetector(args.ball_weights, device, args.imgsz, args.conf, precision)
    pitch = tracker = None
    if args.pitch == "pnl":
        from field_calib import FieldCalibrator, PitchTracker
        async_ = live or args.realtime
        # live: calibrate on the CPU so the ball detector never waits for the GPU
        calib_device = args.calib_device or ("cpu" if async_ else device)
        # camera tracking holds ~6 s from one calibration; CPU calibrations arrive ~3 s late
        tracker = PitchTracker(FieldCalibrator(device=calib_device), interval=args.calib_interval,
                               async_=async_, max_stale=6.0 if calib_device == "cpu" else 4.0)
    elif args.pitch == "keypoints":
        pitch = PitchMapper(args.pitch_weights, device, args.kp_conf, args.hold_frames, precision)
    kf = BallKalman()
    air = AirDetector()
    from motor_speed import MotorSpeed
    motor = MotorSpeed(delay=args.motor_delay)
    from live_passes import LivePasses
    live_passes = LivePasses(mode=args.pass_mode)

    default_csv = "live.ball.csv" if live else Path(name).with_suffix(".ball.csv")
    out_csv = Path(args.out_csv or default_csv)
    csv_file = open(out_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    video_out = None
    if args.out_video:
        video_out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                    source.fps / (1 if live else args.stride), (width, height))

    udp = None
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    selector = BallSelector(width, max_jump=args.max_jump, confirm_n=args.confirm)

    last_px, missed_s, t_prev = None, 0.0, None
    out_xy = None  # last reported position (pitch metres), for the speed cap
    last_ground_t = None  # when the filter last got a mapped ground position
    calm = args.resume_frames  # consecutive processed frames that looked like gameplay
    processed = 0
    t_start = time.time()
    t_report = t_start
    busy = 0.0  # seconds spent processing (vs waiting for frames)

    try:
        for idx, t, frame in source:
            t_work = time.time()
            dt = (t - t_prev) if t_prev is not None else 1 / source.fps
            t_prev = t

            rec = {k: "" for k in CSV_FIELDS}
            rec.update(frame=idx, time_s=round(t, 3), has_homography=0)
            px = None

            gameplay = grass_ratio(frame) >= args.min_grass and not promo_frame(frame)
            balls, people = detector.detect(frame) if gameplay else ([], [])
            tallest = max(((b[3] - b[1]) / height for b in people), default=0.0)
            # stay paused until the wide shot has held for a few frames (avoids flicker at cuts)
            calm = 0 if (not gameplay or tallest > args.max_person) else calm + 1
            if calm < args.resume_frames:
                # crowd shot / close-up: position is unknown, don't let the filters drift
                rec["state"] = "paused"
                kf.reset()
                if pitch is not None:
                    pitch.reset()
                if tracker is not None:
                    tracker.reset()
                air.reset()
                selector.forget()
                last_px, missed_s = None, 0.0
            else:
                H = None
                if tracker is not None:
                    H = tracker.update(frame, idx, t)
                elif pitch is not None:
                    H = pitch.update(frame) if processed % args.pitch_every == 0 else pitch.hold()
                rec["has_homography"] = int(H is not None)

                kf.predict(dt)
                rec["n_detected"] = len(balls)
                balls = plausible_balls(balls, H, frame, allow_off_pitch=air.airborne,
                                        off_pitch_margin=args.off_pitch_margin)
                rec["n_plausible"] = len(balls)
                cand, switched = selector.choose(balls, t, kf, H)

                if cand is not None:
                    cx, cy, bottom, conf = cand[:4]
                    prev_px, missed_s_before = last_px, missed_s
                    px, last_px, missed_s = (cx, cy), (cx, cy), 0.0
                    rec.update(state="live", confidence=round(conf, 3),
                               ball_px_x=round(cx, 1), ball_px_y=round(cy, 1),
                               screen_x=round(cx / width, 4), screen_y=round(cy / height, 4))
                    ground = None
                    if H is not None:
                        # bottom of the box = where the ball touches the grass (for ground balls)
                        fx, fy = to_pitch(H, cx, bottom)
                        ground = (fx, fy) if on_pitch(fx, fy) else None
                    # height on screen with camera motion removed, for the gravity test
                    y_stab = cy
                    if tracker is not None:
                        y_stab = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), np.linalg.inv(tracker.cum))[0, 0, 1]
                    if switched or prev_px is None or missed_s_before > 0 or \
                            np.hypot(cx - prev_px[0], cy - prev_px[1]) > 0.08 * width:
                        air.break_track()  # not a continuous trajectory
                    if switched:
                        # the frames it spent being confirmed are part of its trajectory
                        # (camera motion over those few frames is ignored)
                        for t_seen, c_seen in selector.confirmed_track:
                            y_seen = c_seen[1]
                            if tracker is not None:
                                y_seen = cv2.perspectiveTransform(np.float32([[c_seen[:2]]]),
                                                                  np.linalg.inv(tracker.cum))[0, 0, 1]
                            air.update(t_seen, float(y_seen), vertical_scale(people, c_seen[0], c_seen[1]),
                                       on_ground=at_feet(people, c_seen[0], c_seen[1]))
                    was_airborne = air.airborne
                    airborne = air.update(t, float(y_stab), vertical_scale(people, cx, cy),
                                          on_ground=at_feet(people, cx, cy))
                    rec["air_accel_ms2"] = "" if air.accel is None else round(air.accel, 2)
                    if airborne:
                        # in the air: its ground projection is wrong, so report no position
                        rec["state"] = "air"
                        kf.reset()
                    elif ground is not None:
                        if was_airborne or switched:
                            kf.reset()  # just landed, or confirmed re-acquisition elsewhere: start here
                        rec.update(raw_x_m=round(ground[0] - PITCH_LENGTH / 2, 3),
                                   raw_y_m=round(ground[1] - PITCH_WIDTH / 2, 3), kf_restart=int(kf.x is None))
                        kf.update(ground)
                        last_ground_t = t
                elif air.airborne and t - air.since <= air.max_air_s:
                    missed_s += dt
                    rec["state"] = "air"  # often lost against the crowd mid-flight
                else:
                    missed_s += dt
                    if missed_s > args.max_coast:
                        # the selector keeps the last sighting: the search widens from there
                        rec["state"] = "lost"
                        kf.reset()
                        air.reset()
                        last_px = None
                    else:
                        rec["state"] = "coasting"

                # weak tracks (recent pick confidence low: often a boot) are still followed, but
                # not reported, so the device doesn't act on them
                if rec["state"] in ("live", "coasting"):
                    rec["track_conf"] = round(selector.conf_ema, 3)
                if rec["state"] in ("live", "coasting") and selector.conf_ema < args.min_report_conf:
                    rec["state"] = "uncertain"
                # without a fresh mapped sighting the filter's position is stale: don't send it
                if rec["state"] in ("live", "coasting") and (last_ground_t is None or t - last_ground_t > args.max_coast):
                    rec["state"] = "unmapped" if H is None else rec["state"]
                if kf.x is not None and rec["state"] in ("live", "coasting") and \
                        last_ground_t is not None and t - last_ground_t <= args.max_coast:
                    x, y, vx, vy = kf.x
                    x, y = float(np.clip(x, 0, PITCH_LENGTH)), float(np.clip(y, 0, PITCH_WIDTH))
                    if out_xy is not None:
                        # never faster than a real ball: a re-acquired position is glided to
                        step, cap = np.hypot(x - out_xy[0], y - out_xy[1]), args.max_speed * dt
                        if step > cap:
                            x, y = out_xy[0] + (x - out_xy[0]) * cap / step, out_xy[1] + (y - out_xy[1]) * cap / step
                    out_xy = (x, y)
                    # reported relative to the centre spot: x in [-52.5, 52.5], y in [-34, 34]
                    x, y = x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2
                    # velocity only once the filter has followed the ball for a moment since
                    # restarting; before that it's a guess from a few noisy points
                    settled = kf.age >= args.min_vel_age
                    rec.update(field_x_m=round(x, 2), field_y_m=round(y, 2),
                               field_vx_ms=round(float(vx), 2) if settled else "",
                               field_vy_ms=round(float(vy), 2) if settled else "",
                               field_x_norm=round(x / (PITCH_LENGTH / 2), 4),
                               field_y_norm=round(y / (PITCH_WIDTH / 2), 4))

            if rec["field_x_m"] == "":
                out_xy = None  # no position this frame: the next one starts fresh
            raw_xy = (rec["raw_x_m"], rec["raw_y_m"]) if rec["raw_x_m"] != "" else None
            ev = live_passes.push(t, raw_xy, rec["kf_restart"] == 1)
            lv = live_passes.velocity(t)
            if rec["state"] in ("paused", "lost", "unmapped"):
                lv = None  # nothing reliable to follow right now
            if lv is not None:
                p_ = live_passes.pass_
                rec.update(live_vx_ms=round(lv[0], 2), live_vy_ms=round(lv[1], 2),
                           live_speed_ms=round(float(np.hypot(*lv)), 2), pass_t0=round(p_["t0"], 3),
                           pass_vx0_ms=round(float(p_["v0"][0]), 2), pass_vy0_ms=round(float(p_["v0"][1]), 2),
                           pass_decel_ms2=live_passes.decel if args.pass_mode == "profile" else 0.0)
            rec["new_pass"] = int(ev is not None)
            for m in motor.push(t, raw_xy, rec["kf_restart"] == 1):
                rec.update(motor_t=round(m["t"], 3),
                           motor_speed_ms="" if m["speed"] is None else round(m["speed"], 2),
                           motor_vx_ms="" if m["vx"] is None else round(m["vx"], 2),
                           motor_vy_ms="" if m["vy"] is None else round(m["vy"], 2),
                           motor_touch=int(m["touch"]))
            writer.writerow(rec)

            if udp is not None:
                def val(k):
                    return None if rec[k] == "" else rec[k]
                msg = {"t": rec["time_s"], "state": rec["state"],
                       "x": val("field_x_norm"), "y": val("field_y_norm"),
                       "vx": val("field_vx_ms"), "vy": val("field_vy_ms"),
                       "sx": val("screen_x"), "sy": val("screen_y"),
                       "confidence": val("confidence"),
                       # for the motors: speed profile per pass, describing the ball at motor_t
                       "motor_t": val("motor_t"), "motor_speed": val("motor_speed_ms"),
                       "motor_vx": val("motor_vx_ms"), "motor_vy": val("motor_vy_ms"),
                       "touch": val("motor_touch"),
                       # live pass profile: velocity for the motors now + the pass it comes from
                       "live_vx": val("live_vx_ms"), "live_vy": val("live_vy_ms"),
                       "live_speed": val("live_speed_ms"), "new_pass": val("new_pass"),
                       "pass_t0": val("pass_t0"), "pass_vx0": val("pass_vx0_ms"),
                       "pass_vy0": val("pass_vy0_ms"), "pass_decel": val("pass_decel_ms2")}
                udp[0].sendto(json.dumps(msg).encode(), udp[1])

            if video_out is not None or args.show:
                ball_xy = ((rec["field_x_m"] + PITCH_LENGTH / 2, rec["field_y_m"] + PITCH_WIDTH / 2)
                           if rec["field_x_m"] != "" else None)  # radar draws in 0..105 x 0..68
                annotate(frame, rec, px, ball_xy)
                if video_out is not None:
                    video_out.write(frame)
                if args.show:
                    cv2.imshow("ball tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            processed += 1
            now = time.time()
            busy += now - t_work
            if now - t_report > 2:
                rate = processed / (now - t_start)
                load = busy / (now - t_start)
                pct = f" {100 * idx / source.total:.0f}%" if source.total else ""
                print(f"frame {idx}{pct}  {rate:.1f} fps processed  busy {load:.0%}  "
                      f"dropped {source.dropped}  state={rec['state']}")
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        csv_file.close()
        if video_out is not None:
            video_out.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"Processed {processed} frames in {elapsed:.1f} s ({processed / max(elapsed, 1e-9):.1f} fps), "
          f"busy {busy / max(elapsed, 1e-9):.0%}, dropped {source.dropped}")
    print(f"Wrote {out_csv}" + (f" and {args.out_video}" if args.out_video else ""))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, YouTube URL, 'screen', or a camera index (e.g. 0 for OBS Virtual Camera)")
    p.add_argument("--ball-weights",
                   default="models/ball_person.pt" if Path("models/ball_person.pt").exists() else "yolo11m.pt",
                   help="YOLO weights with 'ball' and 'person' classes (default: the team's fine-tuned "
                        "models/ball_person.pt; falls back to COCO yolo11m)")
    p.add_argument("--pitch", choices=["pnl", "keypoints", "none"], default="pnl",
                   help="pitch mapping: pnl = PnLCalib landmarks + lines (accurate, default); "
                        "keypoints = roboflow/sports landmark model only (metres off); "
                        "none = screen position only")
    p.add_argument("--calib-interval", type=float, default=1.0,
                   help="seconds between PnLCalib calibrations; camera motion is tracked in between")
    p.add_argument("--calib-device", default=None,
                   help="device for PnLCalib (default: cpu when live/--realtime so the ball detector "
                        "keeps the GPU, otherwise the main device)")
    p.add_argument("--pitch-weights", default="models/football-pitch-detection.pt",
                   help="landmark model for --pitch keypoints")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    p.add_argument("--imgsz", type=int, default=1280, help="inference size; larger finds smaller balls")
    p.add_argument("--conf", type=float, default=0.15, help="min ball detection confidence")
    p.add_argument("--kp-conf", type=float, default=0.5, help="min pitch keypoint confidence")
    p.add_argument("--start", type=float, default=0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="seconds to process (default: to the end)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--pitch-every", type=int, default=3,
                   help="--pitch keypoints: run the landmark model every Nth processed frame")
    p.add_argument("--hold-frames", type=int, default=10, help="reuse last homography this many frames")
    p.add_argument("--max-coast", type=float, default=0.2, help="seconds to predict through before 'lost'")
    p.add_argument("--max-jump", type=float, default=0.6,
                   help="max ball speed on screen, in frame-widths per second")
    p.add_argument("--confirm", type=int, default=3,
                   help="frames a ball seen somewhere unexpected must persist before it's accepted")
    p.add_argument("--min-report-conf", type=float, default=0.20,
                   help="positions are only reported while the tracked ball's recent detection "
                        "confidence is at least this (state 'uncertain' otherwise)")
    p.add_argument("--off-pitch-margin", type=float, default=0.5,
                   help="metres; ball candidates mapped further outside the pitch lines are ignored "
                        "(spare balls, ad boards)")
    p.add_argument("--min-vel-age", type=float, default=0.2,
                   help="seconds of tracking needed after a restart before velocity is reported")
    p.add_argument("--pass-mode", choices=["profile", "constant"], default="profile",
                   help="live pass output: 'profile' = start speed slowing at a standard rate; "
                        "'constant' = one fixed speed per pass (start speed x typical ratio)")
    p.add_argument("--motor-delay", type=float, default=0.15,
                   help="seconds of look-ahead for the motor speed signal (it lags by this much)")
    p.add_argument("--max-speed", type=float, default=35.0,
                   help="m/s; the reported position never moves faster than this")
    p.add_argument("--max-person", type=float, default=0.3,
                   help="a person taller than this fraction of the frame means close-up (COCO models only)")
    p.add_argument("--resume-frames", type=int, default=8,
                   help="consecutive gameplay frames needed before tracking resumes after a pause")
    p.add_argument("--min-grass", type=float, default=0.35, help="min grass fraction for a gameplay shot")
    p.add_argument("--out-csv", default=None, help="default: <video>.ball.csv")
    p.add_argument("--out-video", default=None, help="write an annotated video with radar minimap")
    p.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    p.add_argument("--udp", default=None, help="stream JSON to HOST:PORT (e.g. an ESP32)")
    p.add_argument("--realtime", action="store_true",
                   help="treat a video file like a live feed: run at video speed, drop frames when behind")
    p.add_argument("--region", default=None,
                   help="for 'screen': left,top,width,height of the capture area in screen points")
    p.add_argument("--fp32", action="store_true", help="disable half precision on the GPU")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
