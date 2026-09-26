"""Ball and player detection, and checks that reject ball-like candidates."""
import cv2
import numpy as np

from .device import GPU_LOCK
from .pitch import on_pitch, to_pitch

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
