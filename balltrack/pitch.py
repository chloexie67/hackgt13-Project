"""Pitch geometry (metres, 105 x 68) and image <-> pitch helpers."""
import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_BOX_WIDTH, PENALTY_BOX_LENGTH = 40.32, 16.5
GOAL_BOX_WIDTH, GOAL_BOX_LENGTH = 18.32, 5.5
CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
_L, _W = PITCH_LENGTH, PITCH_WIDTH

PITCH_VERTICES = np.array([
    (0, 0),
    (0, (_W - PENALTY_BOX_WIDTH) / 2),
    (0, (_W - GOAL_BOX_WIDTH) / 2),
    (0, (_W + GOAL_BOX_WIDTH) / 2),
    (0, (_W + PENALTY_BOX_WIDTH) / 2),
    (0, _W),
    (GOAL_BOX_LENGTH, (_W - GOAL_BOX_WIDTH) / 2),
    (GOAL_BOX_LENGTH, (_W + GOAL_BOX_WIDTH) / 2),
    (PENALTY_SPOT_DISTANCE, _W / 2),
    (PENALTY_BOX_LENGTH, (_W - PENALTY_BOX_WIDTH) / 2),
    (PENALTY_BOX_LENGTH, (_W - GOAL_BOX_WIDTH) / 2),
    (PENALTY_BOX_LENGTH, (_W + GOAL_BOX_WIDTH) / 2),
    (PENALTY_BOX_LENGTH, (_W + PENALTY_BOX_WIDTH) / 2),
    (_L / 2, 0),
    (_L / 2, _W / 2 - CIRCLE_RADIUS),
    (_L / 2, _W / 2 + CIRCLE_RADIUS),
    (_L / 2, _W),
    (_L - PENALTY_BOX_LENGTH, (_W - PENALTY_BOX_WIDTH) / 2),
    (_L - PENALTY_BOX_LENGTH, (_W - GOAL_BOX_WIDTH) / 2),
    (_L - PENALTY_BOX_LENGTH, (_W + GOAL_BOX_WIDTH) / 2),
    (_L - PENALTY_BOX_LENGTH, (_W + PENALTY_BOX_WIDTH) / 2),
    (_L - PENALTY_SPOT_DISTANCE, _W / 2),
    (_L - GOAL_BOX_LENGTH, (_W - GOAL_BOX_WIDTH) / 2),
    (_L - GOAL_BOX_LENGTH, (_W + GOAL_BOX_WIDTH) / 2),
    (_L, 0),
    (_L, (_W - PENALTY_BOX_WIDTH) / 2),
    (_L, (_W - GOAL_BOX_WIDTH) / 2),
    (_L, (_W + GOAL_BOX_WIDTH) / 2),
    (_L, (_W + PENALTY_BOX_WIDTH) / 2),
    (_L, _W),
    (_L / 2 - CIRCLE_RADIUS, _W / 2),
    (_L / 2 + CIRCLE_RADIUS, _W / 2),
], dtype=np.float32)


def _pitch_line_samples(step=0.75):
    L, W = PITCH_LENGTH, PITCH_WIDTH
    segments = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)),
            ((L / 2, 0), (L / 2, W))]
    for x0, sgn in ((0, 1), (L, -1)):
        for bw, bl in ((PENALTY_BOX_WIDTH, PENALTY_BOX_LENGTH), (GOAL_BOX_WIDTH, GOAL_BOX_LENGTH)):
            y0, y1, x1 = (W - bw) / 2, (W + bw) / 2, x0 + sgn * bl
            segments += [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x1, y0), (x1, y1))]
    pts, ids = [], []
    for i, (a, b) in enumerate(segments):
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step))
        for s in np.linspace(0, 1, n):
            pts.append(a + s * (b - a))
            ids.append(i)
    n = int(2 * np.pi * CIRCLE_RADIUS / step)
    for k, ang in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        pts.append((L / 2 + CIRCLE_RADIUS * np.cos(ang), W / 2 + CIRCLE_RADIUS * np.sin(ang)))
        ids.append(len(segments) + k * 8 // n)  # the circle counts as 8 arcs
    return np.array(pts, np.float32), np.array(ids)


LINE_PTS, LINE_IDS = _pitch_line_samples()


def line_mask(frame):
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


def to_pitch(H, px, py):
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0, 0]
    return float(pt[0]), float(pt[1])


def on_pitch(x, y, margin=3.0):
    return -margin <= x <= PITCH_LENGTH + margin and -margin <= y <= PITCH_WIDTH + margin
