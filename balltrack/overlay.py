"""Drawing the annotated output video."""
import cv2
import numpy as np

from .pitch import (CIRCLE_RADIUS, GOAL_BOX_LENGTH, GOAL_BOX_WIDTH, PENALTY_BOX_LENGTH,
                    PENALTY_BOX_WIDTH, PITCH_LENGTH, PITCH_WIDTH)

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
    cv2.rectangle(radar, P(0, 0), P(PITCH_LENGTH, PITCH_WIDTH), white, 1)
    cv2.line(radar, P(PITCH_LENGTH / 2, 0), P(PITCH_LENGTH / 2, PITCH_WIDTH), white, 1)
    cv2.circle(radar, P(PITCH_LENGTH / 2, PITCH_WIDTH / 2), int(CIRCLE_RADIUS * s), white, 1)
    for x0, x1 in ((0, PENALTY_BOX_LENGTH), (PITCH_LENGTH, PITCH_LENGTH - PENALTY_BOX_LENGTH)):
        cv2.rectangle(radar, P(x0, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2), P(x1, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2), white, 1)
    for x0, x1 in ((0, GOAL_BOX_LENGTH), (PITCH_LENGTH, PITCH_LENGTH - GOAL_BOX_LENGTH)):
        cv2.rectangle(radar, P(x0, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2), P(x1, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2), white, 1)
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
