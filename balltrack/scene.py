"""Is this frame a wide gameplay shot the tracker can use?"""
import cv2
import numpy as np

def grass_ratio(frame: np.ndarray) -> float:
    h = frame.shape[0]
    small = cv2.resize(frame[h // 3:], (160, 60))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    return float(np.count_nonzero(mask)) / mask.size


def promo_frame(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame[:, :100], cv2.COLOR_BGR2HSV)
    blue = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 30)
    return float(blue.mean()) > 0.5
