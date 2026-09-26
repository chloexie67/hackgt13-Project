"""Debug one frame: draw detected pitch keypoints and all ball candidates."""
import sys

import cv2
from ultralytics import YOLO
from balltrack.pitch import PITCH_VERTICES, to_pitch
video, t = sys.argv[1], float(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else "results/debug.png"
cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, frame = cap.read()
pitch = YOLO("models/football-pitch-detection.pt"); ball = YOLO("models/football-ball-detection.pt")
r = pitch.predict(frame, conf=0.3, device="mps", verbose=False)[0]
xy = r.keypoints.xy[0].cpu().numpy(); kc = r.keypoints.conf[0].cpu().numpy()
print("pitch boxes", len(r.boxes), "keypoint shape", xy.shape)
good = []
for i, ((x, y), c) in enumerate(zip(xy, kc)):
    if c > 0.5:
        good.append(i)
        cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 255), -1)
        cv2.putText(frame, f"{i+1}:{c:.2f}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
print("keypoints >0.5 (1-based):", [g + 1 for g in good])
b = ball.predict(frame, imgsz=1280, conf=0.05, device="mps", verbose=False)[0]
for (x1, y1, x2, y2), c in zip(b.boxes.xyxy.cpu().numpy(), b.boxes.conf.cpu().numpy()):
    print(f"ball cand ({(x1+x2)/2:.0f},{(y1+y2)/2:.0f}) conf {c:.2f} size {x2-x1:.0f}px")
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    cv2.putText(frame, f"{c:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
if len(good) >= 4:
    H, _ = cv2.findHomography(xy[good], PITCH_VERTICES[good], cv2.RANSAC, 1.5)
    for (x1, _, x2, y2) in b.boxes.xyxy.cpu().numpy():
        print("  -> pitch", to_pitch(H, (x1 + x2) / 2, y2))
cv2.imwrite(out, frame)
