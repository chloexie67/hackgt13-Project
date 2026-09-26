"""Diagnostic from the team: raw best.pt inference exactly as in Colab (no tracking, filtering,
calibration or half precision). Usage: python -m tools.raw_yolo_test VIDEO START_S DURATION_S OUT.mp4"""
import sys
import cv2
from ultralytics import YOLO

video, start, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
model = YOLO("models/best.pt", task="detect")
print("Classes:", model.names)
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for _ in range(int(dur * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    result = model.predict(source=frame, imgsz=1280, conf=0.25, verbose=False)[0]
    writer.write(result.plot())
cap.release()
writer.release()
print("Saved", out)
