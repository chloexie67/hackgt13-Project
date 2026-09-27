"""Verification video: the tracker's annotated video plus a banner with exactly what demo.py
sends to the ESP32 at each moment, with the clip's original sound.

Usage: python -m tools.verify_video ANNOTATED.mp4 CLIP.mp4 CLIP.timeline.csv OUT.mp4 [--offset 34:57]
--offset is the clip's start time in the full match, so the banner shows the YouTube time.
"""
import argparse
import bisect
import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("annotated")
ap.add_argument("clip")
ap.add_argument("timeline")
ap.add_argument("out")
ap.add_argument("--offset", default="0:00", help="clip start in the full match, e.g. 34:57")
args = ap.parse_args()

offset_s = sum(float(v) * 60 ** i for i, v in enumerate(reversed(args.offset.split(":"))))
rows = list(csv.DictReader(open(args.timeline)))
times = [float(r["time_s"]) for r in rows]

cap = cv2.VideoCapture(args.annotated)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
silent = Path(tempfile.gettempdir()) / "hackgt_verify_silent.mp4"
writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = frame_no / fps
    row = rows[max(bisect.bisect_right(times, t + 1e-6) - 1, 0)]
    match_t = offset_s + t
    clock = f"{int(match_t // 60)}:{match_t % 60:04.1f}"
    if row["valid"] == "1":
        sent = (f"x {float(row['x_m']):6.1f} m   y {float(row['y_m']):6.1f} m   "
                f"vx {float(row['vx_ms']):5.1f}   vy {float(row['vy_ms']):5.1f} m/s")
        sent += f"   pass {row['pass_id']}" if row["pass_id"] != "" else "   (no pass)"
        color = (0, 255, 255)
    else:
        sent, color = "no data (valid = 0)", (160, 160, 160)
    cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"{clock}   sent to ESP32:  {sent}", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    writer.write(frame)
    frame_no += 1
writer.release()
cap.release()

subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", args.clip,
                "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-shortest", args.out], check=True)
print(f"wrote {args.out} ({frame_no} frames)")
