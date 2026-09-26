#!/bin/bash
# Estimate whole-game coverage: run the tracker on N evenly spaced stretches of a video.
# Usage: ./sample_coverage.sh VIDEO N SECONDS
video=$1; n=${2:-10}; secs=${3:-30}
dur=$(.venv/bin/python -c "import cv2;c=cv2.VideoCapture('$video');print(int(c.get(7)/c.get(5)))")
mkdir -p results/sample
for i in $(seq 0 $((n-1))); do
  start=$(( 120 + i * (dur - 240) / (n - 1) ))
  .venv/bin/python track_ball.py "$video" --start $start --duration $secs \
    --out-csv results/sample/$(basename "$video" .mp4)_$start.csv 2>&1 | grep -i "^Wrote"
done
