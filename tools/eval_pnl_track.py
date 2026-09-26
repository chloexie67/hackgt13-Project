"""Run PnLCalib + camera-motion tracking over a stretch; report coverage, speed, accuracy.

Usage: python eval_pnl_track.py VIDEO START_S DURATION_S [INTERVAL_S]
"""
import sys, time, collections, cv2, numpy as np
from balltrack.pitch import LINE_PTS, line_mask, line_score
from balltrack.scene import grass_ratio
from balltrack.calibration import FieldCalibrator, PitchTracker
video, start, dur = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
pt = PitchTracker(FieldCalibrator(device="mps"), interval=interval)
cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
fps = cap.get(cv2.CAP_PROP_FPS); n = int(dur * fps)
truth = {(52.5, 0): 159.5, (52.5, 34 - 9.15): 245.5, (52.5, 34): 291, (52.5, 34 + 9.15): 356.5, (52.5, 68): 702.5}
state, times, snaps, scores = collections.Counter(), [], [], []
for i in range(n):
    ok, f = cap.read()
    if not ok: break
    t = start + i / fps
    t0 = time.time()
    if grass_ratio(f) < 0.35:
        pt.reset(); H = None; st = "not gameplay"
    else:
        H = pt.update(f, i, t); st = "mapped" if H is not None else "none"
    times.append(time.time() - t0); state[st] += 1
    if H is not None and i % 5 == 0:
        m = line_mask(f); d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        s = line_score(H, d)
        if s is not None: scores.append(s)
    if abs(t - 650) < 0.5 / fps and H is not None and "OFbyNU6UQQs" in video:
        p = cv2.perspectiveTransform(np.float32(list(truth))[None], np.linalg.inv(H))[0]
        print(f"t=650 ({(t - pt.last_calib_t):.2f} s after last calibration): crossings dy "
              f"{[round(float(p[k,1]-y)) for k, y in enumerate(truth.values())]}")
    if i % max(1, n // 8) == 0:
        vis = f.copy()
        if H is not None:
            for q in cv2.perspectiveTransform(LINE_PTS[None], np.linalg.inv(H))[0]:
                if 0 <= q[0] < vis.shape[1] and 0 <= q[1] < vis.shape[0]:
                    cv2.circle(vis, (int(q[0]), int(q[1])), 2, (255, 255, 0), -1)
        vis = cv2.resize(vis, (640, 360))
        cv2.putText(vis, f"{t:.1f}s {st}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2); snaps.append(vis)
tot = sum(state.values())
print("frames:", tot, {k: f"{v/tot:.0%}" for k, v in state.most_common()}, f"calibrations {pt.calibrations}, rejected {pt.rejected}")
print(f"line agreement of mapped frames (share of projected line points on a painted line): median {np.median(scores):.2f}, p10 {np.percentile(scores,10):.2f}")
ts = np.array(times) * 1000
print(f"time per frame: median {np.median(ts):.1f} ms, mean {ts.mean():.0f} ms, max {ts.max():.0f} ms")
while len(snaps) % 2: snaps.append(np.zeros_like(snaps[0]))
cv2.imwrite("results/pnl_track_sheet.png", np.vstack([np.hstack(snaps[k:k + 2]) for k in range(0, len(snaps), 2)]))
