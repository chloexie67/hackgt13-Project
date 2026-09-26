"""Compare pitch mappings on fixed game-1 frames: overlay sheet + numeric check at 650 s.

Usage: python -m tools.eval_calib OUT.png [--no-refine]
"""
import sys, time, cv2, numpy as np
from balltrack.pitch import LINE_PTS, line_coverage, line_mask, line_score
from balltrack.calibration import FieldCalibrator
fc = FieldCalibrator(device="mps", pnl_refine="--no-refine" not in sys.argv)
truth = {(52.5, 0): 159.5, (52.5, 34 - 9.15): 245.5, (52.5, 34): 291, (52.5, 34 + 9.15): 356.5, (52.5, 68): 702.5}
cap = cv2.VideoCapture("videos/OFbyNU6UQQs.mp4"); ims = []
for t in (630, 634, 650, 1513.5, 1622.5, 3075.5, 3379.5):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
    t0 = time.time(); H = fc.homography(f); dt = time.time() - t0
    msg = f"t={t:7.1f}  {'ok  ' if H is not None else 'none'}  {dt*1000:.0f} ms"
    lines = line_mask(f); dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 3)
    if H is not None:
        msg += f"  line score {line_score(H, dist) or 0:.2f} coverage {line_coverage(H, lines):.2f}"
    if t == 650 and H is not None:
        p = cv2.perspectiveTransform(np.float32(list(truth))[None], np.linalg.inv(H))[0]
        msg += f"  | crossings dy {[round(float(p[i,1]-y)) for i, y in enumerate(truth.values())]} dx {[round(float(p[i,0]-257)) for i in range(5)]}"
    print(msg)
    vis = f.copy()
    if H is not None:
        for q in cv2.perspectiveTransform(LINE_PTS[None], np.linalg.inv(H))[0]:
            if 0 <= q[0] < 1280 and 0 <= q[1] < 720:
                cv2.circle(vis, (int(q[0]), int(q[1])), 2, (255, 255, 0), -1)
    vis = cv2.resize(vis, (640, 360)); cv2.putText(vis, str(int(t)), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2); ims.append(vis)
ims.append(np.zeros_like(ims[0]))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, 8, 2)]))
