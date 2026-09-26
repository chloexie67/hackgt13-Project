"""Replay recorded raw ground positions through Kalman filter variants and compare their
velocity with a lag-free reference (local quadratic fit using frames before and after).
Usage: python tune_kalman.py results/cx_raw.csv"""
import csv, sys
import numpy as np
from balltrack.kalman import BallKalman

r = [x for x in csv.DictReader(open(sys.argv[1])) if x["raw_x_m"]]
t = np.array([float(x["time_s"]) for x in r])
P = np.array([[float(x["raw_x_m"]), float(x["raw_y_m"])] for x in r])
restart = np.array([x["kf_restart"] == "1" for x in r])

ref = np.full(len(t), np.nan)  # lag-free speed reference
for i in range(len(t)):
    m = abs(t - t[i]) <= 0.2
    seg = np.cumsum(restart[m])  # don't fit across a restart
    if m.sum() >= 7 and not restart[m][1:].any():
        tt = t[m] - t[i]
        ref[i] = np.hypot(*[np.polyfit(tt, P[m, k], 2)[1] for k in range(2)])

def replay(**kw):
    kf, prev, out = BallKalman(**kw), None, []
    for i in range(len(t)):
        if restart[i] or (prev is not None and t[i] - prev > 0.3):
            kf.reset()
        if kf.x is not None:
            kf.predict(t[i] - prev)
        kf.update(P[i]); prev = t[i]
        out.append(np.hypot(*kf.x[2:]))
    return np.array(out)

ok = ~np.isnan(ref)
kick = np.argmin(abs(t - 5.9))
for name, kw in [("current (meas 1.5 m)", dict(meas_std=1.5, maneuver=False)),
                 ("meas 0.15 m", dict(meas_std=0.15, maneuver=False)),
                 ("meas 0.15 m + kick detection", dict(meas_std=0.15, maneuver=True)),
                 ("meas 0.3 m + kick detection", dict(meas_std=0.3, maneuver=True))]:
    try:
        v = replay(**kw)
    except TypeError:
        print(f"{name:32s} (not implemented yet)"); continue
    err = abs(v[ok] - ref[ok])
    print(f"{name:32s} speed error vs reference: mean {err.mean():4.1f} m/s, 90th pct {np.percentile(err,90):4.1f} | "
          f"at 6:05.9 after the kick: {v[kick]:4.1f} (reference {ref[kick]:4.1f})")
