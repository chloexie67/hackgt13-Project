"""Split the tracked ball path into passes and give each pass ONE velocity for the motors.

A pass (or any single ball movement between touches) is modelled as a straight line on the
pitch travelled at a speed that stays level or falls (rolling friction), never rises. The raw
ground positions are cut greedily into the longest pieces that fit that model; a piece ends when
  * the path bends (points drift more than max_lateral_m from the straight line),
  * the ball speeds up (fitted acceleration along the line above max_accel), i.e. a touch,
  * the along-line fit gets poor (rms above max_rms_m), or
  * tracking was interrupted (restart, or a gap longer than max_gap_s).
Each pass gets one velocity: displacement from its start to its end divided by its duration,
so driving the motors at that velocity moves them from the pass's start to its end on time.

Usage: python passes.py TRACKER_CSV [--from S --to S] [--offset S] [--out passes.csv]
"""
import argparse
import csv

import numpy as np


def load(path):
    rows = [x for x in csv.DictReader(open(path))]
    pts = [(float(x["time_s"]), float(x["raw_x_m"]), float(x["raw_y_m"]), x["kf_restart"] == "1")
           for x in rows if x.get("raw_x_m")]
    return np.array([p[0] for p in pts]), np.array([[p[1], p[2]] for p in pts]), np.array([p[3] for p in pts])


def fits(t, xy, max_lateral_m, max_rms_m, max_accel):
    """Does this run of points look like one straight, non-accelerating movement?"""
    c = xy.mean(0)
    d = xy - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    along, lateral = d @ axis, d @ np.array([-axis[1], axis[0]])
    if np.abs(lateral).max() > max_lateral_m:
        return False
    tt = t - t[0]
    if len(t) >= 4:
        coef = np.polyfit(tt, along, 2)
        rms = float(np.sqrt(np.mean((np.polyval(coef, tt) - along) ** 2)))
        v0 = coef[1]
        accel_along_motion = 2 * coef[0] * np.sign(v0 if abs(v0) > 1e-6 else 1)
        if rms > max_rms_m or accel_along_motion > max_accel:
            return False
    return True


def segment(t, xy, restart, max_lateral_m=0.6, max_rms_m=0.25, max_accel=2.0, max_gap_s=0.3,
            min_points=4):
    passes, i, n = [], 0, len(t)
    while i < n:
        j = i + 1
        while j < n and not restart[j] and t[j] - t[j - 1] <= max_gap_s and \
                fits(t[i:j + 1], xy[i:j + 1], max_lateral_m, max_rms_m, max_accel):
            j += 1
        if j - i >= min_points:
            passes.append((i, j - 1))
            i = j - 1 if j < n and not restart[j] and t[j] - t[j - 1] <= max_gap_s else j
        else:
            i += 1
    return passes


def merge(t, xy, passes, max_gap_s=0.12, max_turn_deg=30.0, max_speedup=1.0):
    """Join consecutive pieces that are really one pass slowing down: they touch in time, keep
    the same direction, and the later one isn't faster."""
    if not passes:
        return passes
    out = [passes[0]]
    for a, b in passes[1:]:
        pa, pb = out[-1], (a, b)
        va, vb = [describe(t, xy, [q])[0] for q in (pa, pb)]
        ua, ub = np.array([va["vx"], va["vy"]]), np.array([vb["vx"], vb["vy"]])
        same_dir = (np.linalg.norm(ua) > 0 and np.linalg.norm(ub) > 0 and
                    ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)) > np.cos(np.radians(max_turn_deg)))
        if t[a] - t[pa[1]] <= max_gap_s and same_dir and vb["speed"] <= va["speed"] + max_speedup:
            out[-1] = (pa[0], b)
        else:
            out.append(pb)
    return out


def describe(t, xy, passes):
    out = []
    for a, b in passes:
        tt = t[a:b + 1] - t[a]
        # start/end from a linear fit, so one noisy endpoint doesn't skew the velocity
        p = [np.polyfit(tt, xy[a:b + 1, k], 2) for k in range(2)]
        start = np.array([np.polyval(p[k], 0) for k in range(2)])
        end = np.array([np.polyval(p[k], tt[-1]) for k in range(2)])
        dur = tt[-1]
        v = (end - start) / dur if dur > 0 else np.zeros(2)
        out.append(dict(t_start=t[a], t_end=t[b], duration=dur, x0=start[0], y0=start[1], x1=end[0], y1=end[1],
                        vx=v[0], vy=v[1], speed=float(np.hypot(*v)), points=b - a + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--offset", type=float, default=0.0, help="added to times when printing (video time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t, xy, restart = load(args.csv)
    ps = describe(t, xy, merge(t, xy, segment(t, xy, restart)))
    if args.t0 is not None:
        ps = [p for p in ps if p["t_end"] >= args.t0 and (args.t1 is None or p["t_start"] <= args.t1)]
    fmt = lambda s: f"{int((s + args.offset) // 60)}:{(s + args.offset) % 60:04.1f}"
    print(f"{'start':>7} {'end':>7} {'dur':>5} | {'from (x, y)':>14} -> {'to (x, y)':>14} | {'vx':>6} {'vy':>6} {'speed':>6}")
    for p in ps:
        print(f"{fmt(p['t_start']):>7} {fmt(p['t_end']):>7} {p['duration']:5.2f} | "
              f"({p['x0']:6.1f},{p['y0']:6.1f}) -> ({p['x1']:6.1f},{p['y1']:6.1f}) | "
              f"{p['vx']:6.1f} {p['vy']:6.1f} {p['speed']:6.1f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start", "end", "duration_s", "x_start_m", "y_start_m", "x_end_m", "y_end_m",
                        "vx_ms", "vy_ms", "speed_ms"])
            for p in ps:
                w.writerow([fmt(p["t_start"]), fmt(p["t_end"]), round(p["duration"], 2),
                            *(round(p[k], 2) for k in ("x0", "y0", "x1", "y1", "vx", "vy", "speed"))])
        print("wrote", args.out)


if __name__ == "__main__":
    main()
