"""One velocity per pass for a recorded clip, from the tracker's CSV.

Usage: python -m tools.pass_velocities TRACKER_CSV [--from S --to S] [--offset S] [--out passes.csv]
"""
import argparse
import csv

from balltrack.passes import describe, load, merge, segment


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
