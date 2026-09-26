"""Summarize tracker CSVs: state mix, field coverage, and physically impossible jumps."""
import csv, sys, collections, numpy as np
for f in sys.argv[1:]:
    r = list(csv.DictReader(open(f)))
    st = collections.Counter(x["state"] for x in r); n = len(r)
    fx = [(float(x["time_s"]), float(x["field_x_m"]), float(x["field_y_m"])) for x in r if x["field_x_m"]]
    jumps = sum(1 for a, b in zip(fx, fx[1:]) if b[0] - a[0] < 0.2 and np.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) > 40)
    print(f"{f:22s} live {st['live']/n:4.0%} coast {st['coasting']/n:4.0%} lost {st['lost']/n:4.0%} "
          f"paused {st['paused']/n:4.0%} | field pos {len(fx)/n:4.0%} | >40 m/s jumps {jumps}")
