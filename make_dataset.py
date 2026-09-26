"""Pick frames for training a better ball detector and pre-label them.

Picks a mix of
  * hard frames found by the tracker (from its CSVs): moments where it hopped between
    objects, had only a low-confidence ball, or lost the ball during play;
  * random gameplay frames spread over the whole matches, for variety.
Every frame is pre-labelled with the current model in YOLO format (class 0 = ball,
class 1 = person), so labellers only fix boxes instead of drawing them. People are kept as
a class because the tracker uses the same model's person boxes (close-ups, in-the-air test).

Train/val are split by time (the last 20% of each match is validation), so near-identical
neighbouring frames can't end up on both sides.

Usage:
    python make_dataset.py videos/*.mp4 --hard results/*.csv --n 400
Output: dataset/ (images, labels, data.yaml) ready to upload to Roboflow or open in CVAT.
"""
import argparse
import csv
import glob
import random
from pathlib import Path

import cv2
import numpy as np

import track_ball as tb

# class ids follow the pre-labelling model, so a fine-tune of that model keeps its class order
# (the team's best.pt: 0 = person, 1 = ball)


def hard_times(csv_paths, width=1280):
    """Times (s) where the tracker struggled, from its per-frame CSVs."""
    times = []
    for p in csv_paths:
        rows = list(csv.DictReader(open(p)))
        seen = [r for r in rows if r["ball_px_x"]]
        for a, b in zip(seen, seen[1:]):  # the pick hopped to another object
            if int(b["frame"]) - int(a["frame"]) <= 2 and np.hypot(
                    float(b["ball_px_x"]) - float(a["ball_px_x"]),
                    float(b["ball_px_y"]) - float(a["ball_px_y"])) > 0.08 * width:
                times += [float(a["time_s"]), float(b["time_s"])]
        for r in rows:
            if r["state"] in ("coasting", "lost", "air"):
                times.append(float(r["time_s"]))
            elif r["state"] == "live" and r["confidence"] and float(r["confidence"]) < 0.35:
                times.append(float(r["time_s"]))
    return times


def spread(times, min_gap):
    """Keep times at least min_gap apart (neighbouring frames are near-duplicates)."""
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= min_gap:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--hard", nargs="*", default=[], help="tracker CSVs to mine for hard frames")
    ap.add_argument("--hard-video", default=None,
                    help="video id (file name without extension) the --hard CSVs came from "
                         "(default: the first video); CSVs don't record their video")
    ap.add_argument("--n", type=int, default=400, help="total frames")
    ap.add_argument("--times", default=None,
                    help="only add these frames to an existing dataset, e.g. 'OFbyNU6UQQs:619.5,636' "
                         "(match seconds); nothing else is sampled")
    ap.add_argument("--hard-share", type=float, default=0.4, help="share of frames taken from hard moments")
    ap.add_argument("--min-gap", type=float, default=1.0, help="seconds between picked frames")
    ap.add_argument("--val-share", type=float, default=0.2, help="last part of each match used for validation")
    ap.add_argument("--weights", default="models/ball_person.pt",
                    help="model used for pre-labelling; its class order is kept")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="detector input size for pre-labelling (2560 for tactical-camera footage)")
    ap.add_argument("--skip-edges", type=float, default=60,
                    help="seconds skipped at the start and end of each video (pre-game footage); "
                         "use ~5 for short clips")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    csvs = [p for pat in args.hard for p in glob.glob(pat)]
    hard_vid = args.hard_video or Path(args.videos[0]).stem
    det = tb.BallDetector(args.weights, tb.pick_device("auto"), args.imgsz, 0.1, precision=32)
    BALL, PERSON = det.class_ids[0], det.person_ids[0]
    names = {BALL: "ball", PERSON: "person"}
    out = Path(args.out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    per_video = args.n // len(args.videos)
    stats = {"train": 0, "val": 0, "hard": 0, "boxes_ball": 0, "no_ball": 0}
    for vpath in args.videos:
        vid = Path(vpath).stem
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        val_from = dur * (1 - args.val_share)

        picks = []
        if args.times:
            want = {v: [float(x) for x in ts.split(",")] for v, ts in
                    (part.split(":") for part in args.times.split(";"))}
            picks = [(t, True) for t in want.get(vid, [])]
            per_video = len(picks)
        elif vid == hard_vid and csvs:
            hard = spread(hard_times(csvs), args.min_gap)
            random.shuffle(hard)
            picks += [(t, True) for t in hard[: int(per_video * args.hard_share)]]
        # random gameplay frames over the whole video (minus --skip-edges at each end); drawn 3x
        # over quota because close-ups and replays get skipped below
        tries, n_random = 0, 0
        while not args.times and n_random < 3 * per_video and tries < per_video * 50:
            tries += 1
            t = random.uniform(args.skip_edges, dur - args.skip_edges)
            if all(abs(t - p) >= args.min_gap for p, _ in picks):
                picks.append((t, False))
                n_random += 1

        written = 0
        randoms = [p for p in picks if not p[1]]
        random.shuffle(randoms)  # random order, so the quota covers the whole match
        for t, is_hard in [p for p in picks if p[1]] + randoms:
            if written >= per_video:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            if not is_hard:
                # random picks: keep wide gameplay shots only (close-ups teach little here)
                if tb.grass_ratio(frame) < 0.35:
                    continue
            balls, people = det.detect(frame)
            h, w = frame.shape[:2]
            if not is_hard and any((y2 - y1) / h > 0.3 for _, y1, _, y2 in people):
                continue
            split = "val" if t >= val_from else "train"
            name = f"{vid}_{t:08.2f}"
            cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            # up to 3 ball guesses, most confident first: the labeller deletes the wrong ones
            for cx, cy, _, conf, bw, bh in sorted(balls, key=lambda c: -c[3])[:3]:
                lines.append(f"{BALL} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            for x1, y1, x2, y2 in people:
                lines.append(f"{PERSON} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[split] += 1
            written += 1
            stats["hard"] += is_hard
            stats["boxes_ball"] += min(len(balls), 3)
            stats["no_ball"] += not balls

    if not args.times:
        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: train/images\nval: val/images\nnames:\n"
            + "".join(f"  {k}: {names[k]}\n" for k in sorted(names)))
    print(f"wrote {stats['train']} train + {stats['val']} val frames to {out}/ "
          f"({stats['hard']} from hard moments; {stats['boxes_ball']} ball guesses to check, "
          f"{stats['no_ball']} frames with no ball guess)")


if __name__ == "__main__":
    main()
