"""Our goal is to create a way for the visually impaired to experience soccer.

General outline:

 1. Used existing machine learning library YOLO found from research on Roboflow to detect the soccer ball from broadcast footage
 2. To connect our software to the hardware, we mapped the ball's position on screen to real field coordinates with the center spot at (0,0). We used PnLCalib, which fits the camera to the field's landmarks and painted lines.
 3. Due to lack of accuracy, we cleaned data and trained our own model to increase coverage and precision. We labeled frames on Roboflow, including frames from our review of the tracker's mistakes, and trained in Colab and on a laptop.
 4. Used a Kalman filter to smooth position and velocity. We filtered out close-ups, replays and off-pitch objects. Position is not recorded while the ball is in the air.
 5. Tested and validated against real match video. We reviewed annotated videos frame by frame, checked field positions against the pitch markings, and compared velocities with speeds measured from the raw positions.
 6. Pre-processed the match clip into a timeline of x-y coordinates and x-y velocities (one velocity per pass). For the demo, demo.py plays the clip with sound and sends the timeline to the ESP32 in sync.


Pipeline per frame:
    1. Scene filter   - pause on replays / close-ups / crowd shots (little grass, or a
                        player filling the frame).
    2. Ball detection - YOLO; picks the candidate most consistent with the last position.
    3. Pitch mapping  - PnLCalib (fixed landmarks + visible lines) calibrates the camera about
                        once a second; camera motion is tracked in between. The ball's
                        ground point is mapped to metres on a 105 x 68 pitch (homography).
    3b. In the air    - a lofted ball's ground mapping is wrong, so while it's airborne
                        (gravity-shaped path on screen) no position is reported; tracking
                        restarts from where it lands.
    4. Smoothing      - constant-velocity Kalman filter in pitch coordinates; coasts through
                        short occlusions and gives velocity.
    5. Output         - CSV per frame, and a timeline for demo.py: position and one velocity
                        per pass (passes.py), relative to the centre spot (0, 0). Optional
                        annotated video with a radar minimap.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from balltrack.calibration import FieldCalibrator, KeypointPitchMapper, PitchTracker
from balltrack.detection import BallDetector, plausible_balls
from balltrack.device import pick_device
from balltrack.flight import AirDetector, at_feet, vertical_scale
from balltrack.kalman import BallKalman
from balltrack.overlay import annotate
from balltrack.pitch import PITCH_LENGTH, PITCH_WIDTH, on_pitch, to_pitch
from balltrack.scene import grass_ratio, promo_frame
from balltrack.selection import BallSelector
from balltrack.passes import describe, load, merge, segment
from balltrack.sources import FileSource

CSV_FIELDS = [
    "frame", "time_s", "state", "confidence",
    "ball_px_x", "ball_px_y", "screen_x", "screen_y",
    "field_x_m", "field_y_m", "field_vx_ms", "field_vy_ms",
    "field_x_norm", "field_y_norm", "has_homography",
    "air_accel_ms2", "track_conf", "n_detected", "n_plausible",
    "raw_x_m", "raw_y_m", "kf_restart",
]

TIMELINE_FIELDS = ["time_s", "valid", "x_m", "y_m", "vx_ms", "vy_ms", "pass_id"]


def open_source(args):
    path = args.source
    if path.startswith("http"):
        from balltrack.download import download
        path = str(download(path))
    return FileSource(path, args.start, args.duration, args.stride), path


def run(args):
    source, name = open_source(args)
    width, height = source.size

    device = pick_device(args.device)
    precision = 32 if (args.fp32 or device == "cpu") else 16
    print(f"{name}: {width}x{height} @ {source.fps:.1f} fps, device={device}, fp{precision}")

    detector = BallDetector(args.ball_weights, device, args.imgsz, args.conf, precision)
    pitch_tracker = keypoint_mapper = None
    if args.pitch == "pnl":
        pitch_tracker = PitchTracker(FieldCalibrator(device=device), interval=args.calib_interval)
    elif args.pitch == "keypoints":
        keypoint_mapper = KeypointPitchMapper(args.pitch_weights, device, args.kp_conf, args.hold_frames, precision)
    selector = BallSelector(width, max_jump=args.max_jump, confirm_n=args.confirm)
    ball_kf = BallKalman()
    flight = AirDetector()

    out_csv = Path(args.out_csv or Path(name).with_suffix(".ball.csv"))
    csv_file = open(out_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    video_out = None
    if args.out_video:
        video_out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                    source.fps / args.stride, (width, height))

    last_ball_px, missed_s, t_prev = None, 0.0, None
    last_sent_xy = None
    last_ground_t = None
    gameplay_streak = args.resume_frames
    processed = 0
    t_start = t_report = time.time()

    try:
        for frame_no, t, frame in source:
            dt = (t - t_prev) if t_prev is not None else 1 / source.fps
            t_prev = t

            row = {k: "" for k in CSV_FIELDS}
            row.update(frame=frame_no, time_s=round(t, 3), has_homography=0)
            ball_px = None

            gameplay = grass_ratio(frame) >= args.min_grass and not promo_frame(frame)
            balls, people = detector.detect(frame) if gameplay else ([], [])
            tallest = max(((b[3] - b[1]) / height for b in people), default=0.0)
            gameplay_streak = 0 if (not gameplay or tallest > args.max_person) else gameplay_streak + 1
            if gameplay_streak < args.resume_frames:
                row["state"] = "paused"
                ball_kf.reset()
                if keypoint_mapper is not None:
                    keypoint_mapper.reset()
                if pitch_tracker is not None:
                    pitch_tracker.reset()
                flight.reset()
                selector.forget()
                last_ball_px, missed_s = None, 0.0
            else:
                homography = None
                if pitch_tracker is not None:
                    homography = pitch_tracker.update(frame, t)
                elif keypoint_mapper is not None:
                    homography = (keypoint_mapper.update(frame) if processed % args.pitch_every == 0
                                  else keypoint_mapper.hold())
                row["has_homography"] = int(homography is not None)

                ball_kf.predict(dt)
                row["n_detected"] = len(balls)
                balls = plausible_balls(balls, homography, frame, allow_off_pitch=flight.airborne,
                                        off_pitch_margin=args.off_pitch_margin)
                row["n_plausible"] = len(balls)
                ball, switched = selector.choose(balls, t, ball_kf, homography)

                if ball is not None:
                    cx, cy, bottom, conf = ball[:4]
                    prev_px, missed_before = last_ball_px, missed_s
                    ball_px, last_ball_px, missed_s = (cx, cy), (cx, cy), 0.0
                    row.update(state="live", confidence=round(conf, 3),
                               ball_px_x=round(cx, 1), ball_px_y=round(cy, 1),
                               screen_x=round(cx / width, 4), screen_y=round(cy / height, 4))
                    ground = None
                    if homography is not None:
                        gx, gy = to_pitch(homography, cx, bottom)
                        ground = (gx, gy) if on_pitch(gx, gy) else None

                    if switched or prev_px is None or missed_before > 0 or \
                            np.hypot(cx - prev_px[0], cy - prev_px[1]) > 0.08 * width:
                        flight.break_track()
                    y_stable = cy
                    if pitch_tracker is not None:
                        y_stable = cv2.perspectiveTransform(np.float32([[[cx, cy]]]),
                                                            np.linalg.inv(pitch_tracker.cum))[0, 0, 1]
                    if switched:
                        for t_seen, seen in selector.confirmed_track:
                            y_seen = seen[1]
                            if pitch_tracker is not None:
                                y_seen = cv2.perspectiveTransform(np.float32([[seen[:2]]]),
                                                                  np.linalg.inv(pitch_tracker.cum))[0, 0, 1]
                            flight.update(t_seen, float(y_seen), vertical_scale(people, seen[0], seen[1]),
                                          on_ground=at_feet(people, seen[0], seen[1]))
                    was_airborne = flight.airborne
                    airborne = flight.update(t, float(y_stable), vertical_scale(people, cx, cy),
                                             on_ground=at_feet(people, cx, cy))
                    row["air_accel_ms2"] = "" if flight.accel is None else round(flight.accel, 2)
                    if airborne:
                        row["state"] = "air"  # positions are wrong while the ball is in the air
                        ball_kf.reset()
                    elif ground is not None:
                        if was_airborne or switched:
                            ball_kf.reset()  # just landed, or re-acquired elsewhere: start here
                        row.update(raw_x_m=round(ground[0] - PITCH_LENGTH / 2, 3),
                                   raw_y_m=round(ground[1] - PITCH_WIDTH / 2, 3), kf_restart=int(ball_kf.x is None))
                        ball_kf.update(ground)
                        last_ground_t = t
                elif flight.airborne and t - flight.since <= flight.max_air_s:
                    missed_s += dt
                    row["state"] = "air"
                else:
                    missed_s += dt
                    if missed_s > args.max_coast:
                        row["state"] = "lost"
                        ball_kf.reset()
                        flight.reset()
                        last_ball_px = None
                    else:
                        row["state"] = "coasting"

                if row["state"] in ("live", "coasting"):
                    row["track_conf"] = round(selector.conf_ema, 3)
                if row["state"] in ("live", "coasting") and selector.conf_ema < args.min_report_conf:
                    row["state"] = "uncertain"  # this should filter out low certainty things
                fresh = last_ground_t is not None and t - last_ground_t <= args.max_coast
                if row["state"] in ("live", "coasting") and not fresh:
                    row["state"] = "unmapped" if homography is None else row["state"]
                if ball_kf.x is not None and row["state"] in ("live", "coasting") and fresh:
                    x, y, vx, vy = ball_kf.x
                    x, y = float(np.clip(x, 0, PITCH_LENGTH)), float(np.clip(y, 0, PITCH_WIDTH))
                    if last_sent_xy is not None:
                        step, cap = np.hypot(x - last_sent_xy[0], y - last_sent_xy[1]), args.max_speed * dt
                        if step > cap:
                            x = last_sent_xy[0] + (x - last_sent_xy[0]) * cap / step
                            y = last_sent_xy[1] + (y - last_sent_xy[1]) * cap / step
                    last_sent_xy = (x, y)
                    x, y = x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2
                    vel_ready = ball_kf.age >= args.min_vel_age
                    row.update(field_x_m=round(x, 2), field_y_m=round(y, 2),
                               field_vx_ms=round(float(vx), 2) if vel_ready else "",
                               field_vy_ms=round(float(vy), 2) if vel_ready else "",
                               field_x_norm=round(x / (PITCH_LENGTH / 2), 4),
                               field_y_norm=round(y / (PITCH_WIDTH / 2), 4))

            if row["field_x_m"] == "":
                last_sent_xy = None
            writer.writerow(row)

            if video_out is not None or args.show:
                radar_xy = ((row["field_x_m"] + PITCH_LENGTH / 2, row["field_y_m"] + PITCH_WIDTH / 2)
                            if row["field_x_m"] != "" else None)
                annotate(frame, row, ball_px, radar_xy)
                if video_out is not None:
                    video_out.write(frame)
                if args.show:
                    cv2.imshow("ball tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            processed += 1
            now = time.time()
            if now - t_report > 2:
                pct = f" {100 * frame_no / source.total:.0f}%" if source.total else ""
                print(f"frame {frame_no}{pct}  {processed / (now - t_start):.1f} fps  state={row['state']}")
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        csv_file.close()
        if video_out is not None:
            video_out.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"Processed {processed} frames in {elapsed:.1f} s ({processed / max(elapsed, 1e-9):.1f} fps)")
    print(f"Wrote {out_csv}" + (f" and {args.out_video}" if args.out_video else ""))

    # Timeline for demo.py: position every frame, and one velocity per pass so the motors hold a
    # steady speed for each pass. Between passes (dribbling, a still ball) the filter's velocity.
    t_raw, xy_raw, restarts = load(out_csv)
    passes = describe(t_raw, xy_raw, merge(t_raw, xy_raw, segment(t_raw, xy_raw, restarts))) if len(t_raw) else []
    out_timeline = Path(args.out_timeline or out_csv.with_name(out_csv.name.replace(".ball.csv", "") + ".timeline.csv"))
    with open(out_csv) as f_rows, open(out_timeline, "w", newline="") as f_out:
        timeline = csv.DictWriter(f_out, fieldnames=TIMELINE_FIELDS)
        timeline.writeheader()
        for row in csv.DictReader(f_rows):
            t, valid = float(row["time_s"]), row["field_x_m"] != ""
            pass_id = next((k for k, ps in enumerate(passes) if ps["t_start"] <= t <= ps["t_end"]), None)
            if pass_id is not None:
                vx, vy = round(passes[pass_id]["vx"], 2), round(passes[pass_id]["vy"], 2)
            else:
                vx, vy = float(row["field_vx_ms"] or 0.0), float(row["field_vy_ms"] or 0.0)
            # faster than any ball on the grass: it's in the air (its ground position is wrong)
            # or the mapping slipped, so send nothing rather than a wrong position and speed
            if np.hypot(vx, vy) > args.max_speed:
                valid = False
            if not valid:
                vx = vy = ""
            timeline.writerow(dict(time_s=row["time_s"], valid=int(valid),
                                   x_m=row["field_x_m"] if valid else "", y_m=row["field_y_m"] if valid else "",
                                   vx_ms=vx, vy_ms=vy, pass_id="" if pass_id is None or not valid else pass_id))
    print(f"Wrote {out_timeline} ({len(passes)} passes) - play it with: python demo.py {name} {out_timeline}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file or YouTube URL")
    p.add_argument("--ball-weights",
                   default="models/ball_person.pt" if Path("models/ball_person.pt").exists() else "yolo11m.pt",
                   help="YOLO weights with 'ball' and 'person' classes (default: the team's fine-tuned "
                        "models/ball_person.pt; falls back to COCO yolo11m)")
    p.add_argument("--pitch", choices=["pnl", "keypoints", "none"], default="pnl",
                   help="pitch mapping: pnl = PnLCalib landmarks + lines (accurate, default); "
                        "keypoints = roboflow/sports landmark model only (metres off); "
                        "none = screen position only")
    p.add_argument("--calib-interval", type=float, default=1.0,
                   help="seconds between PnLCalib calibrations; camera motion is tracked in between")
    p.add_argument("--pitch-weights", default="models/football-pitch-detection.pt",
                   help="landmark model for --pitch keypoints")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    p.add_argument("--imgsz", type=int, default=1280, help="inference size; larger finds smaller balls")
    p.add_argument("--conf", type=float, default=0.15, help="min ball detection confidence")
    p.add_argument("--kp-conf", type=float, default=0.5, help="min pitch keypoint confidence")
    p.add_argument("--start", type=float, default=0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="seconds to process (default: to the end)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--pitch-every", type=int, default=3,
                   help="--pitch keypoints: run the landmark model every Nth processed frame")
    p.add_argument("--hold-frames", type=int, default=10, help="reuse last homography this many frames")
    p.add_argument("--max-coast", type=float, default=0.2, help="seconds to predict through before 'lost'")
    p.add_argument("--max-jump", type=float, default=0.6,
                   help="max ball speed on screen, in frame-widths per second")
    p.add_argument("--confirm", type=int, default=3,
                   help="frames a ball seen somewhere unexpected must persist before it's accepted")
    p.add_argument("--min-report-conf", type=float, default=0.20,
                   help="positions are only reported while the tracked ball's recent detection "
                        "confidence is at least this (state 'uncertain' otherwise)")
    p.add_argument("--off-pitch-margin", type=float, default=0.5,
                   help="metres; ball candidates mapped further outside the pitch lines are ignored "
                        "(spare balls, ad boards)")
    p.add_argument("--min-vel-age", type=float, default=0.2,
                   help="seconds of tracking needed after a restart before velocity is reported")
    p.add_argument("--max-speed", type=float, default=35.0,
                   help="m/s; the reported position never moves faster than this")
    p.add_argument("--max-person", type=float, default=0.3,
                   help="a person taller than this fraction of the frame means close-up (COCO models only)")
    p.add_argument("--resume-frames", type=int, default=8,
                   help="consecutive gameplay frames needed before tracking resumes after a pause")
    p.add_argument("--min-grass", type=float, default=0.35, help="min grass fraction for a gameplay shot")
    p.add_argument("--out-csv", default=None, help="default: <video>.ball.csv")
    p.add_argument("--out-timeline", default=None, help="default: <video>.timeline.csv (for demo.py)")
    p.add_argument("--out-video", default=None, help="write an annotated video with radar minimap")
    p.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    p.add_argument("--fp32", action="store_true", help="disable half precision on the GPU")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
