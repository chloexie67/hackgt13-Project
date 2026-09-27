"""Our goal is to create a way for the visually impaired to experience soccer.

General outline:

 1. Used existing machine learning library YOLO found from research on Roboflow to detect the soccer ball from broadcast footage
 2. To connect our software to the hardware, we mapped the ball's position on screen to real field coordinates with the center spot at (0,0). We used PnLCalib, which fits the camera to the field's landmarks and painted lines.
 3. Due to lack of accuracy, we cleaned data and trained our own model to increase coverage and precision. We labeled frames on Roboflow, including frames from our review of the tracker's mistakes, and trained in Colab and on a laptop.
 4. Used a Kalman filter to smooth position and velocity. We filtered out close-ups, replays and off-pitch objects. Position is not recorded while the ball is in the air.
 5. Tested and validated against real match video. We reviewed annotated videos frame by frame, checked field positions against the pitch markings, and compared velocities with speeds measured from the raw positions.
 6. Sent x-y coordinates and x-y velocities to the ESP32 over UDP (Wi-Fi). Information is sent live within 0.3 seconds of the kick, or as one velocity per pass for recorded games.


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
    5. Output         - CSV per frame, optional annotated video with a radar minimap,
                        optional UDP JSON stream for the haptic device. Positions are
                        relative to the centre spot (0, 0).
"""
import argparse
import csv
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from balltrack.calibration import FieldCalibrator, KeypointPitchMapper, PitchTracker
from balltrack.detection import BallDetector, plausible_balls
from balltrack.device import pick_device
from balltrack.flight import AirDetector, at_feet, vertical_scale
from balltrack.kalman import BallKalman
from balltrack.live_passes import LivePasses
from balltrack.motor_speed import MotorSpeed
from balltrack.overlay import annotate
from balltrack.pitch import PITCH_LENGTH, PITCH_WIDTH, on_pitch, to_pitch
from balltrack.scene import grass_ratio, promo_frame
from balltrack.selection import BallSelector
from balltrack.sources import FileSource, LiveSource

CSV_FIELDS = [
    "frame", "time_s", "state", "confidence",
    "ball_px_x", "ball_px_y", "screen_x", "screen_y",
    "field_x_m", "field_y_m", "field_vx_ms", "field_vy_ms",
    "field_x_norm", "field_y_norm", "has_homography",
    "air_accel_ms2", "track_conf", "n_detected", "n_plausible",
    "raw_x_m", "raw_y_m", "kf_restart",
    "motor_t", "motor_speed_ms", "motor_vx_ms", "motor_vy_ms", "motor_touch",
    "live_vx_ms", "live_vy_ms", "live_speed_ms",
    "pass_t0", "pass_vx0_ms", "pass_vy0_ms", "pass_decel_ms2", "new_pass",
]

UDP_FIELDS = {
    "t": "time_s", "state": "state", "x": "field_x_norm", "y": "field_y_norm",
    "vx": "field_vx_ms", "vy": "field_vy_ms", "sx": "screen_x", "sy": "screen_y",
    "confidence": "confidence",
    "motor_t": "motor_t", "motor_speed": "motor_speed_ms", "motor_vx": "motor_vx_ms",
    "motor_vy": "motor_vy_ms", "touch": "motor_touch",
    "live_vx": "live_vx_ms", "live_vy": "live_vy_ms", "live_speed": "live_speed_ms", "new_pass": "new_pass",
    "pass_t0": "pass_t0", "pass_vx0": "pass_vx0_ms", "pass_vy0": "pass_vy0_ms", "pass_decel": "pass_decel_ms2",
}


def open_source(args):
    if args.source == "screen" or args.source.isdigit():
        region = tuple(int(v) for v in args.region.split(",")) if args.region else None
        return LiveSource(args.source, region), args.source
    path = args.source
    if path.startswith("http"):
        from balltrack.download import download
        path = str(download(path))
    return FileSource(path, args.start, args.duration, args.stride, args.realtime), path


def run(args):
    source, name = open_source(args)
    width, height = source.size
    live = isinstance(source, LiveSource)

    device = pick_device(args.device)
    precision = 32 if (args.fp32 or device == "cpu") else 16
    print(f"{name}: {width}x{height} @ {source.fps:.1f} fps, device={device}, fp{precision}")

    detector = BallDetector(args.ball_weights, device, args.imgsz, args.conf, precision)
    pitch_tracker = keypoint_mapper = None
    if args.pitch == "pnl":
        background = live or args.realtime
        calib_device = args.calib_device or ("cpu" if background else device)
        pitch_tracker = PitchTracker(FieldCalibrator(device=calib_device), interval=args.calib_interval,
                                     async_=background, max_stale=6.0 if calib_device == "cpu" else 4.0)
    elif args.pitch == "keypoints":
        keypoint_mapper = KeypointPitchMapper(args.pitch_weights, device, args.kp_conf, args.hold_frames, precision)
    selector = BallSelector(width, max_jump=args.max_jump, confirm_n=args.confirm)
    ball_kf = BallKalman()
    flight = AirDetector()
    motor_signal = MotorSpeed(delay=args.motor_delay)
    live_passes = LivePasses(mode=args.pass_mode)

    out_csv = Path(args.out_csv or ("live.ball.csv" if live else Path(name).with_suffix(".ball.csv")))
    csv_file = open(out_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    video_out = None
    if args.out_video:
        video_out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                    source.fps / (1 if live else args.stride), (width, height))
    udp_sock = udp_addr = None
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        udp_sock, udp_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))

    last_ball_px, missed_s, t_prev = None, 0.0, None
    last_sent_xy = None
    last_ground_t = None
    gameplay_streak = args.resume_frames
    processed = 0
    t_start = t_report = time.time()
    busy_s = 0.0

    try:
        for frame_no, t, frame in source:
            t_work = time.time()
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
                    homography = pitch_tracker.update(frame, frame_no, t)
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
            raw_xy = (row["raw_x_m"], row["raw_y_m"]) if row["raw_x_m"] != "" else None
            pass_event = live_passes.push(t, raw_xy, row["kf_restart"] == 1)
            motor_velocity = live_passes.velocity(t)
            if row["state"] in ("paused", "lost", "unmapped"):
                motor_velocity = None
            if motor_velocity is not None:
                current_pass = live_passes.current_pass
                row.update(live_vx_ms=round(motor_velocity[0], 2), live_vy_ms=round(motor_velocity[1], 2),
                           live_speed_ms=round(float(np.hypot(*motor_velocity)), 2),
                           pass_t0=round(current_pass["t0"], 3),
                           pass_vx0_ms=round(float(current_pass["v0"][0]), 2),
                           pass_vy0_ms=round(float(current_pass["v0"][1]), 2),
                           pass_decel_ms2=live_passes.decel if args.pass_mode == "profile" else 0.0)
            row["new_pass"] = int(pass_event is not None)
            for sample in motor_signal.push(t, raw_xy, row["kf_restart"] == 1):
                row.update(motor_t=round(sample["t"], 3),
                           motor_speed_ms="" if sample["speed"] is None else round(sample["speed"], 2),
                           motor_vx_ms="" if sample["vx"] is None else round(sample["vx"], 2),
                           motor_vy_ms="" if sample["vy"] is None else round(sample["vy"], 2),
                           motor_touch=int(sample["touch"]))
            writer.writerow(row)
            if udp_sock is not None:
                message = {key: (None if row[col] == "" else row[col]) for key, col in UDP_FIELDS.items()}
                udp_sock.sendto(json.dumps(message).encode(), udp_addr)

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
            busy_s += now - t_work
            if args.print and now - t_report >= 0.1:
                shown = {k: f"{row[k]:7.2f}" if row[k] != "" else "      -"
                         for k in ("field_x_m", "field_y_m", "field_vx_ms", "field_vy_ms",
                                   "live_vx_ms", "live_vy_ms", "live_speed_ms")}
                print(f"{t:8.2f}s  {row['state']:9}  x {shown['field_x_m']}  y {shown['field_y_m']}  "
                      f"vx {shown['field_vx_ms']}  vy {shown['field_vy_ms']}  |  motor vx {shown['live_vx_ms']}  "
                      f"vy {shown['live_vy_ms']}  speed {shown['live_speed_ms']}"
                      + ("  NEW PASS" if row["new_pass"] == 1 else ""))
                t_report = now
            elif not args.print and now - t_report > 2:
                elapsed = now - t_start
                pct = f" {100 * frame_no / source.total:.0f}%" if source.total else ""
                print(f"frame {frame_no}{pct}  {processed / elapsed:.1f} fps processed  busy {busy_s / elapsed:.0%}  "
                      f"dropped {source.dropped}  state={row['state']}")
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
    print(f"Processed {processed} frames in {elapsed:.1f} s ({processed / max(elapsed, 1e-9):.1f} fps), "
          f"busy {busy_s / max(elapsed, 1e-9):.0%}, dropped {source.dropped}")
    print(f"Wrote {out_csv}" + (f" and {args.out_video}" if args.out_video else ""))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, YouTube URL, 'screen', or a camera index (e.g. 0 for OBS Virtual Camera)")
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
    p.add_argument("--calib-device", default=None,
                   help="device for PnLCalib (default: cpu when live/--realtime so the ball detector "
                        "keeps the GPU, otherwise the main device)")
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
    p.add_argument("--pass-mode", choices=["profile", "constant"], default="profile",
                   help="live pass output: 'profile' = start speed slowing at a standard rate; "
                        "'constant' = one fixed speed per pass (start speed x typical ratio)")
    p.add_argument("--motor-delay", type=float, default=0.15,
                   help="seconds of look-ahead for the motor speed signal (it lags by this much)")
    p.add_argument("--max-speed", type=float, default=35.0,
                   help="m/s; the reported position never moves faster than this")
    p.add_argument("--max-person", type=float, default=0.3,
                   help="a person taller than this fraction of the frame means close-up (COCO models only)")
    p.add_argument("--resume-frames", type=int, default=8,
                   help="consecutive gameplay frames needed before tracking resumes after a pause")
    p.add_argument("--min-grass", type=float, default=0.35, help="min grass fraction for a gameplay shot")
    p.add_argument("--out-csv", default=None, help="default: <video>.ball.csv")
    p.add_argument("--out-video", default=None, help="write an annotated video with radar minimap")
    p.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    p.add_argument("--print", action="store_true",
                   help="print position (m), velocity (m/s) and the motor velocity 10 times a second")
    p.add_argument("--udp", default=None, help="stream JSON to HOST:PORT (e.g. an ESP32)")
    p.add_argument("--realtime", action="store_true",
                   help="treat a video file like a live feed: run at video speed, drop frames when behind")
    p.add_argument("--region", default=None,
                   help="for 'screen': left,top,width,height of the capture area in screen points")
    p.add_argument("--fp32", action="store_true", help="disable half precision on the GPU")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
