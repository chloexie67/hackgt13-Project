# Soccer ball tracker for a haptic device

Tracks the ball in soccer video (TV broadcasts or tactical-camera recordings, recorded or live)
and turns it into positions and pass velocities that drive haptic motors, so visually impaired
fans can feel where the ball is and how it moves.

## Pipeline

1. **Scene filter**: pauses on replays, close-ups, crowd shots and promo frames.
2. **Ball detection**: fine-tuned YOLO11 (`models/ball_person.pt`, classes 0 = person, 1 = ball),
   plus shape, colour and off-pitch checks against boots and spare balls.
3. **Tracking**: a candidate far from the predicted position must persist for several frames before it is accepted.
4. **Pitch mapping**: [PnLCalib](https://github.com/mguti97/PnLCalib) calibrates the camera about once a
   second from pitch landmarks and lines; camera motion is tracked in between. The ball's ground
   point becomes metres on a 105 × 68 m pitch, with the centre spot at (0, 0).
5. **In the air**: a lofted ball (gravity-shaped path on screen) reports no ground position until it lands.
6. **Smoothing**: Kalman filter for position and velocity.
7. **Motor output**:
   - `live_passes.py`: live, one speed profile per pass (starting velocity, then a steady slow-down), announced about 0.2 s after the touch.
   - `passes.py`: for recorded games, one constant velocity per pass.
   - `motor_speed.py`: a speed signal that never rises within a pass.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./setup_pnlcalib.sh        # PnLCalib code + weights (~530 MB, GPL-2.0)
```

## Usage

```bash
# recorded broadcast clip, with an annotated video (yellow circle = sent to the device)
.venv/bin/python track_ball.py match.mp4 --start 600 --duration 60 --out-video annotated.mp4

# tactical-camera footage: the ball is tiny, so enlarge the detector input
.venv/bin/python track_ball.py tactical.mp4 --imgsz 2560

# live: capture the screen and stream JSON to the device over UDP
.venv/bin/python track_ball.py screen --region 0,100,1280,720 --udp 192.168.1.50:5005

# one velocity per pass for a recorded clip
.venv/bin/python passes.py match.ball.csv --out passes.csv
```

`download.py` fetches YouTube matches (720p video only).

## Output

One CSV row per frame, and the same fields per UDP message:

| Field | Meaning |
|---|---|
| `state` | `live`, `coasting`, `air`, `lost`, `paused`, `uncertain`, `unmapped` |
| `field_x_m`, `field_y_m` | position in metres from the centre spot: +x toward the right-hand goal (as seen by the camera), +y toward the camera-side touchline |
| `field_vx_ms`, `field_vy_ms` | Kalman velocity in m/s |
| `live_vx_ms`, `live_vy_ms`, `live_speed_ms` | what the motors should do **now** (live pass profile) |
| `new_pass`, `pass_t0`, `pass_vx0_ms`, `pass_vy0_ms`, `pass_decel_ms2` | the current pass profile: speed(t) = start speed − decel × (t − pass_t0) |
| `motor_speed_ms` (+ `motor_t`) | speed that never rises within a pass, 0.15 s behind |

Empty values mean nothing reliable (ball lost, in the air, close-up): stop or hold the motors.

## Training the detector

- `make_dataset.py`: picks hard and random frames and pre-labels them for Roboflow.
- `split_dataset.py`: splits a dataset between labellers.
- `remap_classes.py`: puts exports back in the model's class order.
- `build_training_set.py`: merges exports, un-stretching Roboflow-resized images.

Then fine-tune from the current model:

```bash
yolo detect train model=models/ball_person.pt data=training_set/data.yaml imgsz=1280 epochs=40 batch=2
```

## Evaluation tools

- `compare.py`: summarises tracker CSVs.
- `eval_calib.py`, `eval_pnl_track.py`: pitch-mapping accuracy.
- `tune_kalman.py`: replays raw positions through filter settings.
- `raw_yolo_test.py`: raw detector check.
- `debug_frame.py`: single-frame keypoints and candidates.
- `sample_coverage.sh`: whole-match coverage estimate.
