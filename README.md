# Soccer ball tracker for a haptic device

Tracks the ball in a recorded soccer clip (TV broadcast or tactical camera) and turns it into
positions and pass velocities for haptic motors, so visually impaired fans can feel where the ball
is and how it moves. The clip is pre-processed once; for the demo, `demo.py` plays it with sound
and sends the matching data to the ESP32.

## Pipeline

1. **Scene filter**: pauses on replays, close-ups, crowd shots and promo frames.
2. **Ball detection**: fine-tuned YOLO11 (`models/ball_person_v2.pt`, classes 0 = person, 1 = ball),
   plus shape, colour and off-pitch checks against boots and spare balls.
3. **Tracking**: a candidate far from the predicted position must persist for several frames before it is accepted.
4. **Pitch mapping**: [PnLCalib](https://github.com/mguti97/PnLCalib) calibrates the camera about once a
   second from pitch landmarks and lines; camera motion is tracked in between. The ball's ground
   point becomes metres on a 105 × 68 m pitch, with the centre spot at (0, 0).
5. **In the air**: a lofted ball (gravity-shaped path on screen) reports no ground position until it lands.
6. **Smoothing**: Kalman filter for position and velocity.
7. **Passes**: the ball path is split into passes (`balltrack/passes.py`) and each pass gets one
   constant velocity, so the motors hold a steady speed for the whole pass.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./setup_pnlcalib.sh        # PnLCalib code + weights (~530 MB, GPL-2.0)
```

## Usage

**1. Get a clip** (or use your own mp4 with sound):
```bash
.venv/bin/python -m balltrack.download "https://www.youtube.com/watch?v=..." --from 6:00 --to 7:30
```

**2. Pre-process it** (once; writes `<clip>.ball.csv` and `<clip>.timeline.csv`):
```bash
.venv/bin/python track_ball.py videos/clip.mp4 --imgsz 1920
```
Use `--imgsz 1920` for 1080p broadcasts and `--imgsz 2560` for tactical-camera footage (tiny ball);
add `--out-video annotated.mp4` to check the tracking (yellow circle = data sent).

**3. Demo**: play the clip with sound and send the data in sync:
```bash
.venv/bin/python demo.py videos/clip.mp4 --udp 192.168.1.50:5005          # ESP32 over Wi-Fi
.venv/bin/python demo.py videos/clip.mp4                                   # ESP32 plugged in by USB
```
It finds the ESP32's USB port automatically; use `--serial PORT` to pick one, or `--serial none` to just play and
print what it would send. `q` stops it.

## Layout

| Path | Contents |
|---|---|
| `track_ball.py` | command line and the per-frame loop |
| `demo.py` | demo player: video + sound, data to the ESP32 |
| `esp32/` | ESP32 sketches: `ball_tilt_position` (ball data to stepper positions) and simple receivers for testing the link |
| `demo_data/` | tracker output for the demo clip: what `demo.py` sends, and full per-frame detail |
| `notes/` | whiteboard photos of our planning and wiring |
| `balltrack/` | the pipeline, one module per stage: `scene`, `detection`, `selection`, `calibration`, `pitch`, `flight`, `kalman`, `passes`, `sources`, `overlay`, `download` |
| `training/` | building datasets for the detector |
| `tools/` | evaluation and debugging scripts |

Run scripts from the repository root, e.g. `python -m tools.compare results/*.csv`.

## The ESP32

The ESP32 is the device's controller: it turns the ball data from the laptop into movement you can feel.

- **Link:** it's plugged into the laptop by USB and receives `demo.py`'s messages as plain serial
  text at 115200 baud.
- **Motors:** it drives two stepper motors (through two stepper drivers) that tilt a
  pitch-shaped board under the user's hands, up to 20° along the length of the pitch and 15° across it.
- **Tracking mode:** the ball's position is mapped to a tilt on each axis, so the board leans
  toward where the ball is on the field.
- **Kick mode** (currently switched off in the sketch): when a new pass starts faster than 2 m/s, the board jerks in the direction of the
  pass for a fixed number of steps, then goes back to tracking.
- **Planned:** four vibration motors (switched by MOSFETs) for extra haptic cues, a centring
  button, a score button and an IMU to auto-level the board. The wiring is in `notes/IMG_0075.jpg`.

`esp32/ball_tilt_position` computes the target step position for each motor and prints it on the
USB serial port at 115200; the motor driver code is added separately.

## Data sent to the ESP32

`demo.py` sends one text line per message (20 per second by default):

```
x_pos,y_pos,x_vel,y_vel
-8.58,10.65,-1.37,9.13
```

| Field | Meaning |
|---|---|
| `x_pos`, `y_pos` | metres from the centre spot: +x toward the right-hand goal (as seen by the camera), +y toward the near touchline |
| `x_vel`, `y_vel` | m/s; one constant velocity for each pass |

Nothing is sent while there is no ball data (replay, close-up, ball lost or in the air), so the ESP32 holds its last position.

The same values are in `<clip>.timeline.csv`; `<clip>.ball.csv` has the full per-frame detail.

`demo_data/` has both files for our demo clip, [Argentina vs France](https://www.youtube.com/watch?v=RgqKdplLIk4) from 34:57 to 37:02
(video time 0 = match time 34:57). The video itself isn't in the repo; `balltrack.download` can fetch it.

## Training the detector

- `training.make_dataset`: picks hard and random frames and pre-labels them for Roboflow.
- `training.split_dataset`: splits a dataset between labellers.
- `training.remap_classes`: puts exports back in the model's class order.
- `training.build_training_set`: merges exports, un-stretching Roboflow-resized images.

Then fine-tune from the current model:

```bash
yolo detect train model=models/ball_person.pt data=training_set/data.yaml imgsz=1280 epochs=40 batch=2
```

## Evaluation tools

- `tools.compare`: summarises tracker CSVs.
- `tools.eval_calib`, `tools.eval_pnl_track`: pitch-mapping accuracy.
- `tools.tune_kalman`: replays raw positions through filter settings.
- `tools.raw_yolo_test`: raw detector check.
- `tools.debug_frame`: single-frame keypoints and candidates.
- `tools/sample_coverage.sh`: whole-match coverage estimate.
