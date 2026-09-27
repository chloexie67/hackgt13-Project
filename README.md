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
.venv/bin/python demo.py videos/clip.mp4 --serial /dev/cu.usbserial-0001   # USB or Bluetooth serial
```
Without `--udp`/`--serial` it just plays and prints what it would send. `q` stops it.

## Layout

| Path | Contents |
|---|---|
| `track_ball.py` | command line and the per-frame loop |
| `demo.py` | demo player: video + sound, data to the ESP32 |
| `balltrack/` | the pipeline, one module per stage: `scene`, `detection`, `selection`, `calibration`, `pitch`, `flight`, `kalman`, `passes`, `sources`, `overlay`, `download` |
| `training/` | building datasets for the detector |
| `tools/` | evaluation and debugging scripts |

Run scripts from the repository root, e.g. `python -m tools.compare results/*.csv`.

## Data sent to the ESP32

`demo.py` sends one text line per message (20 per second by default):

```
t,valid,x,y,vx,vy
7.14,1,-8.58,10.65,-1.37,9.13
```

| Field | Meaning |
|---|---|
| `t` | video time in seconds |
| `valid` | 1 = ball position known; 0 = no data (replay, close-up, ball lost or in the air): hold or stop the motors |
| `x`, `y` | metres from the centre spot: +x toward the right-hand goal (as seen by the camera), +y toward the near touchline |
| `vx`, `vy` | m/s; one constant velocity for each pass |

The same values are in `<clip>.timeline.csv`; `<clip>.ball.csv` has the full per-frame detail.

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
