# Development log

How the project developed during the hackathon (Friday 25 to Saturday 26 September 2026). The
repository was created on Saturday evening, so the commit times don't show this; the times
below come from the timestamps of the files and outputs made along the way.

Built with the help of Claude Code (an AI coding assistant), which wrote most of the code;
the team tested the output against match video and trained the detector.

## Friday evening: first tracker (about 20:00 to 21:00)

- Planned the pipeline: video → ball detection → pitch coordinates → data for the motors.
- First `track_ball.py` (20:30): YOLO ball detection, Kalman smoothing, homography to pitch
  metres using a landmark model from roboflow/sports, CSV and UDP output.
- First real test on Portugal–Spain (World Cup 2018): it ran at 3.5 fps, locked onto a
  player's boot, and positions jumped. Fixed by lowering the detector input size, picking the
  model from an 80-frame comparison, and adding a close-up filter.

## Friday late evening: accurate pitch mapping (about 21:00 to 23:30)

- Measured the landmark-only mapping against real line crossings: it was 9–62 px off, up to
  about 6 m. Tried snapping the mapping to the painted lines ourselves; it worked on some
  frames but failed on others.
- Switched to PnLCalib (23:19): 1–6 px error, under 1 m. It is slow (about 0.5 s per frame),
  so it calibrates about once a second and camera motion is tracked in between, with drift
  under 4 px over 2 s.
- In-the-air detection (23:30): a lofted ball follows a gravity curve on screen; no ground
  position is reported until it lands.

## Saturday early morning: fewer wrong picks (about 00:00 to 02:00)

- Frame-by-frame review of annotated video by the team; each error time was reported back.
- Added shape, size and colour checks (neon boots), confirmation before switching to a new
  candidate, a speed cap, and an "uncertain" state for weak picks.
- Built the dataset pipeline (01:08): hard frames from the team's reviews, pre-labelled for Roboflow.

## Saturday midday: custom detector (about 12:50 to 14:00)

- A teammate trained `best.pt` on 289 labelled broadcast frames in Colab.
- Integrated it (12:57). A raw Colab-style check matched; the tracker's old reporting
  threshold was hiding correct detections and was retuned.
- Recognised the FIFA archive promo frame, shown in about 20% of the archive videos, and pause during it.
- Measured whole-match coverage: a position is sent in 24% of broadcast frames, because close-ups
  and replays can't be tracked.

## Saturday afternoon: tactical camera (about 14:00 to 16:30)

- A tactical-camera recording (EURO 2020) gave 53% coverage straight away, with no pauses and
  mapping in 100% of frames. A 2× enlarged detector input found the tiny ball far more often.
- Generated 300 tactical frames; 150 were labelled (15:49).
- Found that the broadcast training images had been stretched to squares by Roboflow's
  resize step. Built a combined training set with them restored to 16:9 (16:09) and
  started fine-tuning on the laptop GPU.

## Saturday evening: velocity for the motors (about 17:00 to 18:00)

- Validated against Real Madrid–Barcelona: after a pass the velocity ramped up instead of
  starting fast. The filter assumed 1.5 m of position noise; the measured noise was 0.1 m.
  Retuning halved the speed error.
- The motors need one clean speed per pass:
  - `motor_speed.py` (17:24): speed that never rises within a pass.
  - `passes.py` (17:42): one velocity per pass for recorded games.
  - `live_passes.py` (17:53): a live pass profile announced about 0.2 s after the touch,
    within the 0.3 s limit.
- Created this repository.
