"""Frame sources: video files, the screen, or a camera. Each yields (frame_number, time_s, frame)."""
import time

import cv2
import numpy as np

class FileSource:
    """A video file. With realtime=True it behaves like a live feed: frames are skipped
    whenever processing falls behind the video clock, exactly as a live stream would."""

    def __init__(self, path, start=0.0, duration=None, stride=1, realtime=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.start_idx = int(start * self.fps)
        self.end_idx = self.start_idx + int(duration * self.fps) if duration else None
        self.stride, self.realtime = stride, realtime
        self.dropped = 0

    def __iter__(self):
        idx = self.start_idx
        if idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        t0 = time.time()
        next_idx = idx
        while self.end_idx is None or next_idx < self.end_idx:
            if self.realtime:
                # the frame "on air" now; never process ahead of it, skip anything behind it
                live_idx = self.start_idx + int((time.time() - t0) * self.fps)
                if next_idx > live_idx:
                    time.sleep((next_idx - live_idx) / self.fps)
                else:
                    next_idx = max(next_idx, live_idx)
            while idx < next_idx:  # skip without the cost of converting the frame
                if not self.cap.grab():
                    return
                idx += 1
                if self.realtime and idx % self.stride == 0:
                    self.dropped += 1
            ok, frame = self.cap.read()
            if not ok:
                return
            yield idx, idx / self.fps, frame
            idx += 1
            next_idx = (idx + self.stride - 1) // self.stride * self.stride

    def release(self):
        self.cap.release()


class LiveSource:
    """Screen region or camera (e.g. OBS Virtual Camera), captured on a background thread
    so capture never waits on inference; the tracker always gets the newest frame."""

    def __init__(self, spec, region=None, width=1280):
        import threading

        self.spec, self.region, self.width = spec, region, width
        self.latest = None
        self.cond = threading.Condition()
        self.running = True
        self.fps = 30.0
        self.total = 0
        self.dropped = 0
        self.captured = 0
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        with self.cond:  # wait for the first frame so the size is known
            self.cond.wait_for(lambda: self.latest is not None or self.error, timeout=10)
        if self.error or self.latest is None:
            raise SystemExit(f"Could not capture from {spec}: {self.error or 'no frames within 10 s'}")
        h, w = self.latest[2].shape[:2]
        self.size = (w, h)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self):
        try:
            if self.spec == "screen":
                import mss

                with mss.mss() as sct:
                    mon = sct.monitors[1]
                    if self.region:
                        x, y, w, h = self.region
                        mon = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
                    while self.running:
                        frame = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
                        self._publish(self._resize(np.ascontiguousarray(frame)))
            else:
                cap = cv2.VideoCapture(int(self.spec))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("camera returned no frame")
                    self._publish(self._resize(frame))
                cap.release()
        except Exception as e:  # surfaced to the main thread
            self.error = e
            with self.cond:
                self.cond.notify_all()

    def _publish(self, frame):
        with self.cond:
            if self.latest is not None and not self.latest[3]:
                self.dropped += 1  # previous frame was never processed
            self.captured += 1
            self.latest = [self.captured, time.time(), frame, False]
            self.cond.notify_all()

    def __iter__(self):
        t0 = None
        while True:
            with self.cond:
                self.cond.wait_for(lambda: (self.latest is not None and not self.latest[3]) or self.error)
                if self.error:
                    raise SystemExit(f"Capture failed: {self.error}")
                self.latest[3] = True
                n, t, frame, _ = self.latest
            t0 = t if t0 is None else t0
            yield n, t - t0, frame

    def release(self):
        self.running = False
