"""Reading a video file frame by frame. Yields (frame_number, time_s, frame)."""
import cv2


class FileSource:
    def __init__(self, path, start=0.0, duration=None, stride=1):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.start_idx = int(start * self.fps)
        self.end_idx = self.start_idx + int(duration * self.fps) if duration else None
        self.stride = stride

    def __iter__(self):
        idx = self.start_idx
        if idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        next_idx = idx
        while self.end_idx is None or next_idx < self.end_idx:
            while idx < next_idx:
                if not self.cap.grab():
                    return
                idx += 1
            ok, frame = self.cap.read()
            if not ok:
                return
            yield idx, idx / self.fps, frame
            idx += 1
            next_idx = (idx + self.stride - 1) // self.stride * self.stride

    def release(self):
        self.cap.release()
