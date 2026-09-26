"""Pitch registration with PnLCalib (github.com/mguti97/PnLCalib, GPL-2.0).

Two HRNet models find pitch keypoints and line endpoints; a camera is fitted to them and
refined against the detected lines (the "PnL refinement"). We only need where the grass
plane lands in the image, so the 3x4 camera matrix is reduced to an image -> pitch
homography in this project's coordinates (x 0..105 goal line to goal line, y 0..68).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent / "third_party" / "PnLCalib"

# Apple's Metal backend crashes if two threads encode GPU work at once, so everything that
# runs a model on the GPU while calibration may be running on another thread takes this.
GPU_LOCK = __import__("threading").Lock()


class FieldCalibrator:
    def __init__(self, device="mps", weights_kp=None, weights_lines=None, pnl_refine=True,
                 kp_threshold=0.3434, line_threshold=0.7867):
        import torch
        import torchvision.transforms as T
        import yaml

        sys.path.insert(0, str(REPO))
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l

        self.torch, self.device = torch, device
        if device == "cpu":
            torch.set_num_threads(4)  # leave cores for the main loop
        self.pnl_refine, self.kp_threshold, self.line_threshold = pnl_refine, kp_threshold, line_threshold
        weights_kp = weights_kp or REPO / "weights" / "SV_kp"
        weights_lines = weights_lines or REPO / "weights" / "SV_lines"

        cfg = yaml.safe_load(open(REPO / "config" / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(REPO / "config" / "hrnetv2_w48_l.yaml"))
        self.model = get_cls_net(cfg)
        self.model.load_state_dict(torch.load(weights_kp, map_location="cpu"))
        self.model.to(device).eval()
        self.model_l = get_cls_net_l(cfg_l)
        self.model_l.load_state_dict(torch.load(weights_lines, map_location="cpu"))
        self.model_l.to(device).eval()
        self.resize = T.Resize((540, 960))
        self.cam = None
        self.last_params = None

    def camera(self, frame):
        """Full camera parameters for a BGR frame, or None if calibration failed."""
        import torchvision.transforms.functional as F
        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (complete_keypoints, coords_to_dict,
                                         get_keypoints_from_heatmap_batch_maxpool,
                                         get_keypoints_from_heatmap_batch_maxpool_l)

        h0, w0 = frame.shape[:2]
        if self.cam is None or (self.cam.image_width, self.cam.image_height) != (w0, h0):
            self.cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)

        x = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float().unsqueeze(0)
        if x.shape[-1] != 960:
            x = self.resize(x)
        x = x.to(self.device)
        _, _, h, w = x.shape
        with self.torch.no_grad():
            if self.device == "cpu":
                heat, heat_l = self.model(x), self.model_l(x)
            else:
                with GPU_LOCK:
                    heat = self.model(x)
                with GPU_LOCK:
                    heat_l = self.model_l(x)
        kp = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(heat[:, :-1].cpu()), threshold=self.kp_threshold)
        ln = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(heat_l[:, :-1].cpu()), threshold=self.line_threshold)
        kp, ln = complete_keypoints(kp[0], ln[0], w=w, h=h, normalize=True)
        self.cam.update(kp, ln)
        self.last_params = self.cam.heuristic_voting(refine_lines=self.pnl_refine)
        return self.last_params

    def homography(self, frame):
        """Image -> pitch homography (metres, this project's axes), or None."""
        params = self.camera(frame)
        if params is None:
            return None
        P = projection_matrix(params)
        # grass plane z=0; PnLCalib's world origin is the centre spot
        G = P[:, [0, 1, 3]] @ np.array([[1, 0, -52.5], [0, 1, -34.0], [0, 0, 1]])
        try:
            return np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return None


def projection_matrix(params):
    c = params["cam_params"]
    K = np.array([[c["x_focal_length"], 0, c["principal_point"][0]],
                  [0, c["y_focal_length"], c["principal_point"][1]],
                  [0, 0, 1]])
    It = np.eye(4)[:-1]
    It[:, -1] = -np.array(c["position_meters"])
    return K @ (np.array(c["rotation_matrix"]) @ It)


class CameraMotion:
    """Image motion of a broadcast camera relative to a reference frame.

    A broadcast camera pans, tilts and zooms from a fixed spot, so any two frames are
    related by one homography for everything in view (grass, lines, stands). Points are
    followed frame to frame with Lucas-Kanade flow (kept only if tracking them back lands
    where they started), but the homography is always fitted from the keyframe where they
    were first detected, so frame-to-frame errors don't compound. When too few points
    survive, the current frame becomes the new keyframe.
    `cum` maps reference-frame pixels to current-frame pixels.
    """

    def __init__(self, scale=0.5, max_pts=600, min_pts=60):
        self.scale, self.max_pts, self.min_pts = scale, max_pts, min_pts
        self.S = np.diag([scale, scale, 1.0])
        self.reset()

    def reset(self):
        self.prev = None
        self.cum = np.eye(3)

    def _detect(self, g):
        pts = cv2.goodFeaturesToTrack(g, self.max_pts, 0.005, 10)
        return None if pts is None else pts.reshape(-1, 2)

    def step(self, frame):
        """Advance to this frame. Returns False if motion was lost (cut, heavy blur)."""
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev, self.key_pts, self.pts, self.key_base = g, self._detect(g), None, self.cum.copy()
            self.pts = self.key_pts
            return True
        if self.pts is None or len(self.pts) < self.min_pts:
            return self._lost(g)
        p0 = self.pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, winSize=(21, 21), maxLevel=3)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - p0).reshape(-1, 2), axis=1) < 0.5)
        if good.sum() < self.min_pts:
            return self._lost(g)
        key_pts, cur = self.key_pts[good], p1.reshape(-1, 2)[good]
        M, inl = cv2.findHomography(key_pts, cur, cv2.RANSAC, 1.0)  # keyframe -> current (half size)
        if M is None or inl.sum() < self.min_pts:
            return self._lost(g)
        inl = inl.ravel().astype(bool)
        self.cum = np.linalg.inv(self.S) @ M @ self.S @ self.key_base
        self.prev = g
        if inl.sum() < 2 * self.min_pts:
            # running low: start a new keyframe here, continuing from the current estimate
            self.key_base, self.key_pts = self.cum.copy(), self._detect(g)
            self.pts = self.key_pts
        else:
            self.key_pts, self.pts = key_pts[inl], cur[inl]
        return True

    def _lost(self, g):
        self.prev, self.cum = None, np.eye(3)
        return False


class PitchTracker:
    """Image -> pitch mapping for every frame: PnLCalib every `interval` seconds, camera
    motion tracking in between.

    Live (async_=True) the calibration runs on a background thread on the newest frame,
    and its result is carried forward to the current frame through the tracked motion.
    Offline (async_=False) it runs inline, so results don't depend on timing.
    A calibration is only accepted if its projected lines land on the painted lines.
    """

    def __init__(self, calibrator, interval=1.0, async_=False, max_stale=4.0, min_line_score=0.4):
        import threading

        self.cal, self.interval, self.async_ = calibrator, interval, async_
        self.max_stale, self.min_line_score = max_stale, min_line_score
        self.motion = CameraMotion()
        self.lock = threading.Lock()
        self.busy = False
        self.generation = 0        # bumped on reset so stale worker results are dropped
        self.calibrations = self.rejected = 0
        self._new_reference()

    def _new_reference(self):
        self.cum = np.eye(3)       # reference-frame pixels -> current-frame pixels
        self.history = {}          # frame number -> (cum at that frame, time)
        self.H_ref = None          # reference-frame pixels -> pitch metres
        self.last_calib_t = None   # video time of the frame the last accepted calibration used
        self.last_submit_t = -1e9
        with self.lock:
            self.pending = None
            self.generation += 1

    def reset(self):
        """After a cut or a close-up: forget everything and recalibrate as soon as possible."""
        self.motion.reset()
        self._new_reference()

    def _calibrate(self, frame):
        import track_ball as tb

        H = self.cal.homography(frame)
        if H is None:
            return None
        lines = tb.line_mask(frame)
        dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 3)
        score = tb.line_score(H, dist)
        if score is None or score < self.min_line_score:
            return None
        return H

    def _worker(self, frame, frame_no, generation):
        try:
            H = self._calibrate(frame)
        except Exception as e:  # never leave the worker marked busy
            import traceback
            traceback.print_exc()
            self.last_error, H = e, None
        with self.lock:
            self.busy = False
            if generation == self.generation:
                self.pending = (frame_no, H)

    def _accept(self, frame_no, H):
        if H is None:
            self.rejected += 1
            return
        if frame_no not in self.history:
            return  # tracking restarted since this frame was calibrated
        cum_k, t_k = self.history[frame_no]
        self.calibrations += 1
        self.H_ref = H @ cum_k     # reference -> image(frame_no) -> pitch
        self.last_calib_t = t_k

    def update(self, frame, frame_no, t):
        import threading

        if not self.motion.step(frame):
            self._new_reference()  # motion lost (cut, heavy blur): old calibrations don't apply
            self.motion.step(frame)  # this frame starts the new reference
        self.cum = self.motion.cum
        self.history[frame_no] = (self.cum.copy(), t)
        if len(self.history) > 600:
            self.history.pop(next(iter(self.history)))

        # recalibrate on schedule, or quickly (every 0.25 s) while there is no mapping
        wait = self.interval if self.H_ref is not None else 0.25
        due = t - self.last_submit_t >= wait
        if self.async_:
            with self.lock:
                result, self.pending = self.pending, None
                start = due and not self.busy
                if start:
                    self.busy = True
                generation = self.generation
            if result is not None:
                self._accept(*result)
            if start:
                self.last_submit_t = t
                threading.Thread(target=self._worker, daemon=True,
                                 args=(frame.copy(), frame_no, generation)).start()
        elif due:
            self.last_submit_t = t
            self._accept(frame_no, self._calibrate(frame))

        if self.H_ref is None or self.last_calib_t is None or t - self.last_calib_t > self.max_stale:
            return None  # no calibration, or only dead reckoning for too long
        return self.H_ref @ np.linalg.inv(self.cum)
