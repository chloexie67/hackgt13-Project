"""Camera calibration: PnLCalib (GPL-2.0, github.com/mguti97/PnLCalib) about once a second,
camera-motion tracking in between, and a landmark-only fallback."""
import sys
from pathlib import Path

import cv2
import numpy as np

from .pitch import PITCH_VERTICES, line_mask, line_score

PNLCALIB_DIR = Path(__file__).resolve().parent.parent / "third_party" / "PnLCalib"



class FieldCalibrator:
    def __init__(self, device="mps", weights_kp=None, weights_lines=None, pnl_refine=True,
                 kp_threshold=0.3434, line_threshold=0.7867):
        import torch
        import torchvision.transforms as T
        import yaml

        sys.path.insert(0, str(PNLCALIB_DIR))
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l

        self.torch, self.device = torch, device
        self.pnl_refine, self.kp_threshold, self.line_threshold = pnl_refine, kp_threshold, line_threshold
        weights_kp = weights_kp or PNLCALIB_DIR / "weights" / "SV_kp"
        weights_lines = weights_lines or PNLCALIB_DIR / "weights" / "SV_lines"

        cfg = yaml.safe_load(open(PNLCALIB_DIR / "config" / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(PNLCALIB_DIR / "config" / "hrnetv2_w48_l.yaml"))
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
            heat, heat_l = self.model(x), self.model_l(x)
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

    def __init__(self, calibrator, interval=1.0, max_stale=4.0, min_line_score=0.4):
        self.cal, self.interval = calibrator, interval
        self.max_stale, self.min_line_score = max_stale, min_line_score
        self.motion = CameraMotion()
        self.calibrations = self.rejected = 0
        self._new_reference()

    def _new_reference(self):
        self.cum = np.eye(3)       # reference-frame pixels -> current-frame pixels
        self.H_ref = None          # reference-frame pixels -> pitch metres
        self.last_calib_t = None   # video time of the last accepted calibration
        self.last_calib_try = -1e9

    def reset(self):
        self.motion.reset()
        self._new_reference()

    def _calibrate(self, frame):
        H = self.cal.homography(frame)
        if H is None:
            return None
        lines = line_mask(frame)
        dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 3)
        score = line_score(H, dist)
        if score is None or score < self.min_line_score:
            return None
        return H

    def update(self, frame, t):
        if not self.motion.step(frame):
            self._new_reference()
            self.motion.step(frame)
        self.cum = self.motion.cum

        wait = self.interval if self.H_ref is not None else 0.25
        if t - self.last_calib_try >= wait:
            self.last_calib_try = t
            H = self._calibrate(frame)
            if H is None:
                self.rejected += 1
            else:
                self.calibrations += 1
                self.H_ref = H @ self.cum
                self.last_calib_t = t

        if self.H_ref is None or t - self.last_calib_t > self.max_stale:
            return None
        return self.H_ref @ np.linalg.inv(self.cum)


class KeypointPitchMapper:

    MIN_INLIERS = 5
    MAX_DISAGREE_M = 4.0

    def __init__(self, weights: str, device: str, kp_conf: float, hold_frames: int, precision: int = 32):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device, self.kp_conf, self.hold_frames = device, kp_conf, hold_frames
        self.precision = precision
        self.H = None
        self.age = 0
        self.grid = None

    def reset(self):
        self.H, self.age = None, 0

    def update(self, frame):
        H = self._fit(frame)
        if H is not None and self.H is not None:
            if self.grid is None:
                h, w = frame.shape[:2]
                gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 4) * w, np.linspace(0.35, 0.95, 3) * h)
                self.grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)[None]
            disagree = np.median(np.linalg.norm(cv2.perspectiveTransform(self.grid, H)[0]
                                                - cv2.perspectiveTransform(self.grid, self.H)[0], axis=1))
            if disagree <= self.MAX_DISAGREE_M:
                H = 0.5 * H / H[2, 2] + 0.5 * self.H / self.H[2, 2]
        if H is not None:
            self.H, self.age = H, 0
        else:
            self.age += 1
            if self.age > self.hold_frames:
                self.H = None
        return self.H

    def hold(self):
        self.age += 1
        if self.age > self.hold_frames:
            self.H = None
        return self.H

    def _fit(self, frame):
        r = self.model.predict(frame, conf=0.3, device=self.device, quantize=self.precision,
                               verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        xy = r.keypoints.xy[0].cpu().numpy()
        conf = (r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None
                else np.ones(len(xy)))
        n = min(len(xy), len(PITCH_VERTICES))
        mask = (conf[:n] > self.kp_conf) & (xy[:n, 0] > 1) & (xy[:n, 1] > 1)
        if mask.sum() < self.MIN_INLIERS:
            return None
        H, inliers = cv2.findHomography(xy[:n][mask], PITCH_VERTICES[:n][mask], cv2.RANSAC, 1.5)
        if H is None or inliers is None or inliers.sum() < self.MIN_INLIERS:
            return None
        return H
