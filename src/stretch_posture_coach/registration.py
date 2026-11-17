"""6-DOF registration of the head camera in the map frame.

x, y, yaw        lidar SLAM (``slam2d.Localizer``), wheel odometry in between scans / without a lidar
pan, tilt        head joint encoders (``scene.HeadKinematics``)
roll, pitch, z   the floor plane seen by the D435i: a small constant tilt-encoder offset, camera roll on its
                 mount, or a wrong camera height make every keyframe land at a slightly different height and
                 slope - the "stacked floors" and slanted ceiling of an unregistered map. ``FloorAligner``
                 measures them from the floor and corrects all poses.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from .scene import HeadKinematics, make_T, map_T_base, rot_z


def rot_z4(a: float) -> np.ndarray:
    return make_T(rot_z(a), [0, 0, 0])


def fit_plane_ransac(pts: np.ndarray, iters: int = 120, tol: float = 0.02, rng=None):
    """(unit normal n, offset d) with n·p + d = 0, and the inlier mask. Least-squares refit on inliers."""
    rng = rng or np.random.default_rng(0)
    n_pts = len(pts)
    if n_pts < 3:
        return None
    best_inl, best = None, None
    for _ in range(iters):
        a, b, c = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ a
        inl = np.abs(pts @ n + d) < tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (n, d)
    if best is None:
        return None
    q = pts[best_inl]
    cen = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
    n = vt[2]
    if n @ best[0] < 0:
        n = -n
    d = -n @ cen
    return n, d, np.abs(pts @ n + d) < tol


class FloorAligner:
    """Online estimate of (tilt offset, camera roll, camera height offset) from floor planes."""

    def __init__(self, kin: HeadKinematics, cfg: dict | None = None):
        c = cfg or {}
        self.kin = kin
        self.enabled = bool(c.get("floor_align", True))
        self.alpha = float(c.get("floor_alpha", 0.15))
        self.max_tilt = math.radians(float(c.get("floor_max_tilt_deg", 10.0)))
        self.max_dh = float(c.get("floor_max_dh_m", 0.25))
        self.dtilt = math.radians(float(c.get("tilt_offset_deg", 0.0)))
        self.droll = math.radians(float(c.get("roll_offset_deg", 0.0)))
        self.dh = float(c.get("height_offset_m", 0.0))
        self.n = 0
        self.last_inliers = 0
        self.lock = threading.Lock()
        self._rng = np.random.default_rng(1)

    def base_T_cam(self, pan: float, tilt: float) -> np.ndarray:
        with self.lock:
            dt, dr, dh = self.dtilt, self.droll, self.dh
        T = self.kin.base_T_cam(pan, tilt + dt) @ rot_z4(dr)
        T[2, 3] += dh
        return T

    def observe(self, pts_opt: np.ndarray, pan: float, tilt: float) -> dict | None:
        """``pts_opt``: optical-frame points (N,3) of one depth frame (people already removed).
        Updates the correction if a floor plane is visible. Returns the measurement or None."""
        if not self.enabled or len(pts_opt) < 300:
            return None
        T = self.base_T_cam(pan, tilt)
        pb = pts_opt @ T[:3, :3].T + T[:3, 3]
        horiz = np.hypot(pb[:, 0], pb[:, 1])
        cand = (np.abs(pb[:, 2]) < 0.2) & (horiz > 0.6) & (horiz < 4.5)  # not the robot's own base
        if cand.sum() < 300:
            return None
        q = pts_opt[cand]
        if len(q) > 3000:
            q = q[self._rng.choice(len(q), 3000, replace=False)]
        fit = fit_plane_ransac(q, rng=self._rng)
        if fit is None:
            return None
        n, d, inl = fit
        up_exp = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # expected up, optical frame
        if n @ up_exp < 0:
            n, d = -n, -d
        if n @ up_exp < math.cos(math.radians(20)) or inl.sum() < 0.5 * len(q):
            return None
        # n = observed up in the optical frame; camera height above the floor = d (n·p + d = 0 on the floor)
        tilt_obs = math.atan2(n[2], math.hypot(n[0], n[1]))
        roll_obs = math.atan2(-n[0], -n[1])
        h_obs = float(d)
        dtilt = tilt_obs - tilt
        z_kin = self.kin.base_T_cam(pan, tilt + dtilt)[2, 3]
        dh = h_obs - z_kin
        if abs(dtilt) > self.max_tilt or abs(roll_obs) > self.max_tilt or abs(dh) > self.max_dh:
            return None
        with self.lock:
            a = 1.0 if self.n == 0 else self.alpha
            self.dtilt += a * (dtilt - self.dtilt)
            self.droll += a * (roll_obs - self.droll)
            self.dh += a * (dh - self.dh)
            self.n += 1
        self.last_inliers = int(inl.sum())
        return {"dtilt_deg": math.degrees(dtilt), "roll_deg": math.degrees(roll_obs), "dh_m": dh}

    def status(self) -> dict:
        return {
            "floor_fits": self.n,
            "tilt_offset_deg": round(math.degrees(self.dtilt), 2),
            "roll_deg": round(math.degrees(self.droll), 2),
            "height_offset_m": round(self.dh, 3),
        }


def map_T_cam(base_pose, pan: float, tilt: float, aligner: FloorAligner) -> np.ndarray:
    x, y, th = base_pose
    return map_T_base(x, y, th) @ aligner.base_T_cam(pan, tilt)
