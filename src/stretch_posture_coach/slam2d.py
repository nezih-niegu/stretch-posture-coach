"""2-D lidar SLAM for the Stretch base: occupancy grid + correlative scan-to-map matching with a wheel-odometry
prior. It gives a drift-corrected base pose (x, y, theta) in a fixed ``map`` frame; together with the head joint
states and the floor plane (``registration.py``) that pins every RGB-D frame in 6-DOF.

Map frame = the base frame at the first scan (z up, floor at z = 0), same convention as ``scene.py``.
No loop closure: scan-to-map matching keeps a room-scale map consistent; for building-scale maps use ROS 2
(stretch_nav2) or the offline OneSLAM bridge.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    """SE(2) a ⊕ b."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap(at + bt))


def inverse(a):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (-c * x - s * y, s * x - c * y, wrap(-t))


def between(a, b):
    """Pose of b expressed in a: a⁻¹ ⊕ b."""
    return compose(inverse(a), b)


class OccupancyGrid:
    """Log-odds grid, row = y, col = x. Cell (0,0) is at ``origin`` (map metres)."""

    L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 0.9, -4.0, 6.0

    def __init__(self, res: float = 0.05, size_m: float = 40.0, sigma_m: float = 0.08):
        self.res = res
        self.n = int(round(size_m / res))
        self.origin = np.array([-size_m / 2, -size_m / 2])
        self.logodds = np.zeros((self.n, self.n), np.float32)
        self.sigma = sigma_m
        self.field = np.zeros((self.n, self.n), np.float32)
        self.version = 0
        self.scans = 0
        self._bbox = None  # (r0, r1, c0, c1) of observed cells

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.origin) / self.res  # float (col, row)

    def integrate(self, pose, pts_base: np.ndarray, sensor_xy=(0.0, 0.0)) -> None:
        if len(pts_base) < 10:
            return
        x, y, t = pose
        c, s = math.cos(t), math.sin(t)
        R = np.array([[c, -s], [s, c]])
        pm = pts_base @ R.T + [x, y]
        o = np.array(sensor_xy) @ R.T + [x, y]
        cells = np.round(self.to_cell(pm)).astype(np.int32)
        oc = np.round(self.to_cell(o[None, :])).astype(np.int32)[0]
        free = np.zeros((self.n, self.n), np.uint8)
        # free space: the scan polygon seen from the sensor (endpoints ordered by bearing)
        ang = np.arctan2(pm[:, 1] - o[1], pm[:, 0] - o[0])
        order = np.argsort(ang)
        poly = np.vstack([oc[None, :], cells[order]])
        cv2.fillPoly(free, [poly.reshape(-1, 1, 2)], 1)
        for p in cells[order][:: max(1, len(cells) // 180)]:  # thin rays fill gaps of the polygon
            cv2.line(free, tuple(int(v) for v in oc), tuple(int(v) for v in p), 1, 1)
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < self.n) & (cells[:, 1] >= 0) & (cells[:, 1] < self.n)
        cells = cells[ok]
        occ = np.zeros_like(free)
        occ[cells[:, 1], cells[:, 0]] = 1
        free[occ > 0] = 0
        self.logodds[free > 0] += self.L_FREE
        self.logodds[occ > 0] += self.L_OCC
        np.clip(self.logodds, self.L_MIN, self.L_MAX, out=self.logodds)
        r0, r1 = int(min(poly[:, 1].min(), oc[1])) - 2, int(max(poly[:, 1].max(), oc[1])) + 3
        c0, c1 = int(min(poly[:, 0].min(), oc[0])) - 2, int(max(poly[:, 0].max(), oc[0])) + 3
        if self._bbox is None:
            self._bbox = [r0, r1, c0, c1]
        else:
            b = self._bbox
            self._bbox = [min(b[0], r0), max(b[1], r1), min(b[2], c0), max(b[3], c1)]
        self._bbox = [
            max(0, self._bbox[0]),
            min(self.n, self._bbox[1]),
            max(0, self._bbox[2]),
            min(self.n, self._bbox[3]),
        ]
        self.scans += 1
        self.version += 1
        self._update_field()

    def occupied(self) -> np.ndarray:
        return self.logodds > 0.6

    def _update_field(self):
        occ = self.occupied()
        if not occ.any():
            self.field[:] = 0
            return
        dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * self.res
        self.field = np.exp(-(dist**2) / (2 * self.sigma**2)).astype(np.float32)

    def image(self) -> tuple[np.ndarray, dict]:
        """Cropped uint8 image (0 = occupied, 128 = unknown, 255 = free; row 0 = min y) + its placement."""
        if self._bbox is None:
            return np.zeros((1, 1), np.uint8), {"origin": [0, 0], "res": self.res, "w": 1, "h": 1}
        r0, r1, c0, c1 = self._bbox
        lo = self.logodds[r0:r1, c0:c1]
        img = np.full(lo.shape, 128, np.uint8)
        img[lo < -0.6] = 255
        img[lo > 0.6] = 0
        org = self.origin + np.array([c0, r0]) * self.res
        return img, {
            "origin": [float(org[0]), float(org[1])],
            "res": self.res,
            "w": img.shape[1],
            "h": img.shape[0],
        }


@dataclass
class MatchResult:
    pose: tuple
    score: float  # mean likelihood of the scan points (0..1)
    ok: bool


def match(
    grid: OccupancyGrid,
    pts_base: np.ndarray,
    prior,
    stages=((0.30, 0.05, 10.0, 2.0), (0.06, 0.01, 2.0, 0.4)),
    min_score: float = 0.35,
    max_points: int = 400,
) -> MatchResult:
    """Coarse-to-fine exhaustive search around ``prior`` maximizing the likelihood field."""
    if len(pts_base) > max_points:
        pts_base = pts_base[np.linspace(0, len(pts_base) - 1, max_points).astype(int)]
    n = grid.n
    best = (prior, -1.0)
    cx, cy, ct = prior
    for win, step, win_deg, step_deg in stages:
        offs = np.arange(-win, win + 1e-9, step)
        ths = ct + np.radians(np.arange(-win_deg, win_deg + 1e-9, step_deg))
        top = (best[0], -1.0)
        for th in ths:
            c, s = math.cos(th), math.sin(th)
            p = pts_base @ np.array([[c, s], [-s, c]]) + [cx, cy]
            q = grid.to_cell(p)
            ix = np.rint(q[None, :, 0] + offs[:, None] / grid.res).astype(np.int32)  # (nx, N)
            iy = np.rint(q[None, :, 1] + offs[:, None] / grid.res).astype(np.int32)  # (ny, N)
            np.clip(ix, 0, n - 1, out=ix)
            np.clip(iy, 0, n - 1, out=iy)
            sc = grid.field[iy[:, None, :], ix[None, :, :]].mean(axis=2)  # (ny, nx)
            # weak prior: prefer the odometry prediction when the scan is ambiguous (corridors)
            d2 = (offs[:, None] + cy - prior[1]) ** 2 + (offs[None, :] + cx - prior[0]) ** 2
            sc = sc - 0.02 * d2 - 0.01 * wrap(th - prior[2]) ** 2
            k = np.unravel_index(np.argmax(sc), sc.shape)
            if sc[k] > top[1]:
                top = ((cx + offs[k[1]], cy + offs[k[0]], wrap(th)), float(sc[k]))
        best = top
        cx, cy, ct = best[0]
    return MatchResult(best[0], best[1], best[1] >= min_score)


@dataclass
class SlamState:
    pose: tuple = (0.0, 0.0, 0.0)  # map <- base at the last scan
    odom_at_scan: tuple | None = None
    score: float = 0.0
    matched: int = 0
    rejected: int = 0
    last_scan_t: float = 0.0
    history: list = field(default_factory=list)


class Localizer:
    """Fuses wheel odometry (between scans) with scan-to-map matching (at each scan).

    ``pose(odom_now)`` is the current base pose in the map frame, from any thread."""

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.c = c
        self.grid = OccupancyGrid(float(c.get("map_res_m", 0.05)), float(c.get("map_size_m", 40.0)))
        self.st = SlamState()
        self.lock = threading.Lock()
        self.mount = (float(c.get("x", 0.0)), float(c.get("y", 0.0)))
        self.yaw_offset = float(c.get("yaw_offset_deg", 180.0))
        self.clockwise = bool(c.get("clockwise", True))
        self.min_range = float(c.get("min_range_m", 0.25))
        self.max_range = float(c.get("max_range_m", 10.0))
        self.update_dist = float(c.get("map_update_m", 0.10))
        self.update_rot = math.radians(float(c.get("map_update_deg", 5.0)))
        self._last_map_pose = None
        self.latest_points: np.ndarray = np.zeros((0, 2))  # last scan, base frame
        self.enabled = False

    def scan_points(self, scan) -> np.ndarray:
        from .lidar import to_base

        return to_base(scan, self.yaw_offset, self.clockwise, *self.mount, self.min_range, self.max_range)

    def on_scan(self, scan, odom) -> MatchResult | None:
        """Process one revolution. ``odom`` = wheel odometry (x, y, theta) at about the scan time, or None."""
        pts = self.scan_points(scan)
        self.enabled = True
        with self.lock:
            st = self.st
            if st.odom_at_scan is not None and odom is not None:
                prior = compose(st.pose, between(st.odom_at_scan, odom))
            else:
                prior = st.pose
        self.latest_points = pts
        if len(pts) < 30:
            return None
        if self.grid.scans == 0:  # first scan defines the map frame
            res = MatchResult((0.0, 0.0, 0.0) if odom is None else prior, 1.0, True)
        else:
            res = match(self.grid, pts, prior, min_score=float(self.c.get("min_score", 0.35)))
        with self.lock:
            st = self.st
            if res.ok:
                st.pose, st.score = res.pose, res.score
                st.matched += 1
            else:  # poor match (e.g. people all around, or a new room): trust odometry, don't corrupt the map
                st.pose, st.score = prior, res.score
                st.rejected += 1
            st.odom_at_scan = odom
            st.last_scan_t = scan.t
            st.history.append((scan.t, *st.pose))
            if len(st.history) > 20000:
                del st.history[:5000]
        lp = self._last_map_pose
        moved = (
            lp is None
            or math.hypot(res.pose[0] - lp[0], res.pose[1] - lp[1]) > self.update_dist
            or abs(wrap(res.pose[2] - lp[2])) > self.update_rot
            or self.grid.scans < 5
        )
        if res.ok and moved:
            self.grid.integrate(res.pose, pts, self.mount)
            self._last_map_pose = res.pose
        return res

    def pose(self, odom_now=None) -> tuple:
        with self.lock:
            st = self.st
            if st.odom_at_scan is None or odom_now is None:
                return st.pose
            return compose(st.pose, between(st.odom_at_scan, odom_now))

    def status(self) -> dict:
        st = self.st
        return {
            "enabled": self.enabled,
            "matched": st.matched,
            "rejected": st.rejected,
            "score": round(st.score, 2),
            "map_scans": self.grid.scans,
        }
