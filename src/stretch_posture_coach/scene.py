"""Scene understanding: RGB-D keyframe recording, a static 3-D map of the room (people removed), where the
person was, free space around them - and a bridge to OneSLAM (https://github.com/arcadelab/OneSLAM) for
camera-only localization.

Frames
  optical  camera optical frame of the *upright* (rotated) head image: x right, y down, z forward
  head     x forward, y left, z up, origin on the head pan axis at camera height
  base     Stretch base_link: x forward, y left, z up (floor at z = 0)
  map      the base frame at the start of the session (z up, floor at z = 0)
T_A_B maps coordinates from frame B into frame A (4x4).

Two pose sources
  kinematic  head pan/tilt joint states + wheel odometry. Always available on the robot; a reference, NOT
             camera-only (it uses wheel odometry when the base moves).
  oneslam    camera-only trajectory from OneSLAM (run offline on a GPU machine), metric scale recovered from
             the D435i depth, converted to base poses with the head joint states:
             T_map_base(t) = T_map_cam(t) @ inv(T_base_cam(t)).
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ONESLAM_REPO = "https://github.com/arcadelab/OneSLAM"
ONESLAM_COMMIT = "05c1543d79ecb7b1567bab31adc85a16a52c9c2a"  # pinned upstream commit (2025-05-21)

R_HEAD_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical -> head axes


# ------------------------------------------------------------------------------------------ geometry
def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


@dataclass
class HeadKinematics:
    """Approximate Stretch 3 head geometry (good to a few cm, enough for room-scale maps)."""

    camera_height_m: float = 1.3
    pan_axis_x: float = 0.0
    pan_axis_y: float = 0.0
    camera_forward_m: float = 0.05
    forward_pan_rad: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> HeadKinematics:
        s = cfg.get("scene", {})
        return cls(
            float(cfg.get("camera", {}).get("height_m", 1.3)),
            float(s.get("pan_axis_x", 0.0)),
            float(s.get("pan_axis_y", 0.0)),
            float(s.get("camera_forward_m", 0.05)),
            float(cfg.get("follow", {}).get("forward_pan_rad", 0.0)),
        )

    def base_T_cam(self, pan: float, tilt: float) -> np.ndarray:
        """Camera optical frame -> base frame, from the head joint states (tilt < 0 looks down)."""
        R_pt = rot_z(pan - self.forward_pan_rad) @ rot_y(-tilt)
        t = np.array([self.pan_axis_x, self.pan_axis_y, self.camera_height_m]) + R_pt @ [
            self.camera_forward_m,
            0,
            0,
        ]
        return make_T(R_pt @ R_HEAD_OPT, t)


def map_T_base(x: float, y: float, theta: float) -> np.ndarray:
    return make_T(rot_z(theta), [x, y, 0.0])


# ------------------------------------------------------------------------------------------ TUM poses
def quat_from_R(R: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw)"""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def R_from_quat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def write_tum(
    path: str | Path, poses: dict[int, np.ndarray], header: str = "frame_index tx ty tz qx qy qz qw"
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for k in sorted(poses):
            T = poses[k]
            q = quat_from_R(T[:3, :3])
            f.write(
                f"{k} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n"
            )


def read_tum(path: str | Path) -> dict[int, np.ndarray]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 8:
            continue
        idx = int(round(float(parts[0])))
        out[idx] = make_T(R_from_quat([float(v) for v in parts[4:8]]), [float(v) for v in parts[1:4]])
    return out


# ------------------------------------------------------------------------------------------ recording
def person_mask(pose, shape: tuple[int, int], min_vis: float = 0.5, grow: float = 0.2) -> np.ndarray:
    """uint8 mask (255 = person) from the convex hull of the visible landmarks, grown by ``grow`` x body size."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if pose is None:
        return mask
    pts = pose.image[pose.visibility >= min_vis]
    if len(pts) < 3:
        return mask
    px = np.stack([pts[:, 0] * (w - 1), pts[:, 1] * (h - 1)], axis=1).astype(np.int32)
    hull = cv2.convexHull(px)
    cv2.fillConvexPoly(mask, hull, 255)
    size = max(np.ptp(px[:, 0]), np.ptp(px[:, 1]), 4)
    k = max(3, int(size * grow) | 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def torso_point_cam(pose, depth_m: np.ndarray | None, intr, min_vis: float = 0.5) -> list | None:
    """Torso center in optical coordinates (m) from the aligned depth, or None."""
    if pose is None or depth_m is None:
        return None
    idx = [i for i in (11, 12, 23, 24) if pose.visibility[i] >= min_vis]
    if len(idx) < 3:
        return None
    u, v = pose.image[idx].mean(axis=0)
    h, w = depth_m.shape[:2]
    px, py = int(round(u * (w - 1))), int(round(v * (h - 1)))
    if not (0 <= px < w and 0 <= py < h):
        return None
    win = depth_m[max(0, py - 6) : py + 7, max(0, px - 6) : px + 7]
    valid = win[(win > 0.3) & (win < 8.0)]
    if valid.size < 8:
        return None
    z = float(np.median(valid))
    sx, sy = intr.w / w, intr.h / h
    return [
        round((px * sx - intr.cx) / intr.fx * z, 3),
        round((py * sy - intr.cy) / intr.fy * z, 3),
        round(z, 3),
    ]


class SceneRecorder:
    """Writes RGB-D keyframes + joint states to <session>/scene/ on a background thread (bounded queue)."""

    def __init__(self, session_dir: str | Path, cfg: dict):
        self.c = cfg
        self.dir = Path(session_dir) / "scene"
        for sub in ("rgb", "depth", "mask"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.period = 1.0 / max(float(cfg.get("record_hz", 1.0)), 0.05)
        self.decim = max(1, int(cfg.get("depth_decimation", 2)))
        self.max_frames = int(cfg.get("max_frames", 3600))
        self.quality = int(cfg.get("jpeg_quality", 85))
        self.n = 0
        self._last = -1e9
        self._meta = open(self.dir / "frames.jsonl", "a", encoding="utf-8")
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, daemon=True, name="scene-rec")
        self._thread.start()

    def maybe_record(
        self,
        ts: float,
        frame: np.ndarray,
        depth_m: np.ndarray | None,
        intr,
        pan: float | None,
        tilt: float | None,
        base: tuple[float, float, float],
        pose=None,
        posture: str | None = None,
        min_vis: float = 0.5,
        extra: dict | None = None,
    ) -> bool:
        if self.n >= self.max_frames or ts - self._last < self.period:
            return False
        self._last = ts
        h, w = frame.shape[:2]
        dh, dw = h // self.decim, w // self.decim
        depth_small = None
        if depth_m is not None and depth_m.shape[:2] == (h, w):
            depth_small = cv2.resize(depth_m, (dw, dh), interpolation=cv2.INTER_NEAREST)
        meta = {
            "i": self.n,
            "t": round(ts, 3),
            "wall": round(time.time(), 3),
            "pan": pan,
            "tilt": tilt,
            "base": [round(float(v), 4) for v in base],
            "intr": [intr.fx, intr.fy, intr.cx, intr.cy, intr.w, intr.h],
            "depth_decimation": self.decim,
            "has_depth": depth_small is not None,
            "posture": posture,
            "person_cam": torso_point_cam(pose, depth_m, intr, min_vis),
            **(extra or {}),
        }
        mask = person_mask(pose, (dh, dw), min_vis)
        try:
            self._q.put_nowait((self.n, frame, depth_small, mask, meta))
        except queue.Full:
            return False  # disk is slow: skip this keyframe rather than block the camera loop
        self.n += 1
        return True

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            i, frame, depth, mask, meta = item
            try:
                name = f"{i:08d}"
                cv2.imwrite(
                    str(self.dir / "rgb" / f"{name}.jpg"),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
                )
                if depth is not None:
                    mm = np.clip(np.nan_to_num(depth) * 1000.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(self.dir / "depth" / f"{name}.png"), mm)
                if mask.any():
                    cv2.imwrite(str(self.dir / "mask" / f"{name}.png"), mask)
                self._meta.write(json.dumps(meta) + "\n")
                self._meta.flush()
            except Exception as e:  # never break the session over a keyframe
                log.debug("scene keyframe %s failed: %s", i, e)

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=10)
        self._meta.close()


# ------------------------------------------------------------------------------------------ loading
def load_frames(session_dir: str | Path) -> list[dict]:
    p = Path(session_dir) / "scene" / "frames.jsonl"
    if not p.exists():
        return []
    frames = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            frames.append(json.loads(line))
        except ValueError:
            continue
    return sorted(frames, key=lambda f: f["i"])


def kinematic_poses(frames: list[dict], kin: HeadKinematics) -> dict[int, np.ndarray]:
    """T_map_cam for every frame with joint states."""
    out = {}
    for f in frames:
        pan, tilt = f.get("pan"), f.get("tilt")
        x, y, th = (f.get("base") or [0.0, 0.0, 0.0])[:3]
        out[f["i"]] = map_T_base(x, y, th) @ kin.base_T_cam(pan or 0.0, tilt or 0.0)
    return out


def _depth(session_dir: Path, i: int) -> np.ndarray | None:
    p = session_dir / "scene" / "depth" / f"{i:08d}.png"
    if not p.exists():
        return None
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) / 1000.0


def _mask(session_dir: Path, i: int, shape) -> np.ndarray:
    p = session_dir / "scene" / "mask" / f"{i:08d}.png"
    if p.exists():
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape == shape:
            return m > 0
    return np.zeros(shape, bool)


def _depth_intr(f: dict, shape) -> tuple[float, float, float, float]:
    fx, fy, cx, cy, w, h = f["intr"]
    sx, sy = shape[1] / w, shape[0] / h
    return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5


def backproject(
    depth: np.ndarray,
    intr4,
    stride: int = 2,
    keep: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Optical-frame points (N,3) and their pixel indices (N,2: row, col) from a depth image."""
    fx, fy, cx, cy = intr4
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[rows, cols]
    ok = (z > zmin) & (z < zmax)
    if keep is not None:
        ok &= keep[rows, cols]
    r, c, z = rows[ok], cols[ok], z[ok]
    pts = np.stack([(c - cx) / fx * z, (r - cy) / fy * z, z], axis=1)
    return pts, np.stack([r, c], axis=1)


# ------------------------------------------------------------------------------------------ map building
@dataclass
class SceneMap:
    voxel: float
    centers: np.ndarray  # (N,3) map frame
    colors: np.ndarray  # (N,3) uint8 RGB
    hits: np.ndarray  # (N,)


def fuse(
    session_dir: str | Path,
    frames: list[dict],
    poses: dict[int, np.ndarray],
    voxel: float = 0.05,
    stride: int = 2,
    min_hits: int = 2,
    max_frames: int = 600,
    exclude_people: bool = True,
) -> SceneMap:
    """Voxel-fuse the depth keyframes into a colored static map (person pixels excluded)."""
    session_dir = Path(session_dir)
    use = [f for f in frames if f.get("has_depth") and f["i"] in poses]
    if len(use) > max_frames:
        use = [use[int(k)] for k in np.linspace(0, len(use) - 1, max_frames)]
    keys_all, col_all, hits_all = [], [], []
    for f in use:
        d = _depth(session_dir, f["i"])
        if d is None:
            continue
        keep = ~_mask(session_dir, f["i"], d.shape) if exclude_people else None
        pts, rc = backproject(d, _depth_intr(f, d.shape), stride, keep)
        if not len(pts):
            continue
        T = poses[f["i"]]
        pw = pts @ T[:3, :3].T + T[:3, 3]
        rgb = cv2.imread(str(session_dir / "scene" / "rgb" / f"{f['i']:08d}.jpg"))
        if rgb is not None:
            rgb = cv2.resize(rgb, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
            col = rgb[rc[:, 0], rc[:, 1], ::-1]
        else:
            col = np.full((len(pw), 3), 180, np.uint8)
        keys_all.append(_pack(np.floor(pw / voxel).astype(np.int64)))
        hits_all.append(np.ones(len(pw), np.float64))
        col_all.append(col.astype(np.float64))
        if sum(len(a) for a in keys_all) > 3_000_000:  # keep memory bounded
            k, c, h = _reduce(keys_all, col_all, hits_all)
            keys_all, col_all, hits_all = [k], [c], [h]
    if not keys_all:
        return SceneMap(voxel, np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, int))
    keys, csum, hits = _reduce(keys_all, col_all, hits_all)
    good = hits >= min_hits
    keys, csum, hits = keys[good], csum[good], hits[good]
    centers = (_unpack(keys) + 0.5) * voxel
    return SceneMap(voxel, centers, (csum / np.maximum(hits, 1)[:, None]).astype(np.uint8), hits.astype(int))


_OFF = 1 << 20  # voxel index offset (±1M voxels = ±50 km at 5 cm)


def _pack(k: np.ndarray) -> np.ndarray:
    k = k + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def _unpack(key: np.ndarray) -> np.ndarray:
    m = (1 << 21) - 1
    return np.stack([(key >> 42) & m, (key >> 21) & m, key & m], axis=1).astype(np.int64) - _OFF


def _reduce(keys_list, col_list, hits_list):
    """Merge duplicate voxels: (unique keys, summed colors, summed hit counts)."""
    keys = np.concatenate(keys_list)
    cols = np.concatenate(col_list)
    hits_in = np.concatenate(hits_list)
    uniq, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    hits = np.bincount(inv, weights=hits_in, minlength=len(uniq))
    csum = np.stack([np.bincount(inv, weights=cols[:, j], minlength=len(uniq)) for j in range(3)], axis=1)
    return uniq, csum, hits


def carve_people(smap: SceneMap, track: list[dict], radius: float = 0.35) -> SceneMap:
    """Remove voxels in a vertical column around every tracked person position (mask edges, frames where the
    detector missed them). Furniture within ``radius`` of where someone stood is removed too."""
    if not track or not len(smap.centers) or radius <= 0:
        return smap
    P = np.unique(np.round(np.array([[p["x"], p["y"]] for p in track]) / 0.05) * 0.05, axis=0)
    keep = np.ones(len(smap.centers), bool)
    xy, z = smap.centers[:, :2], smap.centers[:, 2]
    for px, py in P:  # person positions are few (deduplicated on a 5 cm grid)
        near = (np.abs(xy[:, 0] - px) < radius) & (np.abs(xy[:, 1] - py) < radius)
        near &= np.hypot(xy[:, 0] - px, xy[:, 1] - py) < radius
        keep &= ~(near & (z > 0.12) & (z < 2.1))
    return SceneMap(smap.voxel, smap.centers[keep], smap.colors[keep], smap.hits[keep])


def person_track(frames: list[dict], poses: dict[int, np.ndarray]) -> list[dict]:
    out = []
    for f in frames:
        pc = f.get("person_cam")
        if pc is None or f["i"] not in poses:
            continue
        p = poses[f["i"]] @ np.array([*pc, 1.0])
        cam = poses[f["i"]][:3, 3]
        out.append(
            {
                "i": f["i"],
                "t": f["t"],
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "posture": f.get("posture"),
                "cam_x": float(cam[0]),
                "cam_y": float(cam[1]),
            }
        )
    return out


def analyze(smap: SceneMap, track: list[dict], cam_xy: np.ndarray, cell: float = 0.05) -> dict:
    """Top-down occupancy + scene statistics."""
    if not len(smap.centers):
        return {"empty": True}
    c = smap.centers
    floor_band = np.abs(c[:, 2]) <= 0.12
    obst = (c[:, 2] > 0.15) & (c[:, 2] < 1.9)
    xy = c[:, :2]
    pts_for_extent = [xy] + ([np.array([[p["x"], p["y"]] for p in track])] if track else []) + [cam_xy]
    allxy = np.concatenate(pts_for_extent)
    lo = np.floor(allxy.min(axis=0) / cell) * cell - 0.5
    hi = np.ceil(allxy.max(axis=0) / cell) * cell + 0.5
    W, H = int(round((hi[0] - lo[0]) / cell)), int(round((hi[1] - lo[1]) / cell))

    # grid rows = +x forward going UP the image, cols = +y left going LEFT -> standard top-down
    def to_rc(p):
        col = ((hi[1] - p[:, 1]) / cell).astype(int)
        row = ((hi[0] - p[:, 0]) / cell).astype(int)
        return np.clip(row, 0, W - 1), np.clip(col, 0, H - 1)

    floor = np.zeros((W, H), bool)
    r, cc = to_rc(xy[floor_band])
    floor[r, cc] = True
    r, cc = to_rc(xy[obst])
    flat = np.full(W * H, -np.inf)
    np.maximum.at(flat, r * H + cc, c[obst, 2])
    height = np.where(np.isfinite(flat), flat, np.nan).reshape(W, H).astype(np.float32)
    occ = ~np.isnan(height)
    free = floor & ~occ
    stats = {
        "voxel_m": smap.voxel,
        "voxels": int(len(c)),
        "observed_area_m2": round(float((free | occ).sum()) * cell * cell, 2),
        "free_floor_m2": round(float(free.sum()) * cell * cell, 2),
        "obstacle_area_m2": round(float(occ.sum()) * cell * cell, 2),
        "extent_m": [round(float(np.ptp(xy[:, 0])), 2), round(float(np.ptp(xy[:, 1])), 2)],
    }
    grid = {"lo": lo.tolist(), "hi": hi.tolist(), "cell": cell, "shape": [W, H]}
    if track:
        P = np.array([[p["x"], p["y"]] for p in track])
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(0)
        walked = float(steps[(steps > 0.08) & (steps < 2.0)].sum())  # ignore jitter and jumps
        # clearance: distance from the person to the nearest obstacle cell
        clear = None
        if occ.any():
            dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * cell
            r, cc = to_rc(P)
            clear = dist[r, cc]
        postures = {}
        for p in track:
            postures[p.get("posture") or "unknown"] = postures.get(p.get("posture") or "unknown", 0) + 1
        stats["person"] = {
            "samples": len(track),
            "walked_m": round(walked, 2),
            "area_used_m2": round(float(len({(int(x // 0.25), int(y // 0.25)) for x, y in P})) * 0.0625, 2),
            "clearance_mean_m": None if clear is None else round(float(np.mean(clear)), 2),
            "clearance_min_m": None if clear is None else round(float(np.min(clear)), 2),
            "posture_samples": postures,
            "distance_to_robot_mean_m": round(
                float(np.mean([math.hypot(p["x"] - p["cam_x"], p["y"] - p["cam_y"]) for p in track])), 2
            ),
        }
    return {"stats": stats, "grid": grid, "floor": floor, "height": height, "occ": occ, "to_rc": to_rc}


def render_topdown(
    an: dict, track: list[dict], cam_xy: np.ndarray, cam_fwd: np.ndarray, path: Path, px_per_cell: int = 4
) -> None:
    W, H = an["grid"]["shape"]
    cell = an["grid"]["cell"]
    img = np.full((W, H, 3), 40, np.uint8)  # unknown: dark
    img[an["floor"]] = (225, 224, 220)  # free floor: light
    hgt = an["height"]
    occ = an["occ"]
    if occ.any():  # obstacles: sequential blue by height (low = light, high = dark)
        tnorm = np.clip(hgt[occ] / 1.9, 0, 1)
        light, dark = np.array([213, 160, 115]), np.array([120, 52, 13])  # BGR of #73a0d5 -> #0d3478
        img[occ] = (light[None, :] * (1 - tnorm[:, None]) + dark[None, :] * tnorm[:, None]).astype(np.uint8)
    img = cv2.resize(img, (H * px_per_cell, W * px_per_cell), interpolation=cv2.INTER_NEAREST)
    to_rc = an["to_rc"]

    def pix(p):
        r, c = to_rc(np.atleast_2d(p))
        return np.stack([c * px_per_cell + px_per_cell // 2, r * px_per_cell + px_per_cell // 2], axis=1)

    if track:  # where the person was: heat + path
        P = np.array([[p["x"], p["y"]] for p in track])
        heat = np.zeros(img.shape[:2], np.float32)
        for x, y in pix(P):
            cv2.circle(heat, (int(x), int(y)), 3 * px_per_cell, 1.0, -1)
        heat = cv2.GaussianBlur(heat, (0, 0), 3 * px_per_cell)
        if heat.max() > 0:
            a = (heat / heat.max())[..., None] * 0.65
            img = (img * (1 - a) + np.array([52, 104, 235])[None, None, :] * a).astype(np.uint8)  # orange
        cv2.polylines(img, [pix(P).astype(np.int32)], False, (20, 70, 200), 1, cv2.LINE_AA)
    if len(cam_xy):  # robot / camera path + viewing direction
        cp = pix(cam_xy).astype(np.int32)
        if len(cp) > 1:
            cv2.polylines(img, [cp], False, (30, 30, 30), 2, cv2.LINE_AA)
        start = tuple(int(v) for v in cp[0])
        cv2.circle(img, start, 6, (30, 30, 30), -1, cv2.LINE_AA)
        tip = pix(cam_xy[0] + 0.6 * cam_fwd[0])[0].astype(int)
        cv2.arrowedLine(img, start, (int(tip[0]), int(tip[1])), (30, 30, 30), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(
            img,
            "Stretch",
            (start[0] + 8, start[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    # 1 m scale bar
    L = int(round(1.0 / cell)) * px_per_cell
    y0 = img.shape[0] - 14
    cv2.line(img, (12, y0), (12 + L, y0), (255, 255, 255), 3)
    cv2.putText(img, "1 m", (12, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def write_ply(path: Path, smap: SceneMap) -> None:
    n = len(smap.centers)
    header = (
        f"ply\nformat binary_little_endian 1.0\ncomment stretch-posture-coach static scene map\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = smap.centers[:, 0], smap.centers[:, 1], smap.centers[:, 2]
    rec["r"], rec["g"], rec["b"] = smap.colors[:, 0], smap.colors[:, 1], smap.colors[:, 2]
    path.write_bytes(header + rec.tobytes())


def write_points_bin(path: Path, smap: SceneMap, cam_xyz: np.ndarray, track: list[dict]) -> None:
    """Compact viewer format: [u32 n][n x (f32 xyz)][n x (u8 rgb)][u32 m][m x f32 xyz cam path][u32 k][k x f32 xyz]."""
    P = np.array([[p["x"], p["y"], p["z"]] for p in track], np.float32).reshape(-1, 3)
    parts = [
        struct.pack("<I", len(smap.centers)),
        smap.centers.astype("<f4").tobytes(),
        smap.colors.astype(np.uint8).tobytes(),
    ]
    pad = (-len(smap.colors) * 3) % 4
    parts.append(b"\0" * pad)
    parts += [
        struct.pack("<I", len(cam_xyz)),
        cam_xyz.astype("<f4").tobytes(),
        struct.pack("<I", len(P)),
        P.astype("<f4").tobytes(),
    ]
    path.write_bytes(b"".join(parts))


def build_scene(session_dir: str | Path, cfg: dict | None = None, source: str = "auto") -> dict | None:
    """Build scene/map.png, scene/scene.json, scene/scene.ply, scene/points.bin. Returns the stats dict."""
    session_dir = Path(session_dir)
    cfg = cfg or {}
    frames = load_frames(session_dir)
    if not frames:
        return None
    scfg = cfg.get("scene", {})
    kin = HeadKinematics.from_cfg(cfg)
    poses = kinematic_poses(frames, kin)
    used = "kinematic"
    os_path = session_dir / "scene" / "oneslam" / "poses_cam_map.txt"
    if source in ("auto", "oneslam") and os_path.exists():
        os_poses = read_tum(os_path)
        if os_poses:
            # frames OneSLAM did not localize (non-keyframes) fall back to the kinematic pose
            poses = {**{k: v for k, v in poses.items() if source == "auto"}, **os_poses}
            used = "oneslam" if source == "oneslam" else "oneslam+kinematic"
    smap = fuse(
        session_dir,
        frames,
        poses,
        float(scfg.get("voxel_m", 0.05)),
        int(scfg.get("map_stride", 2)),
        int(scfg.get("min_hits", 2)),
        int(scfg.get("max_map_frames", 600)),
    )
    track = person_track(frames, poses)
    smap = carve_people(smap, track, float(scfg.get("person_radius_m", 0.35)))
    ks = sorted(poses)
    cam_xyz = np.array([poses[k][:3, 3] for k in ks]).reshape(-1, 3)
    cam_fwd = np.array(
        [poses[k][:3, 2][:2] / max(np.linalg.norm(poses[k][:3, 2][:2]), 1e-6) for k in ks]
    ).reshape(-1, 2)
    out = {
        "source": used,
        "frames": len(frames),
        "frames_with_depth": sum(1 for f in frames if f.get("has_depth")),
    }
    if len(smap.centers):
        an = analyze(smap, track, cam_xyz[:, :2] if len(cam_xyz) else np.zeros((1, 2)))
        out.update(an["stats"])
        render_topdown(an, track, cam_xyz[:, :2], cam_fwd, session_dir / "scene" / "map.png")
        write_ply(session_dir / "scene" / "scene.ply", smap)
        write_points_bin(session_dir / "scene" / "points.bin", smap, cam_xyz, track)
    else:
        out["empty"] = True
    camera_travel = float(np.linalg.norm(np.diff(cam_xyz, axis=0), axis=1).sum()) if len(cam_xyz) > 1 else 0.0
    out["camera_travel_m"] = round(camera_travel, 2)
    om = session_dir / "scene" / "oneslam" / "oneslam.json"
    if om.exists():
        out["oneslam"] = json.loads(om.read_text(encoding="utf-8"))
    (session_dir / "scene" / "scene.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------ OneSLAM bridge
def export_oneslam(
    session_dir: str | Path,
    out_dir: str | Path | None = None,
    with_depths: bool = False,
    cfg: dict | None = None,
) -> Path:
    """Write a OneSLAM data_root: images/NNNNNNNN.jpg + calibration.json (+ depth_mm/, poses_kinematic.txt)."""
    session_dir = Path(session_dir)
    frames = [
        f for f in load_frames(session_dir) if (session_dir / "scene" / "rgb" / f"{f['i']:08d}.jpg").exists()
    ]
    if not frames:
        raise FileNotFoundError(f"no scene keyframes in {session_dir} (record with scene.record: true)")
    out = Path(out_dir) if out_dir else session_dir / "scene" / "oneslam_input"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "depth_mm").mkdir()
    if with_depths:
        (out / "depths").mkdir()
    fx, fy, cx, cy, w, h = frames[0]["intr"]
    hz = 1.0
    if len(frames) > 1:
        dt = np.median(np.diff([f["t"] for f in frames]))
        hz = round(1.0 / dt, 3) if dt > 0 else 1.0
    (out / "calibration.json").write_text(
        json.dumps({"intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, "fps": hz}, indent=2),
        encoding="utf-8",
    )
    index = []
    depths = []
    for new_i, f in enumerate(frames):
        name = f"{new_i:08d}"
        shutil.copyfile(session_dir / "scene" / "rgb" / f"{f['i']:08d}.jpg", out / "images" / f"{name}.jpg")
        d = _depth(session_dir, f["i"])
        if d is not None:
            full = cv2.resize(d, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
            mm = np.clip(full * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out / "depth_mm" / f"{name}.png"), mm)
            if with_depths:
                cv2.imwrite(str(out / "depths" / f"{name}.png"), mm)
            v = d[(d > 0.3) & (d < 8)]
            if v.size:
                depths.append(float(np.median(v)))
        index.append(
            {
                "oneslam_index": new_i,
                "session_frame": f["i"],
                "t": f["t"],
                "pan": f.get("pan"),
                "tilt": f.get("tilt"),
                "base": f.get("base"),
            }
        )
    (out / "frame_index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n", encoding="utf-8")
    kin = HeadKinematics.from_cfg(cfg or {})
    kp = kinematic_poses(frames, kin)
    write_tum(
        out / "poses_kinematic.txt",
        {n: kp[f["i"]] for n, f in enumerate(frames)},
        "frame_index tx ty tz qx qy qz qw  (camera-to-map from joint states + wheel odometry; a reference, NOT ground truth)",
    )
    travel = float(
        sum(
            np.linalg.norm(kp[b["i"]][:3, 3] - kp[a["i"]][:3, 3])
            for a, b in zip(frames, frames[1:], strict=False)
        )
    )
    depth_scale = round(float(np.median(depths)), 3) if depths else 2.5
    notes = {
        "frames": len(frames),
        "fps": hz,
        "image_size": [int(w), int(h)],
        "camera_travel_m": round(travel, 2),
        "suggested_args": f"--depth_scale {depth_scale}",
        "warning": (
            None
            if travel >= 0.3
            else "The camera barely translated (head rotation only). Monocular SLAM cannot recover geometry from "
            "pure rotation: record a session where the base moves slowly (e.g. `posture-coach follow` or "
            "driving), with the head fixed (-s robot.head_tracking=false), and record_hz >= 10."
        ),
        "oneslam_commit": ONESLAM_COMMIT,
    }
    (out / "stretch_export.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return out


def _scale_residuals(
    src_pts: np.ndarray, R_ij: np.ndarray, t_ij: np.ndarray, depth_i: np.ndarray, intr_i, scales: np.ndarray
) -> np.ndarray:
    """Median relative depth error when frame-j points are moved into frame i with translation s * t_ij."""
    fx, fy, cx, cy = intr_i
    h, w = depth_i.shape
    rot = src_pts @ R_ij.T
    res = np.full(len(scales), np.inf)
    for k, s in enumerate(scales):
        p = rot + s * t_ij
        z = p[:, 2]
        ok = z > 0.2
        u = np.round(p[ok, 0] / z[ok] * fx + cx).astype(int)
        v = np.round(p[ok, 1] / z[ok] * fy + cy).astype(int)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inb.sum() < 50:
            continue
        zm = depth_i[v[inb], u[inb]]
        zp = z[ok][inb]
        valid = (zm > 0.3) & (zm < 8)
        if valid.sum() < 50:
            continue
        res[k] = float(np.median(np.abs(zp[valid] - zm[valid]) / zm[valid]))
    return res


def estimate_metric_scale(
    session_dir: Path, frames_by_idx: dict[int, dict], os_poses: dict[int, np.ndarray], gaps=(2, 4, 8)
) -> dict:
    """Camera-only metric scale for OneSLAM's (scale-free) trajectory, from depth reprojection consistency."""
    keys = sorted(os_poses)
    coarse = 10 ** np.arange(-4, 4.001, 0.05)
    found = []
    for gap in gaps:
        for a, b in zip(keys, keys[gap:], strict=False):
            fa, fb = frames_by_idx.get(a), frames_by_idx.get(b)
            if not fa or not fb:
                continue
            da, db = _depth(session_dir, fa["i"]), _depth(session_dir, fb["i"])
            if da is None or db is None:
                continue
            T_ij = inv_T(os_poses[a]) @ os_poses[b]  # cam b -> cam a (unscaled translation)
            if np.linalg.norm(T_ij[:3, 3]) < 1e-9:
                continue
            pts, _ = backproject(db, _depth_intr(fb, db.shape), 4, ~_mask(session_dir, fb["i"], db.shape))
            if len(pts) < 200:
                continue
            ia = _depth_intr(fa, da.shape)
            res = _scale_residuals(pts, T_ij[:3, :3], T_ij[:3, 3], da, ia, coarse)
            if not np.isfinite(res).any():
                continue
            k = int(np.nanargmin(res))
            fine = coarse[k] * 10 ** np.linspace(-0.05, 0.05, 21)
            rf = _scale_residuals(pts, T_ij[:3, :3], T_ij[:3, 3], da, ia, fine)
            kf = int(np.nanargmin(rf))
            best, rbest = fine[kf], rf[kf]
            # informative pair: clear minimum (pure rotation gives a flat curve -> no scale information)
            lo = _scale_residuals(pts, T_ij[:3, :3], T_ij[:3, 3], da, ia, np.array([best * 0.5, best * 2.0]))
            if rbest < 0.06 and np.nanmin(lo) > 1.5 * rbest + 0.01:
                found.append(best)
    if not found:
        return {
            "scale": None,
            "pairs": 0,
            "note": "no frame pair constrained the scale (camera did not translate enough, or depth missing)",
        }
    s = np.array(found)
    med = float(np.median(s))
    return {
        "scale": med,
        "pairs": len(found),
        "spread_pct": round(float(np.median(np.abs(s - med)) / med * 100), 2),
    }


def import_oneslam(session_dir: str | Path, poses_pred: str | Path, cfg: dict | None = None) -> dict:
    """Bring a OneSLAM run back: metric scale from depth, map-frame camera poses, base poses, validation."""
    session_dir = Path(session_dir)
    poses_pred = Path(poses_pred)
    if poses_pred.is_dir():
        cands = sorted(poses_pred.rglob("poses_pred.txt"))
        if not cands:
            raise FileNotFoundError(f"no poses_pred.txt under {poses_pred}")
        poses_pred = cands[-1]
    exp = session_dir / "scene" / "oneslam_input" / "frame_index.jsonl"
    if not exp.exists():
        raise FileNotFoundError("export first: posture-coach scene export-oneslam <session>")
    index = {
        r["oneslam_index"]: r
        for r in (json.loads(x) for x in exp.read_text(encoding="utf-8").splitlines() if x)
    }
    frames = {f["i"]: f for f in load_frames(session_dir)}
    os_raw = read_tum(poses_pred)
    # re-key OneSLAM frame indices -> session frame indices
    os_poses = {index[k]["session_frame"]: T for k, T in os_raw.items() if k in index}
    if len(os_poses) < 3:
        raise ValueError(f"only {len(os_poses)} OneSLAM poses matched the exported frames")
    sc = estimate_metric_scale(session_dir, frames, os_poses)
    kin = HeadKinematics.from_cfg(cfg or {})
    kp = kinematic_poses(list(frames.values()), kin)
    k0 = min(os_poses)
    T0 = inv_T(os_poses[k0])
    s = sc["scale"] if sc["scale"] else 1.0
    cam_map = {}
    for k, T in os_poses.items():
        rel = T0 @ T  # relative to OneSLAM's first keyframe
        rel = make_T(rel[:3, :3], rel[:3, 3] * s)
        cam_map[k] = kp[k0] @ rel  # anchored at the first keyframe's kinematic camera pose (gravity, height)
    base_map = {}
    for k, T in cam_map.items():
        f = frames[k]
        base_map[k] = T @ inv_T(kin.base_T_cam(f.get("pan") or 0.0, f.get("tilt") or 0.0))
    out_dir = session_dir / "scene" / "oneslam"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(poses_pred, out_dir / "poses_pred_raw.txt")
    metrics: dict = {
        "poses": len(os_poses),
        "scale": sc,
        "source_file": str(poses_pred),
        "metric": bool(sc["scale"]),
    }
    if sc["scale"]:
        write_tum(
            out_dir / "poses_cam_map.txt",
            cam_map,
            "frame_index tx ty tz qx qy qz qw  (camera-to-map, metric)",
        )
        write_tum(
            out_dir / "poses_base_map.txt",
            base_map,
            "frame_index tx ty tz qx qy qz qw  (base-to-map, metric)",
        )
        metrics.update(_validate(frames, kp, cam_map, base_map))
    else:
        metrics["note"] = (
            "OneSLAM trajectory imported but metric scale could not be recovered; the scene map keeps "
            "the kinematic poses. Record with base motion for camera-only localization."
        )
    (out_dir / "oneslam.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _validate(frames: dict, kp: dict, cam_map: dict, base_map: dict) -> dict:
    """Plan step 4/7 checks: head-only motion must not move the base; drift vs. the kinematic reference."""
    keys = sorted(cam_map)
    out = {}
    # (a) camera trajectory vs. kinematic reference (joint states + wheel odometry; not ground truth)
    d = np.array([np.linalg.norm(cam_map[k][:3, 3] - kp[k][:3, 3]) for k in keys])
    out["cam_vs_kinematic_rmse_m"] = round(float(np.sqrt(np.mean(d**2))), 3)
    out["cam_vs_kinematic_final_m"] = round(float(d[-1]), 3)
    # (b) head-only segments: wheel odometry says the base did not move, but the head did
    drift, segs, cur = [], 0, []
    for a, b in zip(keys, keys[1:], strict=False):
        ba, bb = np.array(frames[a].get("base") or [0, 0, 0]), np.array(frames[b].get("base") or [0, 0, 0])
        still = np.linalg.norm(bb[:2] - ba[:2]) < 0.01 and abs(bb[2] - ba[2]) < math.radians(0.5)
        head_moved = (
            abs((frames[b].get("pan") or 0) - (frames[a].get("pan") or 0)) > 0.02
            or abs((frames[b].get("tilt") or 0) - (frames[a].get("tilt") or 0)) > 0.02
        )
        if still:
            cur = cur or [a]
            cur.append(b)
            if head_moved:
                cur.append(-1)  # marker: this segment contains head motion
        else:
            if -1 in cur:
                segs += 1
                pts = np.array([base_map[k][:3, 3] for k in cur if k != -1])
                drift.append(float(np.max(np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1))))
            cur = []
    if -1 in cur:
        segs += 1
        pts = np.array([base_map[k][:3, 3] for k in cur if k != -1])
        drift.append(float(np.max(np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1))))
    out["head_only_segments"] = segs
    out["head_only_base_drift_max_m"] = round(max(drift), 3) if drift else None
    out["head_only_check"] = (
        None if not drift else ("pass" if max(drift) < 0.10 else "fail: base moves when only the head moves")
    )
    return out
