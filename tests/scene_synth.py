"""Ray-traced synthetic RGB-D room for scene / OneSLAM tests."""

from __future__ import annotations

import numpy as np

from stretch_posture_coach.camera import Intrinsics
from stretch_posture_coach.pose import PoseResult

ROOM = {"x": (-2.0, 4.0), "y": (-2.5, 2.5), "z": (0.0, 2.6)}
TABLE = {"x": (1.5, 2.3), "y": (-1.6, -0.8), "z": (0.0, 0.75)}
PERSON = {"x": (2.3, 2.6), "y": (0.4, 0.8), "z": (0.0, 1.7)}


def _box_hit(o, d, box, inside=False):
    """Ray/box distance (inside=True: exit distance from inside the room)."""
    lo = np.array([box["x"][0], box["y"][0], box["z"][0]])
    hi = np.array([box["x"][1], box["y"][1], box["z"][1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    if inside:
        return tmax
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(hit, tmin, np.inf)


def render(T_map_cam: np.ndarray, intr: Intrinsics, with_person: bool = True):
    """Depth (m) + BGR image + person mask for a camera pose."""
    h, w = intr.h, intr.w
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack(
        [(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u, float)], axis=-1
    ).reshape(-1, 3)
    d = rays @ T_map_cam[:3, :3].T
    o = np.broadcast_to(T_map_cam[:3, 3], d.shape)
    t_room = _box_hit(o, d, ROOM, inside=True)
    t_tab = _box_hit(o, d, TABLE)
    t_per = _box_hit(o, d, PERSON) if with_person else np.full(len(d), np.inf)
    t = np.minimum(np.minimum(t_room, t_tab), t_per)
    z = (t * rays[:, 2]).reshape(h, w).astype(np.float32)  # optical z (rays have z = 1)
    p = o + d * t[:, None]
    color = np.zeros((len(d), 3), np.uint8)
    color[:] = (200, 190, 180)
    color[p[:, 2] < 0.01] = (90, 140, 90)  # floor
    checker = ((np.floor(p[:, 0] * 2) + np.floor(p[:, 1] * 2) + np.floor(p[:, 2] * 2)) % 2) == 0
    color[checker & (p[:, 2] >= 0.01)] = (120, 110, 100)  # textured walls
    color[t == t_tab] = (40, 80, 160)
    person = (t == t_per) & np.isfinite(t_per)
    color[person] = (30, 30, 220)
    return z, color.reshape(h, w, 3), person.reshape(h, w)


def person_pose(T_map_cam: np.ndarray, intr: Intrinsics) -> PoseResult | None:
    """A crude 33-landmark pose whose image points cover the projected person box."""
    cx, cy = np.mean(PERSON["x"]), np.mean(PERSON["y"])
    pts_w = []
    for z in np.linspace(0.1, 1.65, 11):
        for dy in (-0.15, 0.15):
            pts_w.append([cx, cy + dy, z])
    pts_w = np.array(pts_w[:33] + [[cx, cy, 1.0]] * (33 - len(pts_w[:33])))
    Tc = np.linalg.inv(T_map_cam)
    pc = pts_w @ Tc[:3, :3].T + Tc[:3, 3]
    if np.any(pc[:, 2] <= 0.2):
        return None
    u = (pc[:, 0] / pc[:, 2] * intr.fx + intr.cx) / (intr.w - 1)
    v = (pc[:, 1] / pc[:, 2] * intr.fy + intr.cy) / (intr.h - 1)
    inside = (0 < u) & (u < 1) & (0 < v) & (v < 1)
    if inside.sum() < 6:  # like MediaPipe: partially visible people are still detected
        return None
    img = np.stack([np.clip(u, 0, 1), np.clip(v, 0, 1)], axis=1)
    vis = np.where(inside, 1.0, 0.0)
    # torso landmarks 11,12,23,24 placed at shoulder/hip heights in the middle of the person
    for i, zz, dy in ((11, 1.4, 0.15), (12, 1.4, -0.15), (23, 0.95, 0.1), (24, 0.95, -0.1)):
        q = Tc[:3, :3] @ np.array([cx - 0.15, cy + dy, zz]) + Tc[:3, 3]
        uu, vv = (
            (q[0] / q[2] * intr.fx + intr.cx) / (intr.w - 1),
            (q[1] / q[2] * intr.fy + intr.cy) / (intr.h - 1),
        )
        img[i] = [uu, vv]
        vis[i] = 1.0 if (0 < uu < 1 and 0 < vv < 1) else 0.0
    return PoseResult(image=img, world=np.zeros((33, 3)), visibility=vis)
