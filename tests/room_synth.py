"""Synthetic furnished rooms with ground truth, for testing and evaluating the scene graph.

Each object is sampled as the voxel centres of its visible surfaces (top and sides; 5 cm), with range noise,
dropout and stray points, the way the live map sees furniture after carving."""

from __future__ import annotations

import math

import numpy as np

CELL = 0.05


def _box_surface(cx, cy, yaw, ln, w, z0, z1, top=True, sides=True, step=CELL):
    """Points on the top and the four sides of an oriented box."""
    pts = []
    us = np.arange(-ln / 2, ln / 2 + 1e-9, step)
    vs = np.arange(-w / 2, w / 2 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    if top:
        U, V = np.meshgrid(us, vs)
        pts.append(np.stack([U.ravel(), V.ravel(), np.full(U.size, z1)], 1))
    if sides:
        for u in (-ln / 2, ln / 2):
            V, Z = np.meshgrid(vs, zs)
            pts.append(np.stack([np.full(V.size, u), V.ravel(), Z.ravel()], 1))
        for v in (-w / 2, w / 2):
            U, Z = np.meshgrid(us, zs)
            pts.append(np.stack([U.ravel(), np.full(U.size, v), Z.ravel()], 1))
    P = np.concatenate(pts)
    c, s = math.cos(yaw), math.sin(yaw)
    x = cx + c * P[:, 0] - s * P[:, 1]
    y = cy + s * P[:, 0] + c * P[:, 1]
    return np.stack([x, y, P[:, 2]], 1)


def _local(cx, cy, yaw, du, dv):
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + c * du - s * dv, cy + s * du + c * dv


def furniture(kind: str, cx: float, cy: float, yaw: float) -> tuple[np.ndarray, dict]:
    """Points + ground truth (label, footprint polygon dims, top surface) for one piece of furniture."""
    if kind == "table":
        ln, w, top = 1.2, 0.7, 0.75
        P = [_box_surface(cx, cy, yaw, ln, w, top - 0.04, top)]
        for du in (-ln / 2 + 0.05, ln / 2 - 0.05):
            for dv in (-w / 2 + 0.05, w / 2 - 0.05):
                x, y = _local(cx, cy, yaw, du, dv)
                P.append(_box_surface(x, y, yaw, 0.05, 0.05, 0.0, top - 0.04, top=False))
        gt = dict(label="table", ln=ln, w=w, top=top)
    elif kind == "chair":
        ln, w, top = 0.45, 0.45, 0.46
        P = [_box_surface(cx, cy, yaw, ln, w, top - 0.04, top)]
        x, y = _local(cx, cy, yaw, -ln / 2 + 0.03, 0)
        P.append(_box_surface(x, y, yaw, 0.05, w, top, 0.9))  # backrest
        for du in (-ln / 2 + 0.03, ln / 2 - 0.03):
            for dv in (-w / 2 + 0.03, w / 2 - 0.03):
                x, y = _local(cx, cy, yaw, du, dv)
                P.append(_box_surface(x, y, yaw, 0.04, 0.04, 0.0, top - 0.04, top=False))
        gt = dict(label="chair", ln=ln, w=w, top=top)
    elif kind == "sofa":
        ln, w, top = 1.9, 0.9, 0.45
        P = [_box_surface(cx, cy, yaw, ln, w - 0.2, 0.0, top)]
        x, y = _local(cx, cy, yaw, 0, -(w / 2 - 0.1))
        P.append(_box_surface(x, y, yaw, ln, 0.2, 0.0, 0.85))  # back
        gt = dict(label="sofa", ln=ln, w=w, top=top)
    elif kind == "bed":
        ln, w, top = 2.0, 1.4, 0.55
        P = [_box_surface(cx, cy, yaw, ln, w, 0.0, top)]
        gt = dict(label="bed", ln=ln, w=w, top=top)
    elif kind == "cabinet":
        ln, w, top = 0.8, 0.4, 1.8
        P = [_box_surface(cx, cy, yaw, ln, w, 0.0, top)]
        gt = dict(label="cabinet", ln=ln, w=w, top=top)
    elif kind == "box":
        ln, w, top = 0.4, 0.35, 0.25
        P = [_box_surface(cx, cy, yaw, ln, w, 0.0, top)]
        gt = dict(label="low_object", ln=ln, w=w, top=top)
    else:
        raise ValueError(kind)
    gt.update(kind=kind, cx=cx, cy=cy, yaw=yaw)
    return np.concatenate(P), gt


def walls(W: float = 6.0, D: float = 5.0, H: float = 2.5) -> np.ndarray:
    return np.concatenate(
        [
            _box_surface(W / 2, 0.0, 0.0, W, 0.1, 0.0, H, top=False),
            _box_surface(W / 2, D, 0.0, W, 0.1, 0.0, H, top=False),
            _box_surface(0.0, D / 2, 0.0, 0.1, D, 0.0, H, top=False),
            _box_surface(W, D / 2, 0.0, 0.1, D, 0.0, H, top=False),
        ]
    )


def _footprint_ok(placed, cx, cy, r, W, D):
    if not (r + 0.15 < cx < W - r - 0.15 and r + 0.15 < cy < D - r - 0.15):
        return False
    return all(math.hypot(cx - x, cy - y) > r + rr + 0.15 for x, y, rr in placed)


HARD = dict(noise=0.02, dropout=0.4, outliers=0.03, gap=(0.0, 0.05), any_yaw=True)


def random_room(
    seed: int,
    W: float = 6.0,
    D: float = 5.0,
    noise: float = 0.01,
    dropout: float = 0.2,
    outliers: float = 0.01,
    chairs_at_tables: bool = True,
    gap=(0.05, 0.10),
    yaw_sd: float = 0.15,
    any_yaw: bool = False,
):
    """A room with 2 tables (chairs pushed in), a sofa or bed, a cabinet and a box. Returns points, gts."""
    rng = np.random.default_rng(seed)
    P = [walls(W, D)]
    gts = []
    placed = []
    kinds = ["table", "table", rng.choice(["sofa", "bed"]), "cabinet", "box", "chair"]
    radius = {"table": 0.9, "sofa": 1.05, "bed": 1.25, "cabinet": 0.5, "box": 0.3, "chair": 0.35}
    for kind in kinds:
        for _ in range(200):
            cx, cy = rng.uniform(0, W), rng.uniform(0, D)
            yaw = (
                float(rng.uniform(0, math.pi))
                if any_yaw
                else float(rng.choice([0.0, math.pi / 2]) + rng.normal(0, yaw_sd))
            )
            if _footprint_ok(placed, cx, cy, radius[kind], W, D):
                break
        else:
            continue
        pts, gt = furniture(kind, cx, cy, yaw)
        P.append(pts)
        gts.append(gt)
        placed.append((cx, cy, radius[kind]))
        if kind == "table" and chairs_at_tables:
            # one chair pushed in at each long side, 5-10 cm from the table edge
            for side in (-1, 1):
                g_ = rng.uniform(*gap)
                du, dv = 0.0, side * (0.35 + g_ + 0.225)
                x, y = _local(cx, cy, yaw, du, dv)
                cyaw = yaw + (math.pi / 2 if side < 0 else -math.pi / 2)
                pts, g = furniture("chair", x, y, cyaw)
                P.append(pts)
                gts.append(g)
    X = np.concatenate(P)
    X = X[rng.random(len(X)) > dropout]
    X = X + rng.normal(0, noise, X.shape)
    n_out = int(outliers * len(X))
    stray = np.stack([rng.uniform(0, W, n_out), rng.uniform(0, D, n_out), rng.uniform(0.1, 2.3, n_out)], 1)
    X = np.concatenate([X, stray])
    # voxelize like the live map (one point per occupied 5 cm voxel)
    k = np.unique(np.floor(X / CELL).astype(np.int64), axis=0)
    return (k + 0.5) * CELL, gts


def gt_cells(gt: dict, cell: float = CELL) -> set:
    """Footprint cells of a ground-truth object."""
    pts = _box_surface(
        gt["cx"], gt["cy"], gt["yaw"], gt["ln"], gt["w"], 0, 0, top=True, sides=False, step=cell / 2
    )
    return set(
        zip(
            np.floor(pts[:, 0] / cell).astype(int).tolist(),
            np.floor(pts[:, 1] / cell).astype(int).tolist(),
            strict=True,
        )
    )
