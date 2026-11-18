"""Control Barrier Functions (CBF) for the mobile base, built from what the lidar sees.

Model: unicycle  ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω  (Stretch is differential-drive).  The footprint is covered by
two circles centred at x = +l and x = -l on the base axis (radius r). For circle k and lidar obstacle point o:

    h_ko(x) = ‖c_k - o‖² - d²        d = r + margin      (h ≥ 0  ⇔  the circle keeps the margin from o)

In the base frame at the current instant  ċ_k = J_k u,  J_k = [[1, 0], [0, l_k]],  u = (v, ω), so

    ḣ_ko = 2 (c_k - o)ᵀ J_k u  ≥  -α h_ko                       (linear in u)

The safety filter solves the QP   min (v - v_ref)² + λ (ω - ω_ref)²   s.t. all constraints,
i.e. it changes the commanded velocity *as little as possible* while guaranteeing the robot never closes the
distance to an obstacle faster than α·h allows - it slows smoothly to a stop at the barrier, still lets you turn
away or back off, and works for every command source (expert joysticks, follow mode, the survey spin).

Obstacles are re-read from every lidar scan, so people (their legs) are barriers too. Points closer than
lidar.min_range_m are the robot itself and are ignored; the hard emergency stop in teleop.LidarGuard stays as a
second layer.

``barrier_lines`` gives the h = 0 level sets (keep-out lines around obstacles) in the map frame for the VR view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CBFResult:
    v: float
    w: float
    v_ref: float
    w_ref: float
    active: list = field(
        default_factory=list
    )  # [(circle_x, obstacle_xy)] constraints that changed the command
    h_min: float | None = None  # smallest barrier value (m²); < 0 = inside the margin
    feasible: bool = True

    @property
    def modified(self) -> bool:
        return abs(self.v - self.v_ref) > 1e-4 or abs(self.w - self.w_ref) > 1e-4


def _solve_qp_2d(r: np.ndarray, A: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, bool]:
    """min ‖u - r‖² s.t. A u ≥ b, by enumerating the (≤ 2) active constraints. Returns (u, feasible)."""

    def ok(u):
        return bool(np.all(A @ u >= b - 1e-7))

    if not len(A) or ok(r):
        return r.copy(), True
    cands = []
    nrm = np.einsum("ij,ij->i", A, A)
    for i in range(len(A)):
        if nrm[i] < eps:
            continue
        cands.append(r + (b[i] - A[i] @ r) / nrm[i] * A[i])
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = np.array([A[i], A[j]])
            det = np.linalg.det(M)
            if abs(det) > eps:
                cands.append(np.linalg.solve(M, [b[i], b[j]]))
    feas = [u for u in cands if ok(u)]
    if feas:
        return min(feas, key=lambda u: float(np.sum((u - r) ** 2))), True
    # infeasible (already inside the margin on several sides): least total violation, prefer slow motion
    cands.append(np.zeros(2))

    def viol(u):
        return float(np.sum(np.maximum(0.0, b - A @ u))) + 1e-3 * float(np.sum(u**2))

    return min(cands, key=viol), False


class CBFSafety:
    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", True))
        self.l = float(c.get("circle_offset_m", 0.08))
        self.r = float(c.get("circle_radius_m", 0.21))
        self.margin = float(c.get("margin_m", 0.20))
        self.alpha = float(c.get("alpha", 1.5))
        self.w_weight = float(c.get("turn_weight", 0.15))  # λ: turning is "cheaper" to change than speed
        self.range = float(c.get("range_m", 2.5))
        self.cell = float(c.get("cell_m", 0.08))
        self.max_constraints = int(c.get("max_constraints", 24))
        self.last: CBFResult | None = None

    @property
    def d(self) -> float:
        return self.r + self.margin

    def obstacles(self, pts_base: np.ndarray | None) -> np.ndarray:
        """Lidar points (base frame) within range, thinned to one per ``cell`` (keeps the QP small)."""
        if pts_base is None or not len(pts_base):
            return np.zeros((0, 2))
        p = pts_base[np.hypot(pts_base[:, 0], pts_base[:, 1]) < self.range]
        if not len(p):
            return p
        k = np.floor(p / self.cell).astype(np.int64)
        _, idx = np.unique(k[:, 0] * 100003 + k[:, 1], return_index=True)
        return p[idx]

    def constraints(self, obs: np.ndarray):
        """(A, b, h, meta) for both circles x all obstacles, keeping the most critical ones."""
        if not len(obs):
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0), []
        A, b, h, meta = [], [], [], []
        for lk in (self.l, -self.l):
            e = np.array([lk, 0.0]) - obs  # c_k - o
            hk = np.einsum("ij,ij->i", e, e) - self.d**2
            a = 2.0 * np.stack([e[:, 0], lk * e[:, 1]], axis=1)  # 2 eᵀ J_k
            A.append(a)
            b.append(-self.alpha * hk)
            h.append(hk)
            meta += [(lk, tuple(o)) for o in obs]
        A, b, h = np.vstack(A), np.concatenate(b), np.concatenate(h)
        keep = np.argsort(h)[: self.max_constraints]
        return A[keep], b[keep], h[keep], [meta[i] for i in keep]

    def filter(self, v_ref: float, w_ref: float, pts_base: np.ndarray | None) -> CBFResult:
        if not self.enabled:
            res = CBFResult(v_ref, w_ref, v_ref, w_ref)
            self.last = res
            return res
        obs = self.obstacles(pts_base)
        A, b, h, meta = self.constraints(obs)
        s = math.sqrt(self.w_weight)
        # change of variables u' = (v, s·ω) makes the weighted QP a plain projection
        A2 = A.copy()
        if len(A2):
            A2[:, 1] /= s
        u2, feas = _solve_qp_2d(np.array([v_ref, s * w_ref]), A2, b)
        v, w = float(u2[0]), float(u2[1] / s)
        res = CBFResult(v, w, v_ref, w_ref, h_min=float(h.min()) if len(h) else None, feasible=feas)
        if len(A) and res.modified:
            u = np.array([v, w])
            tight = np.abs(A @ u - b) < 1e-3 * (1 + np.abs(b))
            res.active = [meta[i] for i in np.nonzero(tight)[0]]
        self.last = res
        return res

    def limit_step(
        self, dx: float, dtheta: float, pts_base, dt: float | None = None
    ) -> tuple[float, float, CBFResult]:
        """Position steps (follow mode, survey spin) use the DISCRETE-time barrier condition

            h(x_k+1) >= (1 - gamma) h(x_k)      for every circle and obstacle

        (a continuous-time rate integrated over a whole step could overshoot the barrier). The step is scaled
        down (bisection on its length) until the condition holds; a step that only moves away is never cut."""
        obs = self.obstacles(pts_base)
        gamma = float(np.clip(self.alpha * 0.33, 0.05, 1.0))

        def h_after(s):
            out = []
            for lk in (self.l, -self.l):
                if abs(dtheta) > 0:
                    c = np.array([lk * math.cos(s * dtheta), lk * math.sin(s * dtheta)])
                else:
                    c = np.array([lk + s * dx, 0.0])
                e = c - obs
                out.append(np.einsum("ij,ij->i", e, e) - self.d**2)
            return np.concatenate(out)

        if not self.enabled or not len(obs):
            res = CBFResult(dx, dtheta, dx, dtheta)
            self.last = res
            return dx, dtheta, res
        h0 = h_after(0.0)

        def ok(s):
            h1 = h_after(s)
            return bool(np.all((h1 >= (1 - gamma) * h0) | (h1 >= h0)))

        if ok(1.0):
            scale = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(18):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            scale = lo
        sx, st = dx * scale, dtheta * scale
        res = CBFResult(sx, st, dx, dtheta, h_min=float(h0.min()))
        if scale < 1.0:
            k = int(np.argmin(h_after(1.0) - (1 - gamma) * h0))
            res.active = [((self.l, -self.l)[k // len(obs)], tuple(obs[k % len(obs)]))]
        self.last = res
        return sx, st, res

    def barrier_lines(self, pts_base: np.ndarray | None, pose, res_m: float = 0.04) -> list[dict]:
        """Keep-out lines (h = 0 for the footprint circles) around lidar obstacles, as map-frame polylines.

        Each: {"pts": [[x, y], ...], "level": "active" | "near" | "clear", "gap": metres to the robot}."""
        obs = self.obstacles(pts_base)
        if not len(obs):
            return []
        R = self.range + self.d + 0.1
        n = int(2 * R / res_m)
        img = np.zeros((n, n), np.uint8)
        ij = np.floor((obs + R) / res_m).astype(np.int32)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        img[ij[ok, 1], ij[ok, 0]] = 255
        k = int(round(self.d / res_m))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        img = cv2.dilate(img, disk)
        contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # free range per 2° bearing (the lidar's line of sight): nearest obstacle in that direction
        ob_r = np.hypot(obs[:, 0], obs[:, 1])
        ob_b = (np.degrees(np.arctan2(obs[:, 1], obs[:, 0])) % 360 / 2).astype(int) % 180
        rng = np.full(180, self.range + self.d)
        np.minimum.at(rng, ob_b, ob_r)
        rng = np.minimum(np.minimum(rng, np.roll(rng, 1)), np.roll(rng, -1))  # close gaps between scan points
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        active = {o for (_, o) in (self.last.active if self.last else [])}
        me = (float(R / res_m), float(R / res_m))  # the robot, in grid pixels
        out = []
        for i, cnt in enumerate(contours):
            is_hole = hier is not None and hier[0][i][3] >= 0
            # a ring of walls around the robot: draw its INNER edge (the hole that faces the robot), not the outer
            if not is_hole and cv2.pointPolygonTest(cnt, me, False) >= 0:
                continue
            p = cnt.reshape(-1, 2).astype(np.float64) * res_m - R + res_m / 2
            if len(p) < 3:
                continue
            gap = float(np.min(np.hypot(p[:, 0], p[:, 1])))
            if is_hole:  # only the obstacles bordering this hole matter for "active"
                parent = contours[hier[0][i][3]]
                hit = any(
                    cv2.pointPolygonTest(
                        parent, (float((o[0] + R) / res_m), float((o[1] + R) / res_m)), False
                    )
                    >= 0
                    for o in active
                )
            else:
                hit = any(
                    cv2.pointPolygonTest(cnt, (float((o[0] + R) / res_m), float((o[1] + R) / res_m)), False)
                    >= 0
                    for o in active
                )
            level = "active" if hit else ("near" if gap < 0.5 else "clear")
            # keep only the part of the line the robot can reach (not the far side of a wall, behind the
            # obstacle as seen by the lidar): split into visible runs
            vis = rng[
                (np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360 / 2).astype(int) % 180
            ] + 0.05 >= np.hypot(p[:, 0], p[:, 1])
            runs, cur = [], []
            for k in range(len(p)):
                if vis[k]:
                    cur.append(k)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                if runs and vis[0]:
                    runs[0] = cur + runs[0]  # wrap around the closed contour
                else:
                    runs.append(cur)
            closed = bool(vis.all())
            for run in runs:
                q = p[run]
                if len(q) < 2:
                    continue
                pm = np.stack([x + c * q[:, 0] - s * q[:, 1], y + s * q[:, 0] + c * q[:, 1]], 1)
                out.append(
                    {"pts": np.round(pm, 3).tolist(), "level": level, "gap": round(gap, 2), "closed": closed}
                )
        return out


class SafeBase:
    """Wraps the robot so that follow mode's relative base steps pass the CBF filter first.

    Everything else is delegated unchanged. ``world`` is attached after construction (it needs the follower)."""

    def __init__(self, robot, world=None):
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "last_note", "")

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __setattr__(self, name, value):
        if name in ("world", "last_note"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._robot, name, value)

    def _filter(self, dx: float, dth: float) -> tuple[float, float]:
        w = self.world
        pts = w.obstacle_points() if w is not None else None
        if pts is None:
            return dx, dth
        sx, st, res = w.cbf.limit_step(dx, dth, pts)
        note = ""
        if abs(sx - dx) > 1e-3 or abs(st - dth) > 1e-3:
            note = f"barrier limited step {dx:+.2f} m/{dth:+.2f} rad -> {sx:+.2f}/{st:+.2f}"
            import logging

            logging.getLogger(__name__).info("follow: %s", note)
        object.__setattr__(self, "last_note", note)
        return sx, st

    def base_translate(self, meters: float) -> None:
        sx, _ = self._filter(meters, 0.0)
        if abs(sx) > 0.005:
            self._robot.base_translate(sx)

    def base_rotate(self, rad: float, compensate_head: bool = True) -> None:
        _, st = self._filter(0.0, rad)
        if abs(st) > 0.005:
            self._robot.base_rotate(st, compensate_head=compensate_head)
