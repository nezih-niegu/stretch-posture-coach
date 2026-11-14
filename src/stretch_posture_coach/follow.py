"""Follow mode: keep Stretch facing the person at a safe 2-3 m distance.

Design (safety first):
  * Distance comes from the D435i depth image (median over the torso), with a monocular fallback.
  * The head keeps tracking the person; the BASE turns so the head returns to its forward pan
    (the body faces the person) and the head is counter-panned during the turn.
  * The base only ever gets small, bounded, RELATIVE position steps (no velocity commands), one at a time,
    and only after the previous step has had time to finish.
  * It never drives forward when the person is closer than ``min_m`` or when distance is unknown, never
    drives forward unless facing the person, limits total back-up distance, stays within ``max_travel_m``
    of where it started, and freezes when the person is lost or the runstop is pressed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

TORSO = (11, 12, 23, 24)


def person_distance(
    pose, depth_m: np.ndarray | None, min_vis: float = 0.5, vfov_deg: float = 69.0
) -> tuple[float | None, str]:
    """Distance (m) from the camera to the person's torso, and its source ("depth", "pose" or "none")."""
    if pose is None:
        return None, "none"
    vis = pose.visibility[list(TORSO)]
    if (vis >= min_vis).sum() < 3:
        return None, "none"
    pts = pose.image[list(TORSO)][vis >= min_vis]

    if depth_m is not None and depth_m.size:
        h, w = depth_m.shape[:2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = max((x1 - x0) * 0.25, 0.01), max((y1 - y0) * 0.25, 0.01)  # central half of the torso box
        xa, xb = int(np.clip((cx - hw) * w, 0, w - 1)), int(np.clip((cx + hw) * w, 1, w))
        ya, yb = int(np.clip((cy - hh) * h, 0, h - 1)), int(np.clip((cy + hh) * h, 1, h))
        patch = depth_m[ya:yb, xa:xb]
        valid = patch[(patch > 0.3) & (patch < 8.0)]
        if valid.size >= 25:
            return float(np.median(valid)), "depth"

    # Monocular fallback: real torso length (MediaPipe world, ~meters) vs its length in pixels.
    world_len = float(
        np.linalg.norm((pose.world[11] + pose.world[12]) / 2 - (pose.world[23] + pose.world[24]) / 2)
    )
    img_len = float(
        np.linalg.norm((pose.image[11] + pose.image[12]) / 2 - (pose.image[23] + pose.image[24]) / 2)
    )
    if img_len < 1e-3 or world_len < 0.2:
        return None, "none"
    focal_norm = 0.5 / math.tan(math.radians(vfov_deg) / 2)  # focal length in units of image height
    return float(focal_norm * world_len / img_len), "pose"


@dataclass
class FollowState:
    distance: float | None = None
    source: str = "none"
    bearing_deg: float | None = None
    action: str = "idle"  # idle | hold | rotate | approach | back_off | blocked | lost | runstop
    note: str = ""


class Follower:
    def __init__(self, cfg: dict, robot_cfg: dict, head):
        self.c = cfg
        self.rc = robot_cfg
        self.head = head
        self.enabled = bool(cfg.get("enabled", False)) and getattr(head, "has_base", False)
        self.state = FollowState()
        self._busy_until = -1e9
        self._stable = 0
        self._last_seen = -1e9
        self._dist_hist: list[float] = []
        # dead-reckoned base pose relative to the start (x forward, y left, theta CCW)
        self.x = self.y = self.theta = 0.0
        self._backed_since_forward = 0.0

    def _p(self, key, default):
        return float(self.c.get(key, default))

    def bearing(self, target_x: float | None) -> float | None:
        """Person direction relative to the robot body (rad, CCW positive)."""
        pan = self.head.pan
        if pan is None:
            return None
        b = pan - self._p("forward_pan_rad", 0.0)
        if target_x is not None:  # residual offset of the person in the image
            b += (
                float(self.rc.get("pan_sign", -1))
                * (target_x - 0.5)
                * math.radians(float(self.rc.get("hfov_deg", 42)))
            )
        return b

    def update(
        self, present: bool, distance: float | None, source: str, target_x: float | None, now: float
    ) -> FollowState:
        s = self.state
        s.distance, s.source = distance, source
        if not self.enabled:
            s.action, s.note = "idle", "follow mode off"
            return s
        if getattr(self.head, "runstop", None):
            s.action, s.note = "runstop", "runstop pressed"
            self._stable = 0
            return s
        if not present or distance is None:
            self._stable = 0
            self._dist_hist.clear()  # don't decide on stale readings when the person comes back
            s.action = "lost"
            s.note = "no person in view" if not present else "no distance (no depth on the torso)"
            s.bearing_deg = None
            return s
        self._last_seen = now
        self._stable += 1
        # median of the last few readings rejects depth glitches
        self._dist_hist = ([*self._dist_hist, distance])[-7:]
        d = float(np.median(self._dist_hist))
        s.distance = round(d, 2)
        b = self.bearing(target_x)
        s.bearing_deg = None if b is None else round(math.degrees(b), 1)
        if b is None:
            s.action, s.note = "hold", "no head pan reading from the robot"
            return s
        need = int(self._p("stable_frames", 6))
        if self._stable < need:
            s.action, s.note = "hold", f"confirming detection {self._stable}/{need}"
            return s
        if now < self._busy_until:
            s.note = "step in progress"
            return s  # previous step still executing

        min_m, max_m, target = self._p("min_m", 2.0), self._p("max_m", 3.0), self._p("target_m", 2.5)
        emergency = self._p("emergency_m", 1.2)
        tol = math.radians(self._p("align_tol_deg", 12))
        max_rot = math.radians(self._p("max_rotate_step_deg", 20))
        max_lin = self._p("max_translate_step_m", 0.15)
        v_lin, v_rot = self._p("translate_speed_m_s", 0.08), self._p("rotate_speed_rad_s", 0.3)
        settle = self._p("settle_s", 0.6)

        # 1) too close -> back off first (only if the person is roughly in front, so backing increases distance)
        if d < min_m and abs(b) < math.radians(60):
            step = min(target - d, max_lin)
            if d < emergency:
                step = max_lin
            if self._backed_since_forward + step > self._p("max_backoff_total_m", 1.0):
                s.action, s.note = "blocked", "back-up limit reached"
                return s
            if not self._within_travel(-step):
                s.action, s.note = "blocked", "travel limit reached"
                return s
            self._translate(-step, now, v_lin, settle)
            self._backed_since_forward += step
            s.action, s.note = "back_off", f"{step:.2f} m"
            return s

        # 2) turn the body toward the person
        if abs(b) > tol:
            rot = max(-max_rot, min(max_rot, b))
            self.head.base_rotate(rot, compensate_head=True)
            self.theta += rot
            self._busy_until = now + abs(rot) / v_rot + settle
            self._stable = max(0, self._stable - 2)
            s.action, s.note = "rotate", f"{math.degrees(rot):+.0f} deg"
            return s

        # 3) too far -> approach, but never closer than min_m, and only when facing the person
        if d > max_m and source != "none":
            step = min(d - target, max_lin)
            if source == "pose":
                step = min(step, max_lin / 2)  # monocular distance is rough: smaller steps
            if d - step < min_m:
                step = max(0.0, d - min_m)
            if step > 0.02:
                if not self._within_travel(step):
                    s.action, s.note = "blocked", "travel limit reached"
                    return s
                self._translate(step, now, v_lin, settle)
                self._backed_since_forward = max(0.0, self._backed_since_forward - step)
                s.action, s.note = "approach", f"{step:.2f} m"
                return s

        s.action = "hold"
        s.note = (
            f"in band ({d:.2f} m, {math.degrees(b):+.0f} deg)"
            if min_m <= d <= max_m
            else f"{d:.2f} m but {math.degrees(b):+.0f} deg off-center"
        )
        return s

    def _within_travel(self, dx: float) -> bool:
        nx = self.x + dx * math.cos(self.theta)
        ny = self.y + dx * math.sin(self.theta)
        return math.hypot(nx, ny) <= self._p("max_travel_m", 3.0) or math.hypot(nx, ny) < math.hypot(
            self.x, self.y
        )

    def _translate(self, dx: float, now: float, v: float, settle: float):
        self.head.base_translate(dx)
        self.x += dx * math.cos(self.theta)
        self.y += dx * math.sin(self.theta)
        self._busy_until = now + abs(dx) / v + settle
