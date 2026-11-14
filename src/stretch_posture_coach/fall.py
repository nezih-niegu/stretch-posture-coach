"""Fall detection: rapid descent + ending up lying (or vanishing downward) -> "are you OK?" -> alarm.

States
  ok        normal
  impact    a rapid downward movement was just seen; waiting up to ``settle_s`` to see how the person ends up
  check     the person is lying (or dropped out of view): Stretch asks "are you OK?" and waits ``check_s``
  alarm     no recovery in time: siren, spoken alert, notifications (until acknowledged or the person recovers)

Cues (any one starts ``impact``; the "lying" check then rules out sitting down fast or bending over):
  * hips drop ``drop_m`` within ``drop_window_s``         (3-D, from the depth camera, gravity-corrected)
  * hips move down >= ``descent_torso_lengths`` torso lengths in the image within the window (no depth needed)
  * person vanishes from view while descending
Lying = trunk inclination >= ``lying_incl_deg`` (0 = upright, 90 = horizontal) and, when depth gives hip
height, hips below ``lying_hip_m``. Also: lying on the floor (hips < ``floor_height_m``) for ``long_lying_s``
without a seen fall -> check.
Recovery (cancels check/alarm): upright (inclination <= ``upright_incl_deg``) for ``recover_s``, or a raised
hand held for ``hand_ok_s`` during the check, or an acknowledgement from the web/VR viewer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

OK, IMPACT, CHECK, ALARM = "ok", "impact", "check", "alarm"


@dataclass
class FallFeatures:
    t: float
    present: bool
    inclination: float | None = None  # deg
    hip_height: float | None = None  # m above the floor (depth), or None
    hip_y: float | None = None  # normalized image y of the hip midpoint (down = larger)
    torso_len: float | None = None  # normalized image length shoulder-mid -> hip-mid
    hand_raised: bool = False


def hip_height_from_depth(
    pose, depth_m, intr, gravity_cam: np.ndarray, camera_height_m: float, min_vis: float = 0.5
) -> float | None:
    """Height of the hip midpoint above the floor, from the aligned depth image and the camera pose."""
    if pose is None or depth_m is None or intr is None:
        return None
    h, w = depth_m.shape[:2]
    pts = []
    for i in (23, 24):
        if pose.visibility[i] < min_vis:
            continue
        u, v = pose.image[i]
        px, py = int(round(u * (w - 1))), int(round(v * (h - 1)))
        if not (0 <= px < w and 0 <= py < h):
            continue
        win = depth_m[max(0, py - 4) : py + 5, max(0, px - 4) : px + 5]
        valid = win[(win > 0.2) & (win < 8.0)]
        if valid.size < 5:
            continue
        z = float(np.median(valid))
        sx, sy = intr.w / w, intr.h / h
        pts.append(np.array([(px * sx - intr.cx) / intr.fx * z, (py * sy - intr.cy) / intr.fy * z, z]))
    if not pts:
        return None
    p = np.mean(pts, axis=0)
    return float(camera_height_m - np.dot(p, gravity_cam))  # gravity_cam points down in camera coords


def features_from(
    t: float, present: bool, pose, metrics: dict | None, hip_height: float | None, min_vis: float = 0.5
) -> FallFeatures:
    f = FallFeatures(t=t, present=present)
    if not present or pose is None or not metrics:
        return f
    f.inclination = metrics.get("trunk_inclination")
    f.hand_raised = bool(metrics.get("hand_raised"))
    f.hip_height = hip_height
    v = pose.visibility
    if min(v[11], v[12], v[23], v[24]) >= min_vis:
        sh = (pose.image[11] + pose.image[12]) / 2
        hp = (pose.image[23] + pose.image[24]) / 2
        f.hip_y = float(hp[1])
        f.torso_len = float(np.linalg.norm(sh - hp))
    return f


class FallDetector:
    def __init__(self, cfg: dict):
        self.c = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.state = OK
        self.reason = ""
        self.since = 0.0
        self._hist: deque[FallFeatures] = deque()
        self._impact_t = None
        self._lying_since = None
        self._floor_since = None
        self._upright_since = None
        self._hand_since = None
        self._last_present_t = None

    def _p(self, k, d):
        return float(self.c.get(k, d))

    def _set(self, state: str, t: float, reason: str) -> tuple[str, str]:
        self.state, self.since, self.reason = state, t, reason
        return state, reason

    def acknowledge(self, t: float, who: str = "viewer") -> tuple[str, str] | None:
        if self.state in (CHECK, ALARM, IMPACT):
            self._impact_t = None
            return self._set(OK, t, f"acknowledged by {who}")
        return None

    # ------------------------------------------------------------------ cues
    def _descent(self, f: FallFeatures) -> str | None:
        win = self._p("drop_window_s", 1.2)
        recent = [h for h in self._hist if f.t - h.t <= win]
        hh = [h.hip_height for h in recent if h.hip_height is not None]
        if f.hip_height is not None and hh and max(hh) - f.hip_height >= self._p("drop_m", 0.45):
            return f"hips dropped {max(hh) - f.hip_height:.2f} m in {win:.1f} s"
        ys = [h for h in recent if h.hip_y is not None and h.torso_len]
        if f.hip_y is not None and f.torso_len and ys:
            top = min(ys, key=lambda h: h.hip_y)
            torso = max(top.torso_len, 1e-3)
            moved = (f.hip_y - top.hip_y) / torso
            if moved >= self._p("descent_torso_lengths", 1.0):
                return f"hips moved down {moved:.1f} torso lengths in {win:.1f} s"
        return None

    def is_down(self, f: FallFeatures) -> bool:
        """Lying with low hips (or lying, when there's no depth) - nobody needs posture tips right now."""
        return f.present and self._lying(f)

    def _lying(self, f: FallFeatures) -> bool:
        if f.inclination is None or f.inclination < self._p("lying_incl_deg", 55):
            return False
        return f.hip_height is None or f.hip_height < self._p("lying_hip_m", 0.65)

    def _upright(self, f: FallFeatures) -> bool:
        if f.inclination is None or f.inclination > self._p("upright_incl_deg", 35):
            return False
        return f.hip_height is None or f.hip_height > self._p("upright_hip_m", 0.45)

    def _was_descending(self, t: float) -> bool:
        """Did the hips move down noticeably just before the person vanished?"""
        recent = [h for h in self._hist if t - h.t <= 1.0 and h.hip_y is not None and h.torso_len]
        if len(recent) < 3:
            return False
        first, last = recent[0], recent[-1]
        return (last.hip_y - first.hip_y) / max(first.torso_len, 1e-3) >= 0.6

    # ------------------------------------------------------------------ main
    def update(self, f: FallFeatures) -> tuple[str, str] | None:
        """Feed one frame. Returns (new_state, reason) when the state changes, else None."""
        if not self.enabled:
            return None
        t = f.t
        out = None

        if f.present:
            self._last_present_t = t
            self._upright_since = (self._upright_since or t) if self._upright(f) else None
            self._hand_since = (self._hand_since or t) if f.hand_raised else None
            on_floor = (
                self._lying(f) and f.hip_height is not None and f.hip_height < self._p("floor_height_m", 0.45)
            )
            self._floor_since = (self._floor_since or t) if on_floor else None
        else:
            self._upright_since = self._hand_since = None

        if self.state == OK:
            if f.present:
                cue = self._descent(f)
                if cue:
                    self._impact_t = t
                    out = self._set(IMPACT, t, cue)
                elif (
                    self._floor_since is not None
                    and self._p("long_lying_s", 30) > 0
                    and t - self._floor_since >= self._p("long_lying_s", 30)
                ):
                    out = self._set(CHECK, t, f"lying on the floor for {t - self._floor_since:.0f} s")
            elif (
                self._last_present_t is not None
                and t - self._last_present_t < 0.6
                and self._was_descending(t)
            ):
                self._impact_t = t
                out = self._set(IMPACT, t, "dropped out of view while going down")

        elif self.state == IMPACT:
            if f.present and self._lying(f):
                out = self._set(CHECK, t, f"{self.reason}; now lying ({f.inclination:.0f} deg)")
            elif not f.present and t - self._impact_t >= 1.0:
                out = self._set(CHECK, t, f"{self.reason}; not visible anymore")
            elif t - self._impact_t > self._p("settle_s", 3.0):
                out = self._set(OK, t, "not lying after the movement (sat down / bent over)")

        elif self.state in (CHECK, ALARM):
            if self._upright_since is not None and t - self._upright_since >= self._p("recover_s", 3.0):
                out = self._set(OK, t, "person is upright again")
            elif (
                self.state == CHECK
                and self.c.get("hand_raise_ok", True)
                and self._hand_since is not None
                and t - self._hand_since >= self._p("hand_ok_s", 1.5)
            ):
                out = self._set(OK, t, "raised a hand: OK")
            elif self.state == CHECK and t - self.since >= self._p("check_s", 12):
                out = self._set(ALARM, t, f"no recovery {self._p('check_s', 12):.0f} s after: {self.reason}")

        self._hist.append(f)
        while self._hist and t - self._hist[0].t > 3.0:
            self._hist.popleft()
        return out

    @property
    def active(self) -> bool:
        return self.state in (CHECK, ALARM)
