"""Full-body joint angles and a static biomechanical load model from MediaPipe pose landmarks.

Coordinates are MediaPipe *world* landmarks (meters, origin between the hips, camera-aligned axes:
x = image right, y = image down, z = away from the camera). Gravity is supplied as a unit vector in that
same frame so that a tilted camera (Stretch head tilt) does not corrupt the vertical reference.

All angles are in degrees. Conventions:
  * trunk_flexion  > 0 leaning forward, < 0 leaning back (relative to gravity)
  * neck_flexion   > 0 head forward/down relative to the trunk, < 0 extension
  * *_side_bend    > 0 toward the person's left
  * upper_arm_*    0 = hanging along the trunk
  * elbow/knee/hip flexion 0 = straight
The numbers are estimates from a monocular pose model and are meant for coaching, not clinical use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81

LM = {
    "nose": 0,
    "l_ear": 7,
    "r_ear": 8,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_pinky": 17,
    "r_pinky": 18,
    "l_index": 19,
    "r_index": 20,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
}

# Segment mass fractions of body mass (Winter 2009 / Dempster) and COM location from proximal end.
MASS_HEAD_NECK = 0.081
MASS_TRUNK_ABOVE_L5 = 0.355  # thorax + abdomen
MASS_UPPER_ARM = 0.028
MASS_FOREARM = 0.016
MASS_HAND = 0.006
COM_UPPER_ARM = 0.436
COM_FOREARM = 0.430
L5S1_FROM_HIP = 0.12  # fraction of hip-mid -> shoulder-mid where L5/S1 is approximated
TRUNK_COM_FROM_L5 = 0.55
ERECTOR_MOMENT_ARM_M = 0.05
NECK_EXTENSOR_MOMENT_ARM_M = 0.05

# Metric keys calibrated by subtracting the value measured in a neutral posture.
CALIBRATED_KEYS = ("neck_flexion", "neck_side_bend", "trunk_flexion", "trunk_side_bend", "head_forward_cm")


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    return math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0))))


def plane_angle(v: np.ndarray, ref: np.ndarray, pos: np.ndarray) -> float:
    """Signed angle of ``v`` in the plane spanned by ``ref`` (0 deg) and ``pos`` (+90 deg)."""
    return math.degrees(math.atan2(float(np.dot(v, pos)), float(np.dot(v, ref))))


def horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    return v - np.dot(v, up) * up


def signed_angle_about(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from ``a`` to ``b`` around ``axis`` (both projected on the plane normal to axis)."""
    a_h, b_h = unit(horizontal(a, axis)), unit(horizontal(b, axis))
    return math.degrees(math.atan2(float(np.dot(np.cross(a_h, b_h), axis)), float(np.dot(a_h, b_h))))


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def gravity_from_pitch(pitch_deg: float) -> np.ndarray:
    """Gravity (unit, pointing down) in an upright camera image frame pitched by ``pitch_deg`` (down < 0)."""
    theta = math.radians(-pitch_deg)
    return np.array([0.0, math.cos(theta), math.sin(theta)])


@dataclass
class Subject:
    body_mass_kg: float = 70.0
    hand_load_kg: float = 0.0
    arms_supported: bool = False


@dataclass
class Calibration:
    offsets: dict[str, float] = field(default_factory=dict)
    ear_shoulder_cm: dict[str, float] = field(
        default_factory=dict
    )  # side -> vertical ear-to-shoulder distance

    def to_dict(self) -> dict:
        return {"offsets": self.offsets, "ear_shoulder_cm": self.ear_shoulder_cm}

    @classmethod
    def from_dict(cls, d: dict | None) -> Calibration:
        d = d or {}
        return cls(offsets=dict(d.get("offsets", {})), ear_shoulder_cm=dict(d.get("ear_shoulder_cm", {})))


class _Pts:
    def __init__(self, world: np.ndarray, vis: np.ndarray, min_vis: float):
        self.world, self.vis, self.min_vis = world, vis, min_vis

    def __getitem__(self, name: str) -> np.ndarray:
        return self.world[LM[name]]

    def ok(self, *names: str) -> bool:
        return all(self.vis[LM[n]] >= self.min_vis for n in names)

    def mid(self, a: str, b: str) -> np.ndarray:
        return (self[a] + self[b]) / 2.0


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(float(x), nd)


def compute_metrics(
    world: np.ndarray,
    visibility: np.ndarray,
    gravity: np.ndarray,
    subject: Subject | None = None,
    calibration: Calibration | None = None,
    min_visibility: float = 0.5,
    apply_calibration: bool = True,
) -> dict:
    """Compute posture metrics from (33, 3) world landmarks and (33,) visibility.

    Returns a flat dict; values are floats/bools/str or None when the needed landmarks are not visible.
    """
    subject = subject or Subject()
    calibration = calibration or Calibration()
    p = _Pts(np.asarray(world, dtype=float), np.asarray(visibility, dtype=float), min_visibility)
    m: dict = {}

    up = -unit(np.asarray(gravity, dtype=float))
    down = -up

    torso_ok = p.ok("l_shoulder", "r_shoulder", "l_hip", "r_hip")
    head_ok = p.ok("l_ear", "r_ear") or p.ok("nose")
    m["torso_visible"] = torso_ok
    if not torso_ok:
        return m

    sh_mid = p.mid("l_shoulder", "r_shoulder")
    hip_mid = p.mid("l_hip", "r_hip")
    ear_mid = p.mid("l_ear", "r_ear") if p.ok("l_ear", "r_ear") else p["nose"]

    # ---- fall-detection features (computed first: the body frame below degenerates when lying) ----
    m["trunk_inclination"] = angle_between(sh_mid - hip_mid, up)  # 0 upright, 90 lying
    m["hand_raised"] = any(
        p.ok(f"{s}_wrist", f"{s}_shoulder") and float(np.dot(p[f"{s}_wrist"] - p[f"{s}_shoulder"], up)) > 0.15
        for s in ("l", "r")
    )

    # ---- body frame: U (up), F (forward), L (person's left) ----
    lateral = (p["l_hip"] - p["r_hip"]) + (p["l_shoulder"] - p["r_shoulder"])
    fwd = unit(horizontal(np.cross(lateral, up), up))
    if head_ok and p.ok("nose"):
        nose_dir = horizontal(p["nose"] - ear_mid, up)
        if np.linalg.norm(nose_dir) > 0.02 and np.dot(nose_dir, fwd) < 0:
            fwd = -fwd  # left/right landmarks swapped (e.g. person facing away); trust the nose
    if np.linalg.norm(fwd) < 1e-6:
        return m
    left = np.cross(up, fwd)

    # ---- trunk ----
    trunk = sh_mid - hip_mid
    m["trunk_flexion"] = plane_angle(trunk, up, fwd)
    m["trunk_side_bend"] = plane_angle(trunk, up, left)
    m["trunk_twist"] = wrap180(
        signed_angle_about(p["l_hip"] - p["r_hip"], p["l_shoulder"] - p["r_shoulder"], up)
    )
    m["shoulder_tilt_cm"] = float(np.dot(p["l_shoulder"] - p["r_shoulder"], up)) * 100.0

    # ---- neck / head ----
    if head_ok:
        head = ear_mid - sh_mid
        m["neck_flexion"] = plane_angle(head, up, fwd) - m["trunk_flexion"]
        m["neck_side_bend"] = plane_angle(head, up, left) - m["trunk_side_bend"]
        m["head_forward_cm"] = float(np.dot(head, fwd)) * 100.0
        if p.ok("l_ear", "r_ear"):
            m["neck_twist"] = wrap180(
                signed_angle_about(p["l_shoulder"] - p["r_shoulder"], p["l_ear"] - p["r_ear"], up)
            )
        for side in ("l", "r"):
            if p.ok(f"{side}_ear", f"{side}_shoulder"):
                m[f"{side}_ear_shoulder_cm"] = (
                    float(np.dot(p[f"{side}_ear"] - p[f"{side}_shoulder"], up)) * 100.0
                )

    # ---- calibration (neutral posture offsets) ----
    if apply_calibration:
        for key in CALIBRATED_KEYS:
            if key in m and key in calibration.offsets:
                m[key] -= calibration.offsets[key]
    for side in ("l", "r"):
        base = calibration.ear_shoulder_cm.get(side)
        cur = m.get(f"{side}_ear_shoulder_cm")
        m[f"{side}_shoulder_raised"] = bool(base and cur is not None and base > 1 and cur < 0.82 * base)

    # ---- arms (relative to trunk) ----
    t_down = unit(hip_mid - sh_mid)
    f_t = unit(fwd - np.dot(fwd, t_down) * t_down)
    l_t = unit(np.cross(t_down, f_t) * -1.0)  # person's left, orthogonal to trunk axis
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        if p.ok(sh, el):
            a = p[el] - p[sh]
            elev = angle_between(a, t_down)
            flex = plane_angle(a, t_down, f_t)
            # abduction = how far the arm points out of the sagittal plane toward its own side
            abd = math.degrees(math.asin(float(np.clip(np.dot(unit(a), sgn * l_t), -1.0, 1.0))))
            extension = bool(np.dot(a, f_t) < 0 and np.dot(a, t_down) > 0 and abs(abd) < 30)
            m[f"{side}_upper_arm_elevation"] = elev
            m[f"{side}_upper_arm_flexion"] = flex
            m[f"{side}_upper_arm_abduction"] = abd
            m[f"{side}_upper_arm_abducted"] = bool(abd > 20.0)
            # RULA/REBA upper-arm angle: negative means extension (arm behind the trunk)
            m[f"{side}_upper_arm_angle"] = -elev if extension else elev
        if p.ok(sh, el, wr):
            m[f"{side}_elbow_flexion"] = 180.0 - angle_between(p[sh] - p[el], p[wr] - p[el])
            m[f"{side}_wrist_across_midline"] = bool(np.dot(p[wr] - sh_mid, sgn * l_t) < -0.03)
            idx, pky = f"{side}_index", f"{side}_pinky"
            if p.ok(idx, pky):
                hand = (p[idx] + p[pky]) / 2.0 - p[wr]
                m[f"{side}_wrist_deviation"] = angle_between(p[wr] - p[el], hand)

    # ---- legs ----
    for side in ("l", "r"):
        sh, hp, kn, an = f"{side}_shoulder", f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        if p.ok(sh, hp, kn):
            m[f"{side}_hip_flexion"] = 180.0 - angle_between(p[sh] - p[hp], p[kn] - p[hp])
        if p.ok(hp, kn, an):
            m[f"{side}_knee_flexion"] = 180.0 - angle_between(p[hp] - p[kn], p[an] - p[kn])
    if p.ok("l_ankle", "r_ankle"):
        m["ankle_height_diff_cm"] = abs(float(np.dot(p["l_ankle"] - p["r_ankle"], up))) * 100.0

    hips = [m[k] for k in ("l_hip_flexion", "r_hip_flexion") if k in m]
    knees = [m[k] for k in ("l_knee_flexion", "r_knee_flexion") if k in m]
    if hips and knees and np.mean(hips) > 55 and np.mean(knees) > 45:
        posture = "sitting"
    elif hips and knees and np.mean(hips) < 35:
        posture = "standing"
    elif hips and np.mean(hips) > 60:
        posture = "sitting"
    else:
        posture = "unknown"
    m["posture"] = posture
    m["legs_supported"] = bool(
        posture == "sitting" or m.get("ankle_height_diff_cm") is None or m["ankle_height_diff_cm"] < 10.0
    )

    # ---- maxima over sides (handy for rules / RULA / REBA) ----
    for base in ("upper_arm_elevation", "elbow_flexion", "wrist_deviation", "knee_flexion", "hip_flexion"):
        vals = [m[f"{s}_{base}"] for s in ("l", "r") if f"{s}_{base}" in m]
        if vals:
            m[f"max_{base}"] = max(vals)

    # ---- static biomechanical load model ----
    m.update(
        _loads(p, sh_mid, hip_mid, ear_mid, head_ok, up, down, left, t_down, subject, m["trunk_flexion"])
    )

    return {k: (_r(v) if isinstance(v, float) else v) for k, v in m.items()}


def _loads(
    p, sh_mid, hip_mid, ear_mid, head_ok, up, down, left, t_down, subject: Subject, trunk_flex: float
) -> dict:
    out: dict = {}
    bm = subject.body_mass_kg
    l5 = hip_mid + L5S1_FROM_HIP * (sh_mid - hip_mid)

    def moment(points_masses, origin) -> np.ndarray:
        total = np.zeros(3)
        for r, mass in points_masses:
            total += np.cross(r - origin, mass * G * down)
        return total

    segments: list[tuple[np.ndarray, float]] = []
    head_com = ear_mid if head_ok else sh_mid + 0.25 * unit(sh_mid - hip_mid)
    segments.append((head_com, MASS_HEAD_NECK * bm))
    segments.append((l5 + TRUNK_COM_FROM_L5 * (sh_mid - l5), MASS_TRUNK_ABOVE_L5 * bm))

    shoulder_moments = []
    for side in ("l", "r"):
        sh, el, wr = p[f"{side}_shoulder"], p[f"{side}_elbow"], p[f"{side}_wrist"]
        if p.ok(f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
            hand = (
                (p[f"{side}_index"] + p[f"{side}_pinky"]) / 2.0
                if p.ok(f"{side}_index", f"{side}_pinky")
                else wr
            )
            arm = [
                (sh + COM_UPPER_ARM * (el - sh), MASS_UPPER_ARM * bm),
                (el + COM_FOREARM * (wr - el), MASS_FOREARM * bm),
                ((wr + hand) / 2.0, MASS_HAND * bm),
            ]
            if subject.hand_load_kg > 0:
                arm.append((hand, subject.hand_load_kg))
            shoulder_moments.append(float(np.linalg.norm(moment(arm, sh))))
        else:  # assume arm hanging along the trunk
            arm = [
                (sh + 0.3 * t_down, (MASS_UPPER_ARM + MASS_FOREARM + MASS_HAND) * bm + subject.hand_load_kg)
            ]
        segments.extend(arm)

    m_l5 = moment(segments, l5)
    flex_moment = float(np.dot(m_l5, left))
    w_upper = sum(mass for _, mass in segments) * G
    trunk_rad = math.radians(trunk_flex)
    out["l5s1_moment_nm"] = flex_moment
    out["l5s1_compression_n"] = max(flex_moment, 0.0) / ERECTOR_MOMENT_ARM_M + w_upper * max(
        math.cos(trunk_rad), 0.0
    )
    out["l5s1_shear_n"] = abs(w_upper * math.sin(trunk_rad))
    if shoulder_moments:
        out["max_shoulder_moment_nm"] = max(shoulder_moments)

    if head_ok:
        w_head = MASS_HEAD_NECK * bm * G
        neck_moment = float(np.dot(np.cross(ear_mid - sh_mid, w_head * down), left))
        out["neck_moment_nm"] = neck_moment
        out["neck_load_kg"] = (max(neck_moment, 0.0) / NECK_EXTENSOR_MOMENT_ARM_M + w_head) / G
    return out


def calibration_from_samples(samples: list[dict]) -> Calibration:
    """Build a neutral-posture calibration from metrics computed with ``apply_calibration=False``."""
    cal = Calibration()
    for key in CALIBRATED_KEYS:
        vals = [s[key] for s in samples if s.get(key) is not None]
        if len(vals) >= 5:
            cal.offsets[key] = round(float(np.median(vals)), 2)
    for side in ("l", "r"):
        vals = [s[f"{side}_ear_shoulder_cm"] for s in samples if s.get(f"{side}_ear_shoulder_cm") is not None]
        if len(vals) >= 5:
            cal.ear_shoulder_cm[side] = round(float(np.median(vals)), 2)
    return cal
