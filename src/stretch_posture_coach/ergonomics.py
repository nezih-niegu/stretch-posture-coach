"""RULA (McAtamney & Corlett 1993) and REBA (Hignett & McAtamney 2000) posture scores.

Scores are computed per side from the metrics produced by :mod:`biomech`; the worse side is reported.
Force/load, coupling and wrist-twist inputs are not observable from a camera, so they come from config
(``subject.hand_load_kg``) or default to the neutral value.
"""

from __future__ import annotations

# ---------------------------------------------------------------- RULA tables
# TABLE_A[upper_arm-1][lower_arm-1][(wrist-1)*2 + (twist-1)]
RULA_TABLE_A = [
    [[1, 2, 2, 2, 2, 3, 3, 3], [2, 2, 2, 2, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3, 4, 4]],
    [[2, 3, 3, 3, 3, 4, 4, 4], [3, 3, 3, 3, 3, 4, 4, 4], [3, 4, 4, 4, 4, 4, 5, 5]],
    [[3, 3, 4, 4, 4, 4, 5, 5], [3, 4, 4, 4, 4, 4, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5]],
    [[4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 5, 5, 5, 6, 6]],
    [[5, 5, 5, 5, 5, 6, 6, 7], [5, 6, 6, 6, 6, 7, 7, 7], [6, 6, 6, 7, 7, 7, 7, 8]],
    [[7, 7, 7, 7, 7, 8, 8, 9], [8, 8, 8, 8, 8, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9]],
]
# TABLE_B[neck-1][(trunk-1)*2 + (legs-1)]
RULA_TABLE_B = [
    [1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
    [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
    [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
    [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
    [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9],
]
# TABLE_C[min(arm_wrist,8)-1][min(neck_trunk_legs,7)-1]
RULA_TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5],
    [2, 2, 3, 4, 4, 5, 5],
    [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7],
]

# ---------------------------------------------------------------- REBA tables
# TABLE_A[neck-1][trunk-1][legs-1]
REBA_TABLE_A = [
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
    [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]
# TABLE_B[lower_arm-1][upper_arm-1][wrist-1]
REBA_TABLE_B = [
    [[1, 2, 2], [1, 2, 3], [3, 4, 5], [4, 5, 5], [6, 7, 8], [7, 8, 8]],
    [[1, 2, 3], [2, 3, 4], [4, 5, 5], [5, 6, 7], [7, 8, 8], [8, 9, 9]],
]
# TABLE_C[score_a-1][score_b-1]
REBA_TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]

TWIST_DEG = 15.0
SIDE_BEND_DEG = 10.0


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _get(m: dict, key: str, default: float = 0.0) -> float:
    v = m.get(key)
    return default if v is None else float(v)


def rula_risk(score: int) -> str:
    if score <= 2:
        return "acceptable"
    if score <= 4:
        return "investigate"
    if score <= 6:
        return "change soon"
    return "change now"


def reba_risk(score: int) -> str:
    if score <= 1:
        return "negligible"
    if score <= 3:
        return "low"
    if score <= 7:
        return "medium"
    if score <= 10:
        return "high"
    return "very high"


def _rula_neck(m: dict) -> int:
    n = _get(m, "neck_flexion")
    if n < -5:
        s = 4
    elif n <= 10:
        s = 1
    elif n <= 20:
        s = 2
    else:
        s = 3
    s += abs(_get(m, "neck_twist")) > 20
    s += abs(_get(m, "neck_side_bend")) > SIDE_BEND_DEG
    return _clamp(s, 1, 6)


def _rula_trunk(m: dict) -> int:
    t = _get(m, "trunk_flexion")
    sitting = m.get("posture") == "sitting"
    if abs(t) <= 5 or (sitting and -15 <= t <= 5):
        s = 1
    elif t <= 20:
        s = 2
    elif t <= 60:
        s = 3
    else:
        s = 4
    s += abs(_get(m, "trunk_twist")) > TWIST_DEG
    s += abs(_get(m, "trunk_side_bend")) > SIDE_BEND_DEG
    return _clamp(s, 1, 6)


def _upper_arm_base(angle: float) -> int:
    if -20 <= angle <= 20:
        return 1
    if angle < -20 or angle <= 45:
        return 2
    if angle <= 90:
        return 3
    return 4


def _side_arm_scores(m: dict, side: str, arms_supported: bool) -> tuple[int, int, int] | None:
    if f"{side}_upper_arm_angle" not in m:
        return None
    ua = _upper_arm_base(_get(m, f"{side}_upper_arm_angle"))
    ua += bool(m.get(f"{side}_shoulder_raised"))
    ua += bool(m.get(f"{side}_upper_arm_abducted"))
    ua -= arms_supported
    ua = _clamp(ua, 1, 6)
    el = m.get(f"{side}_elbow_flexion")
    la = 1 if el is None or 60 <= el <= 100 else 2
    wd = _get(m, f"{side}_wrist_deviation")
    wr = 1 if wd <= 5 else (2 if wd <= 15 else 3)
    return ua, la, wr


def rula(m: dict, arms_supported: bool = False, static: bool = False, load_kg: float = 0.0) -> dict | None:
    """Return RULA scores for the worse side, or None if the torso is not visible."""
    if not m.get("torso_visible") or "trunk_flexion" not in m:
        return None
    muscle = 1 if static else 0
    force = 0 if load_kg < 2 else (1 if load_kg <= 10 else 2)
    neck = _rula_neck(m)
    trunk = _rula_trunk(m)
    legs = 1 if m.get("legs_supported", True) else 2
    score_b = RULA_TABLE_B[neck - 1][(trunk - 1) * 2 + (legs - 1)]
    d = _clamp(score_b + muscle + force, 1, 7)

    best = None
    for side in ("l", "r"):
        s = _side_arm_scores(m, side, arms_supported)
        if s is None:
            continue
        ua, la, wr = s
        la = _clamp(la + bool(m.get(f"{side}_wrist_across_midline")), 1, 3)
        score_a = RULA_TABLE_A[ua - 1][la - 1][(wr - 1) * 2]  # wrist twist assumed mid-range (1)
        c = _clamp(score_a + muscle + force, 1, 8)
        grand = RULA_TABLE_C[c - 1][d - 1]
        cand = {"score": grand, "side": side, "upper_arm": ua, "lower_arm": la, "wrist": wr, "arm_score": c}
        if best is None or grand > best["score"]:
            best = cand
    if best is None:  # arms not visible: assume neutral arms
        c = _clamp(1 + muscle + force, 1, 8)
        best = {
            "score": RULA_TABLE_C[c - 1][d - 1],
            "side": None,
            "upper_arm": None,
            "lower_arm": None,
            "wrist": None,
            "arm_score": c,
        }
    best.update(
        {"neck": neck, "trunk": trunk, "legs": legs, "body_score": d, "risk": rula_risk(best["score"])}
    )
    return best


def reba(m: dict, arms_supported: bool = False, static: bool = False, load_kg: float = 0.0) -> dict | None:
    """Return REBA scores for the worse side, or None if the torso is not visible."""
    if not m.get("torso_visible") or "trunk_flexion" not in m:
        return None
    # trunk
    t = _get(m, "trunk_flexion")
    if abs(t) <= 5:
        trunk = 1
    elif -20 <= t <= 20:
        trunk = 2
    elif t <= 60:
        trunk = 3
    else:
        trunk = 4
    if t < -20:
        trunk = 3
    trunk += abs(_get(m, "trunk_twist")) > TWIST_DEG or abs(_get(m, "trunk_side_bend")) > SIDE_BEND_DEG
    trunk = _clamp(trunk, 1, 5)
    # neck
    n = _get(m, "neck_flexion")
    neck = 1 if -5 <= n <= 20 else 2
    neck += abs(_get(m, "neck_twist")) > 20 or abs(_get(m, "neck_side_bend")) > SIDE_BEND_DEG
    neck = _clamp(neck, 1, 3)
    # legs
    legs = 1 if m.get("legs_supported", True) else 2
    if m.get("posture") != "sitting":
        k = m.get("max_knee_flexion")
        if k is not None:
            legs += 2 if k > 60 else (1 if k >= 30 else 0)
    legs = _clamp(legs, 1, 4)
    load = 0 if load_kg < 5 else (1 if load_kg <= 10 else 2)
    score_a = REBA_TABLE_A[neck - 1][trunk - 1][legs - 1] + load

    best = None
    for side in ("l", "r"):
        s = _side_arm_scores(m, side, arms_supported)
        if s is None:
            continue
        ua, la, _ = s
        wd = _get(m, f"{side}_wrist_deviation")
        wr = 1 if wd <= 15 else 2
        score_b = REBA_TABLE_B[la - 1][ua - 1][wr - 1]  # coupling assumed good (0)
        c = REBA_TABLE_C[_clamp(score_a, 1, 12) - 1][_clamp(score_b, 1, 12) - 1]
        cand = {"score": c, "side": side, "upper_arm": ua, "lower_arm": la, "wrist": wr, "score_b": score_b}
        if best is None or c > best["score"]:
            best = cand
    if best is None:
        best = {
            "score": REBA_TABLE_C[_clamp(score_a, 1, 12) - 1][0],
            "side": None,
            "upper_arm": None,
            "lower_arm": None,
            "wrist": None,
            "score_b": 1,
        }
    best["score"] = _clamp(best["score"] + (1 if static else 0), 1, 15)
    best.update(
        {"neck": neck, "trunk": trunk, "legs": legs, "score_a": score_a, "risk": reba_risk(best["score"])}
    )
    return best
