"""Overlay rendering: color-coded skeleton + metrics side panel."""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from .assessment import Assessment
from .pose import POSE_EDGES, PoseResult

GREEN, YELLOW, RED, GREY, WHITE = (
    (80, 200, 80),
    (0, 200, 255),
    (60, 60, 235),
    (150, 150, 150),
    (240, 240, 240),
)

NECK_EDGES = {(0, 7), (0, 8), (9, 10)}
TRUNK_EDGES = {(11, 12), (11, 23), (12, 24), (23, 24)}
ARM_EDGES = {
    "l": {(11, 13), (13, 15), (15, 17), (15, 19), (17, 19)},
    "r": {(12, 14), (14, 16), (16, 18), (16, 20), (18, 20)},
}


def _level(value, threshold) -> tuple:
    if value is None:
        return GREY
    r = abs(value) / threshold if threshold else 0
    return GREEN if r < 0.7 else (YELLOW if r < 1.0 else RED)


def _edge_color(edge, m: dict, th: dict):
    if edge in NECK_EDGES:
        return _level(m.get("neck_flexion"), th["neck_flexion"])
    if edge in TRUNK_EDGES:
        return _level(m.get("trunk_flexion"), th["trunk_flexion"])
    for side, edges in ARM_EDGES.items():
        if edge in edges:
            return _level(m.get(f"{side}_upper_arm_elevation"), th["upper_arm_elevation"])
    return (200, 160, 60)


def draw(
    frame: np.ndarray,
    pose: PoseResult | None,
    a: Assessment | None,
    th: dict,
    status: dict,
    min_vis: float = 0.5,
    panel_w: int = 300,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    m = a.metrics if a else {}
    if pose is not None:
        pts = (pose.image * [w, h]).astype(int)
        for i, j in POSE_EDGES:
            if pose.visibility[i] >= min_vis and pose.visibility[j] >= min_vis:
                cv2.line(img, tuple(pts[i]), tuple(pts[j]), _edge_color((i, j), m, th), 3, cv2.LINE_AA)
        for k, (x, y) in enumerate(pts):
            if pose.visibility[k] >= min_vis and k not in range(1, 7):
                cv2.circle(img, (int(x), int(y)), 4, WHITE, -1, cv2.LINE_AA)

    panel = np.full((h, panel_w, 3), 28, np.uint8)
    y = 24

    def put(text, color=WHITE, scale=0.5, bold=1, dy=20):
        nonlocal y
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)
        y += dy

    put("STRETCH POSTURE COACH", (255, 210, 120), 0.55, 2, 26)
    put(f"FPS {status.get('fps', 0):.1f}   head tilt {status.get('tilt_deg', '-')}", GREY, 0.42)
    fs = status.get("follow")
    if fs is not None and fs.distance is not None:
        in_band = 2.0 <= fs.distance <= 3.0
        txt = f"Distance {fs.distance:.2f} m ({fs.source})"
        if status.get("follow_enabled"):
            txt += f"  [{fs.action}]"
        put(txt, GREEN if in_band else YELLOW, 0.45, 1, 20)
    if not a or not a.present:
        put("No person detected", YELLOW, 0.55, 1, 26)
        put("Stand 2-3 m from the robot,", GREY, 0.45)
        put("whole body in view.", GREY, 0.45)
    else:
        put(f"Posture: {m.get('posture', '?')}", WHITE, 0.5, 1, 24)
        if a.rula:
            c = GREEN if a.rula["score"] <= 2 else (YELLOW if a.rula["score"] <= 4 else RED)
            put(f"RULA {a.rula['score']}  ({a.rula['risk']})", c, 0.55, 2, 22)
        if a.reba:
            c = GREEN if a.reba["score"] <= 3 else (YELLOW if a.reba["score"] <= 7 else RED)
            put(f"REBA {a.reba['score']}  ({a.reba['risk']})", c, 0.55, 2, 26)

        def row(label, key, unit="deg", thkey=None):
            v = m.get(key)
            col = _level(v, th[thkey]) if thkey and v is not None else WHITE
            put(f"{label:<14}{'-' if v is None else f'{v:6.0f}'} {unit}", col, 0.45, 1, 18)

        row("Neck flex", "neck_flexion", thkey="neck_flexion")
        row("Neck side", "neck_side_bend", thkey="neck_side_bend")
        row("Trunk flex", "trunk_flexion", thkey="trunk_flexion")
        row("Trunk side", "trunk_side_bend", thkey="trunk_side_bend")
        row("Trunk twist", "trunk_twist", thkey="trunk_twist")
        row("Shoulder tilt", "shoulder_tilt_cm", "cm", "shoulder_tilt_cm")
        row("Arm elev max", "max_upper_arm_elevation", thkey="upper_arm_elevation")
        row("Elbow max", "max_elbow_flexion")
        row("Knee max", "max_knee_flexion")
        y += 4
        row("L5/S1 comp", "l5s1_compression_n", "N", "l5s1_compression_n")
        row("Neck load", "neck_load_kg", "kg")
        if a.static:
            put(f"Static posture {a.static_for_s / 60:.1f} min", YELLOW, 0.45, 1, 20)
        y += 6
        if a.active:
            put("Issues:", RED, 0.5, 2, 20)
            for st in a.active[:4]:
                put(f"- {st.rule.label}", RED, 0.42, 1, 17)
    msg = status.get("last_message")
    if msg:
        y = max(y + 8, h - 110)
        put("Stretch says:", (255, 210, 120), 0.45, 1, 18)
        for line in textwrap.wrap(msg, 38)[:5]:
            put(line, WHITE, 0.42, 1, 16)
    return np.hstack([img, panel])
