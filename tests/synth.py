"""Synthetic MediaPipe-style skeletons with known joint angles, for tests and dry runs."""

from __future__ import annotations

import math

import numpy as np

from stretch_posture_coach.pose import PoseResult


def _rot(v_fu, deg):
    """Rotate a (f, u) direction forward by ``deg`` in the sagittal plane."""
    r = math.radians(deg)
    f, u = v_fu
    return (f * math.cos(r) + u * math.sin(r), -f * math.sin(r) + u * math.cos(r))


def skeleton(
    trunk_flex=0.0,
    neck_flex=0.0,
    arm_flex=0.0,
    elbow_flex=90.0,
    sitting=False,
    view="front",
    trunk_side=0.0,
    camera_pitch_deg=0.0,
    abduct=0.0,
) -> PoseResult:
    """Build 33 world landmarks. Body frame (f, l, u), mapped into an upright camera frame.

    view='front': person faces the camera. view='side': person faces image-right.
    camera_pitch_deg: camera pitched down (negative) -> gravity no longer along image y.
    """
    pts = np.zeros((33, 3))  # columns: f, l, u

    def put(i, f, lat, u):
        pts[i] = (f, lat, u)

    trunk_dir = _rot((0.0, 1.0), trunk_flex)  # (f, u)
    # lateral bend of trunk: tilt toward +l
    sb = math.radians(trunk_side)
    sh_mid = np.array([0.5 * trunk_dir[0], 0.5 * math.sin(sb), 0.5 * trunk_dir[1] * math.cos(sb)])
    put(23, 0, 0.1, 0)
    put(24, 0, -0.1, 0)
    pts[11] = sh_mid + [0, 0.18, 0]
    pts[12] = sh_mid + [0, -0.18, 0]
    head_dir = _rot(trunk_dir, neck_flex)
    ear_mid = sh_mid + 0.2 * np.array([head_dir[0], 0, head_dir[1]])
    pts[7] = ear_mid + [0, 0.07, 0]
    pts[8] = ear_mid + [0, -0.07, 0]
    nose_dir = _rot(head_dir, 90)  # forward of the head
    pts[0] = ear_mid + 0.1 * np.array([nose_dir[0], 0, nose_dir[1]])
    # arms
    t_down = (-trunk_dir[0], -trunk_dir[1])
    ua = _rot(t_down, -arm_flex)  # negative rotation of a downward vector moves it forward
    fa = _rot(ua, -elbow_flex)
    for sh_i, el_i, wr_i, pk_i, ix_i, sgn in ((11, 13, 15, 17, 19, 1), (12, 14, 16, 18, 20, -1)):
        ab = math.radians(abduct)
        ua_vec = np.array([ua[0] * math.cos(ab), sgn * math.sin(ab), ua[1] * math.cos(ab)])
        pts[el_i] = pts[sh_i] + 0.3 * ua_vec
        fa_vec = np.array([fa[0], 0, fa[1]])
        pts[wr_i] = pts[el_i] + 0.25 * fa_vec
        pts[pk_i] = pts[wr_i] + 0.08 * fa_vec + [0, -0.02 * sgn, 0]
        pts[ix_i] = pts[wr_i] + 0.08 * fa_vec + [0, 0.02 * sgn, 0]
    # legs
    for kn, an, lat in ((25, 27, 0.1), (26, 28, -0.1)):
        if sitting:
            put(kn, 0.45, lat, 0)
            put(an, 0.45, lat, -0.45)
        else:
            put(kn, 0, lat, -0.45)
            put(an, 0, lat, -0.9)

    if view == "front":
        F, L, U = np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
    else:
        F, U = np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
        L = np.cross(U, F)
    world = pts[:, :1] * F + pts[:, 1:2] * L + pts[:, 2:3] * U
    # camera pitched down by p: rotate world about the camera x axis so gravity is (0, cos, sin)
    th = math.radians(-camera_pitch_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    world = world @ rx.T
    vis = np.ones(33)
    vis[[1, 2, 3, 4, 5, 6, 9, 10, 29, 30, 31, 32]] = 0.9
    img = np.column_stack([0.5 + world[:, 0] / 2.5, 0.5 + world[:, 1] / 2.5])
    return PoseResult(image=img, world=world, visibility=vis)
