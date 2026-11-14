import numpy as np
import pytest
from synth import skeleton

from stretch_posture_coach.biomech import (
    Subject,
    calibration_from_samples,
    compute_metrics,
    gravity_from_pitch,
)

G0 = gravity_from_pitch(0.0)


def metrics(pose, pitch=0.0, **kw):
    return compute_metrics(pose.world, pose.visibility, gravity_from_pitch(pitch), **kw)


@pytest.mark.parametrize("view", ["front", "side"])
@pytest.mark.parametrize("trunk", [0, 15, 40])
@pytest.mark.parametrize("neck", [0, 25])
def test_trunk_and_neck_flexion(view, trunk, neck):
    m = metrics(skeleton(trunk_flex=trunk, neck_flex=neck, view=view))
    assert m["trunk_flexion"] == pytest.approx(trunk, abs=0.6)
    assert m["neck_flexion"] == pytest.approx(neck, abs=0.6)
    assert abs(m["trunk_side_bend"]) < 0.6
    assert abs(m["trunk_twist"]) < 0.6


def test_camera_pitch_is_compensated():
    pose = skeleton(trunk_flex=20, neck_flex=10, camera_pitch_deg=-30)
    assert metrics(pose, pitch=-30)["trunk_flexion"] == pytest.approx(20, abs=0.6)
    # without compensation the reading is wrong by roughly the pitch
    assert abs(metrics(pose, pitch=0)["trunk_flexion"] - 20) > 15


@pytest.mark.parametrize("arm,elbow", [(0, 90), (45, 80), (100, 30)])
def test_arm_angles(arm, elbow):
    m = metrics(skeleton(arm_flex=arm, elbow_flex=elbow))
    for s in "lr":
        assert m[f"{s}_upper_arm_elevation"] == pytest.approx(arm, abs=0.6)
        assert m[f"{s}_elbow_flexion"] == pytest.approx(elbow, abs=0.6)
        assert m[f"{s}_upper_arm_abducted"] is False


def test_abduction_detected():
    m = metrics(skeleton(abduct=45))
    assert m["l_upper_arm_abducted"] and m["r_upper_arm_abducted"]
    assert m["l_upper_arm_abduction"] == pytest.approx(45, abs=1)


def test_arm_extension_is_negative():
    m = metrics(skeleton(arm_flex=-40))
    assert m["l_upper_arm_angle"] == pytest.approx(-40, abs=1)


def test_side_bend_sign():
    m = metrics(skeleton(trunk_side=15))
    assert m["trunk_side_bend"] == pytest.approx(15, abs=1.0)


def test_sitting_vs_standing():
    assert metrics(skeleton(sitting=True))["posture"] == "sitting"
    assert metrics(skeleton(sitting=False))["posture"] == "standing"


def test_lumbar_load_increases_with_flexion_and_load():
    upright = metrics(skeleton(trunk_flex=0), subject=Subject(70))
    bent = metrics(skeleton(trunk_flex=45), subject=Subject(70))
    loaded = metrics(skeleton(trunk_flex=45), subject=Subject(70, hand_load_kg=10))
    assert upright["l5s1_compression_n"] < bent["l5s1_compression_n"] < loaded["l5s1_compression_n"]
    assert 300 < upright["l5s1_compression_n"] < 900  # roughly upper-body weight
    assert loaded["l5s1_compression_n"] > 3000  # 20 kg held while stooped ~ NIOSH action limit (3400 N)
    assert bent["l5s1_shear_n"] > upright["l5s1_shear_n"]


def test_neck_load_increases_with_flexion():
    a = metrics(skeleton(neck_flex=0))["neck_load_kg"]
    b = metrics(skeleton(neck_flex=45))["neck_load_kg"]
    assert b > a * 2


def test_invisible_torso():
    pose = skeleton()
    vis = pose.visibility.copy()
    vis[[11, 12]] = 0.1
    m = compute_metrics(pose.world, vis, G0)
    assert m == {"torso_visible": False}


def test_calibration_offsets_applied():
    # a person whose "neutral" reads as 8 deg neck flexion
    samples = [
        metrics(skeleton(neck_flex=8 + np.random.uniform(-1, 1)), apply_calibration=False) for _ in range(20)
    ]
    cal = calibration_from_samples(samples)
    assert cal.offsets["neck_flexion"] == pytest.approx(8, abs=1)
    m = metrics(skeleton(neck_flex=8), calibration=cal)
    assert abs(m["neck_flexion"]) < 1.0
