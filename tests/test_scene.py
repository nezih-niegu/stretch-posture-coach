import json
import math
import shutil

import numpy as np
import pytest
from scene_synth import person_pose, render

from stretch_posture_coach.biomech import gravity_from_pitch
from stretch_posture_coach.camera import Intrinsics
from stretch_posture_coach.scene import (
    HeadKinematics,
    SceneRecorder,
    build_scene,
    export_oneslam,
    import_oneslam,
    inv_T,
    kinematic_poses,
    load_frames,
    make_T,
    map_T_base,
    read_tum,
    rot_z,
    write_tum,
)

INTR = Intrinsics.from_fov(240, 320, 42)
KIN = HeadKinematics()


def record(tmp, script, hz=1000):
    """script: list of (pan, tilt, x, y, theta). Returns the session dir."""
    sess = tmp / "sess"
    shutil.rmtree(sess, ignore_errors=True)
    sess.mkdir(parents=True)
    rec = SceneRecorder(sess, {"record_hz": hz, "depth_decimation": 1})
    for k, (pan, tilt, x, y, th) in enumerate(script):
        T = map_T_base(x, y, th) @ KIN.base_T_cam(pan, tilt)
        z, bgr, _ = render(T, INTR)
        rec.maybe_record(float(k), bgr, z, INTR, pan, tilt, (x, y, th), person_pose(T, INTR), "standing")
    rec.close()
    return sess


SWEEP = [(math.radians(-60 + 120 * (k % 20) / 19), -0.25, 0.0, 0.0, 0.0) for k in range(20)]
DRIVE = [(math.radians(-30 + 60 * (k % 10) / 9), -0.25, 0.04 * k, 0.0, 0.0) for k in range(30)]


def test_kinematics_consistent_with_gravity_model():
    for tilt in (-0.6, -0.25, 0.0, 0.2):
        T = KIN.base_T_cam(0.3, tilt)
        g_cam = T[:3, :3].T @ np.array([0, 0, -1.0])  # world down in optical coordinates
        assert np.allclose(g_cam, gravity_from_pitch(math.degrees(tilt)), atol=1e-9)
    T = KIN.base_T_cam(0.0, 0.0)
    assert np.allclose(T[:3, 2], [1, 0, 0]) and T[2, 3] == pytest.approx(
        1.3
    )  # looks forward, at camera height
    assert np.allclose(KIN.base_T_cam(math.pi / 2, 0)[:3, 2], [0, 1, 0], atol=1e-9)  # pan +90 deg looks left


def test_tum_roundtrip(tmp_path):
    poses = {i: make_T(rot_z(0.1 * i), [i, -i, 0.5]) for i in range(5)}
    write_tum(tmp_path / "p.txt", poses)
    back = read_tum(tmp_path / "p.txt")
    assert all(np.allclose(back[i], poses[i], atol=1e-6) for i in poses)


def test_scene_map_from_head_sweep(tmp_path):
    sess = record(tmp_path, SWEEP)
    out = build_scene(sess, {})
    assert out["source"] == "kinematic+floor" and out["frames"] == 20  # old sessions get the floor correction
    assert out["extent_m"][1] == pytest.approx(5.0, abs=0.15)  # room is 5 m wide
    assert 0.5 < out["obstacle_area_m2"] < 2.5 and out["free_floor_m2"] > 8
    p = out["person"]
    assert p["samples"] > 5 and p["clearance_mean_m"] == pytest.approx(1.4, abs=0.15)  # nearest: table corner
    assert p["distance_to_robot_mean_m"] == pytest.approx(2.35, abs=0.25)
    for f in ("map.png", "scene.ply", "points.bin", "scene.json"):
        assert (sess / "scene" / f).exists()
    head = (sess / "scene" / "scene.ply").read_bytes()[:200]
    assert b"element vertex" in head and b"binary_little_endian" in head


def test_oneslam_export_matches_upstream_loader(tmp_path):
    sess = record(tmp_path, SWEEP[:6])
    out = export_oneslam(sess, with_depths=True)
    imgs = sorted((out / "images").glob("*.jpg"))
    # upstream: image_dict[int(str(path)[-12:-4])] = path ; frames must be 0..N-1
    assert [int(str(p)[-12:-4]) for p in imgs] == list(range(6))
    cal = json.loads((out / "calibration.json").read_text())
    assert set(cal["intrinsics"]) == {"fx", "fy", "cx", "cy"} and "fps" in cal
    import cv2

    d = cv2.imread(str(out / "depths" / "00000000.png"), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d.shape == (INTR.h, INTR.w)  # same size as images, millimetres
    notes = json.loads((out / "stretch_export.json").read_text())
    assert "pure rotation" in notes["warning"]  # head sweep only: monocular SLAM would be degenerate
    assert notes["suggested_args"].startswith("--depth_scale")
    assert not (out / "poses_gt.txt").exists()  # kinematic poses are a reference, not ground truth


def _fake_oneslam(sess, scale, drift_per_frame=0.0, keyframe_every=1):
    """OneSLAM-style output: camera-to-world in its own first-camera frame, arbitrary scale."""
    frames = load_frames(sess)
    kp = kinematic_poses(frames, KIN)
    T0inv = inv_T(kp[0])
    os = {}
    for f in frames[::keyframe_every]:
        rel = T0inv @ kp[f["i"]]
        t = rel[:3, 3] * scale + np.array([drift_per_frame * f["i"], 0, 0])
        os[f["i"]] = make_T(rel[:3, :3], t)
    path = sess / "poses_pred.txt"
    write_tum(path, os)
    return path


def test_oneslam_import_recovers_metric_scale_and_base_poses(tmp_path):
    sess = record(tmp_path, DRIVE)
    export_oneslam(sess)
    m = import_oneslam(sess, _fake_oneslam(sess, scale=0.37, keyframe_every=2))
    assert m["metric"] and m["scale"]["scale"] == pytest.approx(1 / 0.37, rel=0.03)
    assert m["cam_vs_kinematic_rmse_m"] < 0.05
    base = read_tum(sess / "scene" / "oneslam" / "poses_base_map.txt")
    frames = {f["i"]: f for f in load_frames(sess)}
    for k, T in base.items():  # base poses from camera poses + head joint states = the true base path
        assert T[0, 3] == pytest.approx(frames[k]["base"][0], abs=0.05) and abs(T[1, 3]) < 0.05
    assert m["head_only_check"] is None  # this run never stood still with a moving head
    out = build_scene(sess, {})
    assert out["source"] == "oneslam+kinematic+floor" and out["oneslam"]["metric"]


def test_head_only_check_passes_and_fails(tmp_path):
    script = [(math.radians(-40 + 8 * k), -0.25, 0.0, 0.0, 0.0) for k in range(10)]  # head only
    script += [(0.0, -0.25, 0.05 * k, 0.0, 0.0) for k in range(1, 15)]  # then drive
    sess = record(tmp_path, script)
    export_oneslam(sess)
    good = import_oneslam(sess, _fake_oneslam(sess, scale=2.0))
    assert good["head_only_segments"] == 1 and good["head_only_check"] == "pass"
    bad = import_oneslam(sess, _fake_oneslam(sess, scale=2.0, drift_per_frame=0.06))
    assert bad["head_only_check"].startswith("fail")


def test_rotation_only_session_has_no_metric_scale(tmp_path):
    sess = record(tmp_path, SWEEP)
    export_oneslam(sess)
    m = import_oneslam(sess, _fake_oneslam(sess, scale=0.5))
    # the head's 5 cm lever arm gives almost no baseline: scale may or may not be found, but never wrongly
    if m["metric"]:
        assert m["scale"]["scale"] == pytest.approx(2.0, rel=0.15)
    else:
        assert "could not be recovered" in m["note"]


def test_recorder_rate_limit_and_cap(tmp_path):
    rec = SceneRecorder(tmp_path, {"record_hz": 2, "max_frames": 3})
    z, bgr, _ = render(map_T_base(0, 0, 0) @ KIN.base_T_cam(0, -0.25), INTR)
    took = [rec.maybe_record(t, bgr, z, INTR, 0.0, -0.25, (0, 0, 0)) for t in np.arange(0, 5, 0.1)]
    rec.close()
    assert sum(took) == 3 and len(load_frames(tmp_path)) == 3


def test_cli_export_import_and_web_routes(tmp_path, monkeypatch):
    import urllib.request

    from typer.testing import CliRunner

    from stretch_posture_coach.cli import app
    from stretch_posture_coach.stream import FrameStreamer

    sessions = tmp_path / "sessions"
    sess = record(tmp_path, DRIVE)
    target = sessions / "20261114_160000"
    sessions.mkdir()
    shutil.move(str(sess), target)
    (target / "metrics.csv").write_text("t,wall_time,present\n0,0,1\n1,1,1\n", encoding="utf-8")
    monkeypatch.setenv("POSTURE_COACH_HOME", str(tmp_path))
    r = CliRunner().invoke(app, ["scene", "export-oneslam", "20261114_160000"])
    assert r.exit_code == 0, r.output
    assert "run_oneslam.sh" in r.output and (target / "scene" / "oneslam_input" / "images").is_dir()
    poses = _fake_oneslam(target, scale=0.5, keyframe_every=3)
    r = CliRunner().invoke(app, ["scene", "import-oneslam", "20261114_160000", str(poses)])
    assert r.exit_code == 0, r.output
    html = (target / "report.html").read_text(encoding="utf-8")
    assert "Scene understanding" in html and "OneSLAM (camera-only localization)" in html
    assert "/sessions/20261114_160000/scene3d" in html

    srv = FrameStreamer(port=0, sessions_dir=sessions)
    base = f"http://127.0.0.1:{srv.port}"
    try:

        def get(path):
            with urllib.request.urlopen(base + path, timeout=5) as resp:
                return resp.status, resp.headers.get("Content-Type"), resp.read()

        assert get("/sessions/20261114_160000/scene/map.png")[2][:4] == b"\x89PNG"
        assert b"element vertex" in get("/sessions/20261114_160000/scene/scene.ply")[2][:200]
        assert b"points.bin" in get("/sessions/20261114_160000/scene3d")[2]
        assert len(get("/vendor/three.module.js")[2]) > 100_000
        import urllib.error

        for bad in ("/vendor/../pyproject.toml", "/sessions/20261114_160000/scene/../config.yaml"):
            try:
                code = get(bad)[0]
            except urllib.error.HTTPError as e:
                code = e.code
            assert code == 404
    finally:
        srv.close()
