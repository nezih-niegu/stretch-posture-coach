"""Lidar SLAM, floor registration, live map, posture heat map / captions, and the report's SLAM view."""

import json
import math
import time

import numpy as np
import pytest
import scene_synth
from vr_sim import Assess, SimRobot, person_at, room_segments

from stretch_posture_coach.camera import Intrinsics
from stretch_posture_coach.config import load_config
from stretch_posture_coach.insights import DIVERGING, diverging_rgb, rgb_hex
from stretch_posture_coach.lidar import SimLidar, _ScanAssembler, encode_node, parse_nodes
from stretch_posture_coach.registration import FloorAligner, map_T_cam, rot_z4
from stretch_posture_coach.scene import HeadKinematics, backproject, build_scene, map_T_base
from stretch_posture_coach.slam2d import Localizer, between, compose
from stretch_posture_coach.world import LiveMap, LiveWorld


def test_lidar_nodes_roundtrip_and_resync():
    raw = b"".join(encode_node(i == 0, 40, i * 1.0, 1.0 + i * 0.01) for i in range(360))
    raw += encode_node(True, 40, 0.0, 2.0)  # start of the next revolution closes the first
    noisy = bytearray(b"\x07" + raw[:500] + raw[500:])  # one junk byte in front
    nodes, used = parse_nodes(noisy)
    assert used == len(noisy) and len(nodes) == 361
    assert (
        nodes[0][0]
        and nodes[5][2] == pytest.approx(5.0, abs=0.02)
        and nodes[5][3] == pytest.approx(1.05, abs=1e-3)
    )
    asm = _ScanAssembler()
    scan = asm.feed(nodes, 1.0)
    assert scan is not None and len(scan.range_m) == 360


def test_slam_beats_biased_odometry():
    rng = np.random.default_rng(0)
    true, odom = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    sim = SimLidar(room_segments(), lambda: true)
    loc = Localizer({})
    true0 = None
    for k in range(100):
        v, w = 0.04, math.radians(1.5 if k < 50 else -2.0)
        true = compose(true, (v, 0, w))
        odom = compose(odom, (v * 1.08, 0.003, w * 1.1))
        s = sim.scan_once()
        s.range_m = s.range_m + rng.normal(0, 0.01, s.range_m.shape) * (s.range_m > 0)
        loc.on_scan(s, odom)
        true0 = true0 or true
    est, ref = loc.pose(odom), between(true0, true)
    odo = between((0, 0, 0), odom)
    assert math.hypot(est[0] - ref[0], est[1] - ref[1]) < 0.05
    assert abs(math.degrees(est[2] - ref[2])) < 1.0
    assert math.hypot(odo[0] - ref[0], odo[1] - ref[1]) > 0.3  # the odometry alone drifted a lot
    assert loc.status()["matched"] == 100


def _floor_layers(m):
    p, c = m.snapshot()
    fl = p[(c[:, 1] > 130) & (c[:, 0] < 100)]  # the green synthetic floor
    return len(np.unique(np.round(fl[:, 2], 3))), fl[:, 2].std()


def test_floor_plane_fixes_stacked_floors():
    """Encoder tilt off by 3 deg, camera rolled 1.5 deg and 6 cm higher than configured: the floor aligner
    measures all three and the fused floor collapses to one layer."""
    intr = Intrinsics.from_fov(120, 160, 42)
    kin, true_kin = HeadKinematics(1.3), HeadKinematics(1.36)
    al = FloorAligner(kin, {})
    raw, reg = LiveMap(0.05, 1), LiveMap(0.05, 1)
    for pan in np.linspace(-1.2, 1.2, 9):
        tilt = -0.45
        T_true = true_kin.base_T_cam(pan, tilt + math.radians(-3)) @ rot_z4(math.radians(1.5))
        z, col, _ = scene_synth.render(T_true, intr, with_person=False)
        pts, rc = backproject(z, (intr.fx, intr.fy, intr.cx, intr.cy), 2, None, 0.3, 4.5)
        al.observe(pts, pan, tilt)
        c = col[rc[:, 0], rc[:, 1], ::-1].astype(np.float32)
        for m, T in ((raw, kin.base_T_cam(pan, tilt)), (reg, map_T_cam((0, 0, 0), pan, tilt, al))):
            m.integrate(pts @ T[:3, :3].T + T[:3, 3], c)
    st = al.status()
    assert st["tilt_offset_deg"] == pytest.approx(-3.0, abs=0.3)
    assert st["roll_deg"] == pytest.approx(1.5, abs=0.3)
    assert st["height_offset_m"] == pytest.approx(0.06, abs=0.015)
    n_raw, sd_raw = _floor_layers(raw)
    n_reg, sd_reg = _floor_layers(reg)
    assert n_raw >= 4 and n_reg <= 2 and sd_reg < sd_raw / 2


def test_live_map_deltas_and_see_through_carving():
    m = LiveMap(0.05, min_hits=1)
    wall = np.array([[2.0, y, z] for y in np.arange(-0.5, 0.5, 0.05) for z in np.arange(0.1, 1.5, 0.05)])
    m.integrate(wall, np.full((len(wall), 3), 200.0))
    d0 = m.delta(0)
    assert d0["full"] and len(d0["idx"]) == len(m) and d0["rgb"][0, 0] == 200
    v0 = d0["version"]
    assert len(m.delta(v0)["idx"]) == 0  # nothing new
    # the camera at the origin now sees 4 m deep straight through where the wall was -> carved away
    intr = Intrinsics.from_fov(64, 48, 60)
    T = map_T_base(0, 0, 0) @ HeadKinematics(0.8, camera_forward_m=0).base_T_cam(0, 0)
    depth = np.full((48, 64), 4.0, np.float32)
    for _ in range(3):
        m.integrate(np.zeros((0, 3)), np.zeros((0, 3)), T, depth, (intr.fx, intr.fy, intr.cx, intr.cy), 6.0)
    d1 = m.delta(v0)
    assert len(d1["del"]) > 50 and not d1["full"]
    assert len(m) < len(wall) - 50


def test_heat_colors_follow_the_reference_diverging_palette():
    lo, mid, hi = DIVERGING["dark"]
    assert rgb_hex(diverging_rgb(-1, "dark")) == lo
    assert rgb_hex(diverging_rgb(1, "dark")) == hi
    assert rgb_hex(diverging_rgb(0, "dark")) == mid


def _run_sim_session(tmp_path, steps=48, tilt_offset_deg=-3.0):
    """Drive a simulated robot through a session with live registration; returns the session dir."""
    from stretch_posture_coach.logger import SessionLogger
    from stretch_posture_coach.scene import SceneRecorder

    cfg = load_config(overrides={"logging": {"sessions_dir": str(tmp_path)}, "vr": {"map_hz": 50}})
    robot = SimRobot()
    sim = SimLidar(room_segments(), lambda: robot.true)

    class _NoThreadLidar:  # scans are fed by hand (deterministic)
        on_scan = None

        def close(self):
            pass

    world = LiveWorld(cfg, robot, None, lidar=_NoThreadLidar())
    logger = SessionLogger(tmp_path, cfg, 50)
    world.new_session(logger.dir)
    rec = SceneRecorder(logger.dir, {"record_hz": 1000, "depth_decimation": 1})
    intr = Intrinsics.from_fov(120, 160, 42)
    true_kin = HeadKinematics(1.3)
    wall0 = time.time()
    rows = ["t,wall_time,present,posture,reba,neck_flexion"]
    for k in range(steps):
        t = k * 0.5
        if k % 6 == 5:
            robot.base_translate(0.15)
            robot.base_rotate(0.12)
        robot._pan = -0.5 + 1.0 * (k % 8) / 7 - robot.true[2]  # sweep around the person's side of the room
        world._on_scan(sim.scan_once())
        px, py = 2.0 + 0.4 * math.cos(k * 0.07), 0.6 * math.sin(k * 0.07)
        person_at(px, py)
        x, y, th = robot.true
        T_true = map_T_base(x, y, th) @ true_kin.base_T_cam(
            robot.pan, robot.tilt + math.radians(tilt_offset_deg)
        )
        depth, color, _ = scene_synth.render(T_true, intr)
        pose = scene_synth.person_pose(T_true, intr)
        reba = 9 if py > 0.3 else 3  # worse posture on one side of the room (e.g. at a low shelf)
        wl = world.update(t, color, depth, intr, robot.pan, robot.tilt, pose, 0.5, Assess(reba))
        for _ in range(100):  # let the map thread integrate this keyframe
            if world._q.empty():
                break
            time.sleep(0.01)
        rec.maybe_record(
            t,
            color,
            depth,
            intr,
            robot.pan,
            robot.tilt,
            wl["base"],
            pose,
            "standing",
            0.5,
            {"T_map_cam": [float(v) for v in wl["T"].reshape(-1)], "registration": "lidar"},
        )
        wall = wall0 + t
        rows.append(f"{t},{wall},1,standing,{reba},{32 if k < steps // 2 else 9}")
        # track.jsonl wall times follow the simulated clock
        world._track_f.flush()
    rec.close()
    world.close()
    logger.close()
    # rewrite track walls on the simulated clock + events / metrics
    tr = [json.loads(line) for line in (logger.dir / "track.jsonl").read_text().splitlines()]
    for r in tr:
        r["wall"] = wall0 + r["t"]
    (logger.dir / "track.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tr))
    (logger.dir / "metrics.csv").write_text("\n".join(rows) + "\n")
    tip = wall0 + (steps // 2) * 0.5 - 1
    events = [
        {"kind": "issue", "issue": "neck_flexion", "detail": "head 32 deg forward/down", "wall_time": tip},
        {"kind": "fall", "state": "alarm", "reason": "no recovery 12 s after", "wall_time": wall0 + 3},
        {"kind": "fall", "state": "ok", "reason": "acknowledged", "wall_time": wall0 + 6},
    ]
    (logger.dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return logger.dir, cfg, world


def test_registered_session_report_has_captions_heat_and_flat_floor(tmp_path):
    from stretch_posture_coach.report import write_report

    d, cfg, world = _run_sim_session(tmp_path)
    assert world.aligner.status()["tilt_offset_deg"] == pytest.approx(-3.0, abs=0.5)
    sc = build_scene(d, cfg)
    assert sc["source"] == "registered-lidar"
    ov = json.loads((d / "scene" / "overlay.json").read_text())
    kinds = {c["kind"]: c for c in ov["captions"]}
    assert "fall_alarm" in kinds and kinds["fall_alarm"]["level"] == "critical" and kinds["fall_alarm"]["pos"]
    imp = kinds.get("praise")
    assert imp is not None and imp["level"] == "good" and "32" in imp["text"] and "9" in imp["text"]
    assert len(ov["heat"]["cells"]) >= 3
    rebas = sorted(c["reba"] for c in ov["heat"]["cells"])
    assert rebas[0] < 5 < rebas[-1]  # better and worse areas
    # the registered floor is one layer (the tilt offset was corrected live)
    from stretch_posture_coach.scene import load_frames, registered_poses

    frames = load_frames(d)
    assert len(registered_poses(frames)) == len(frames)
    html = write_report(d).read_text(encoding="utf-8")
    assert "What happened where" in html and "Posture heat map" in html and "lidar SLAM" in html


def test_teleop_mapper_is_relative_and_rotates_with_the_base():
    from stretch_posture_coach.teleop import TeleopCommand, TeleopMapper

    m = TeleopMapper({})
    joints = {"lift": 0.5, "arm": 0.2, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}
    q0 = (0, 0, 0, 1)

    def cmd(p, q=q0, clutch=True, precision=False):
        return TeleopCommand(t=0, hand_p=np.array(p, float), hand_q=q, clutch=clutch, precision=precision)

    theta = math.pi / 2  # base faces map +y: its -y (arm direction) is map +x
    assert m.update(cmd([1, 1, 1]), joints, theta) == pytest.approx(
        {"lift": 0.5, "arm": 0.2, "wrist_yaw": 0, "wrist_pitch": 0, "wrist_roll": 0}, abs=1e-9
    )  # engaging the clutch never moves the robot
    out = m.update(cmd([1.1, 1, 1.15]), joints, theta)
    assert out["lift"] == pytest.approx(0.65) and out["arm"] == pytest.approx(0.3)
    s = math.sin(0.3 / 2)
    out = m.update(cmd([1.1, 1, 1.15], q=(0, s, 0, math.cos(0.3 / 2))), joints, theta)  # turn hand 0.3 rad
    assert out["wrist_yaw"] == pytest.approx(0.3, abs=1e-6)
    assert m.update(cmd([1.1, 1, 1.15], clutch=False), joints, theta) is None
    joints.update(lift=0.65, arm=0.3, wrist_yaw=0.3)  # the robot got there
    out = m.update(cmd([5, 5, 5], precision=True), joints, theta)  # re-clutch somewhere else: no jump
    assert out["lift"] == pytest.approx(0.65) and out["arm"] == pytest.approx(0.3)
    out = m.update(cmd([5, 5, 5.1], precision=True), joints, theta)
    assert out["lift"] == pytest.approx(0.65 + 0.03)  # precision scale 0.3


def test_lidar_guard_and_haptics():
    from stretch_posture_coach.teleop import Haptics, LidarGuard

    g = LidarGuard({})
    wall_ahead = lambda d: np.array([[d, y] for y in np.linspace(-0.3, 0.3, 13)])  # noqa: E731
    # emergency layer only (the CBF filter does the smooth slowing): untouched until 0.12 m from the footprint
    assert g(0.2, wall_ahead(0.80))[0] == pytest.approx(0.2)
    assert g(0.2, wall_ahead(0.30)) == (0.0, pytest.approx("obstacle 0.10 m ahead: stopped"))
    assert g(-0.2, wall_ahead(0.30))[0] == pytest.approx(-0.2)  # backing away is fine
    h = Haptics({})
    assert h({"gripper_effort": 3})["gripper"] == 0
    first = h({"gripper_effort": 40})
    assert first["contact"] and 0.5 < first["gripper"] < 0.7
    assert not h({"gripper_effort": 45})["contact"]  # click only on first contact


def test_teleop_controller_drives_squeezes_and_times_out():
    from stretch_posture_coach.teleop import TeleopCommand, TeleopController

    robot = SimRobot()
    tc = TeleopController(load_config()["teleop"], robot, world=None, server=None)
    now = time.monotonic()
    st = tc.step(TeleopCommand(t=now, drive=(1.0, 0.0), gripper="close"), now)
    assert st["active"] and st["v"] == pytest.approx(0.25)
    assert ("gripper", -100.0) in robot.log
    for _ in range(40):
        robot.step(0.05)
    st = tc.step(TeleopCommand(t=now + 2, drive=(0, 0)), now + 2)
    assert st["haptics"]["gripper"] > 0.5  # the object resists the closing gripper -> vibration
    st = tc.step(TeleopCommand(t=now + 2, drive=(0.5, 0)), now + 3)  # stale command
    assert not st["active"] and robot.log[-1] == ("drive", 0.0, 0.0)
