"""Expert teleop over the VR WebSocket (PIN, one owner, latched gripper, haptics), world streaming
(map deltas, lidar, occupancy, captions, heat), and robot_helper.py against a fake stretch_body."""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from importlib import resources

import numpy as np
import pytest
from test_web_vr import _ctx
from vr_sim import Assess, SimRobot, frame_at, person_at, sim_world

from stretch_posture_coach.camera import Intrinsics
from stretch_posture_coach.config import load_config
from stretch_posture_coach.scene import HeadKinematics
from stretch_posture_coach.vr import T_LIDAR, T_MAP, T_OCC, VRServer, unpack


@pytest.fixture
def vr(tmp_path):
    cfg = load_config()
    vc = cfg["vr"] | {"port": 0, "host": "127.0.0.1", "gripper_camera": False, "nav_camera": False}
    srv = VRServer(
        {**vc, "teleop": cfg["teleop"] | {"pin": "424242"}}, tmp_path, tmp_path / "certs", cfg["thresholds"]
    )
    yield srv
    srv.close()


async def _recv_text(ws, kind, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = await asyncio.wait_for(ws.recv(), end - time.monotonic())
        if isinstance(m, str) and json.loads(m).get("type") == kind:
            return json.loads(m)
    raise TimeoutError(kind)


def test_teleop_pin_owner_gripper_latch_and_haptics(vr):
    from websockets.asyncio.client import connect

    url = f"wss://127.0.0.1:{vr.port}/ws"

    async def scenario():
        async with connect(url, ssl=_ctx()) as a, connect(url, ssl=_ctx()) as b:
            await a.send(json.dumps({"type": "teleop_claim", "pin": "000000"}))
            assert (await _recv_text(a, "teleop"))["state"] == "denied"
            await a.send(json.dumps({"type": "teleop_claim", "pin": "424242"}))
            assert (await _recv_text(a, "teleop"))["state"] == "granted"
            await b.send(json.dumps({"type": "teleop_claim", "pin": "424242"}))
            assert (await _recv_text(b, "teleop"))["state"] == "busy"  # one expert at a time
            # commands from a non-owner are ignored
            await b.send(json.dumps({"type": "teleop", "drive": [1, 0]}))
            await asyncio.sleep(0.2)
            assert vr.teleop_command() is None
            await a.send(
                json.dumps(
                    {
                        "type": "teleop",
                        "drive": [0.5, -0.2],
                        "clutch": True,
                        "gripper": "close",
                        "hand": {"p": [1, 2, 1.1], "q": [0, 0, 0, 1]},
                    }
                )
            )
            await a.send(
                json.dumps({"type": "teleop", "drive": [0.5, -0.2], "clutch": True})
            )  # overwrites...
            await asyncio.sleep(0.2)
            cmd = vr.teleop_command()
            assert (
                cmd.drive == (0.5, -0.2) and cmd.clutch and cmd.gripper == "close"
            )  # ...but the press is latched
            assert vr.teleop_command().gripper is None  # delivered once
            vr.set_teleop_feedback({"active": True, "haptics": {"gripper": 0.7, "contact": True, "arm": 0}})
            fb = await _recv_text(a, "teleop_fb")
            assert fb["haptics"]["gripper"] == 0.7
        await asyncio.sleep(0.3)
        assert vr.teleop_command() is None and not vr.teleop_owner_connected  # owner left -> control dropped

    asyncio.run(scenario())


def test_pin_lockout_after_five_wrong_tries(vr):
    from websockets.asyncio.client import connect

    async def scenario():
        async with connect(f"wss://127.0.0.1:{vr.port}/ws", ssl=_ctx()) as a:
            for _ in range(5):
                await a.send(json.dumps({"type": "teleop_claim", "pin": "1"}))
                await _recv_text(a, "teleop")
            await a.send(json.dumps({"type": "teleop_claim", "pin": "424242"}))
            assert (await _recv_text(a, "teleop"))["state"] == "locked"

    asyncio.run(scenario())


def test_vr_streams_registered_world(tmp_path):
    """Map deltas, lidar scan, occupancy grid, captions and heat reach the headset; the RGB-D header has T."""
    from websockets.asyncio.client import connect

    cfg = load_config(overrides={"vr": {"map_hz": 50}})
    robot = SimRobot()
    world, lid = sim_world(cfg, robot)
    vc = cfg["vr"] | {"port": 0, "host": "127.0.0.1", "gripper_camera": False, "nav_camera": False}
    srv = VRServer({**vc, "teleop": cfg["teleop"]}, tmp_path, tmp_path / "certs", cfg["thresholds"])
    srv.attach_world(world)
    kin, intr = HeadKinematics(1.3), Intrinsics.from_fov(120, 160, 42)
    got: dict = {}

    async def scenario():
        async with connect(f"wss://127.0.0.1:{srv.port}/ws", ssl=_ctx(), max_size=None) as ws:
            end = time.monotonic() + 15
            k = 0
            while time.monotonic() < end and not {T_MAP, T_LIDAR, T_OCC, "captions", "heat", 1} <= set(got):
                robot._pan = -0.4 + 0.1 * (k % 9)
                person_at(1.9, 0.1)
                color, depth, pose = frame_at(robot, kin, intr, k)
                wl = world.update(k * 0.25, color, depth, intr, robot.pan, robot.tilt, pose, 0.5, Assess(7))
                if k == 3:
                    world.event(
                        {"kind": "fall", "state": "check", "reason": "test", "wall_time": time.time()}
                    )
                srv.publish(
                    color,
                    depth,
                    intr,
                    0.0,
                    robot.tilt,
                    pose,
                    {},
                    {},
                    0.5,
                    {"T": [float(v) for v in wl["T"].reshape(-1)]},
                )
                k += 1
                try:
                    m = await asyncio.wait_for(ws.recv(), 0.25)
                except TimeoutError:
                    continue
                if isinstance(m, str):
                    j = json.loads(m)
                    got[j["type"]] = j
                else:
                    t, meta, rest = unpack(m)
                    got.setdefault(t, (meta, rest))
                    await ws.send(json.dumps({"type": "ack", "t": t}))

    try:
        asyncio.run(scenario())
    finally:
        srv.close()
        lid.close()
        world.close()
    meta, rest = got[T_MAP]
    assert meta["full"] and meta["n"] > 100
    idx = np.frombuffer(rest[: meta["n"] * 6], "<i2").reshape(-1, 3)
    assert idx[:, 2].min() >= -2  # nothing below the floor
    assert got[T_LIDAR][0]["n"] > 100 and got[T_OCC][1][:4] == b"\x89PNG"
    assert got["captions"]["items"][0]["kind"] == "fall_check"
    assert got["heat"]["legend"]["colors"][1] == "#383835"
    assert len(got[1][0]["T"]) == 16


FAKE_STRETCH_BODY = {
    "stretch_body/__init__.py": "",
    "stretch_body/head.py": """
        class Head:
            def __init__(self): self.status = {"head_pan": {"pos": 0.0}, "head_tilt": {"pos": 0.0}}
            def startup(self, threaded=False): return True
            def pull_status(self): pass
            def move_to(self, j, x): self.status[j]["pos"] = x
            def stop(self): pass
    """,
    "stretch_body/robot.py": """
        import json, os, time
        from stretch_body.head import Head
        LOG = os.environ["FAKE_SB_LOG"]
        def log(*a):
            with open(LOG, "a") as f: f.write(json.dumps([time.time(), *a]) + "\\n")
        class Joint:
            def __init__(self, name): self.name, self.status = name, {"pos": 0.3, "motor": {"effort_pct": 12.0}}
            def move_to(self, x, v_m=None, a_m=None): log(self.name, x, v_m)
        class Base:
            status = {"x": 0.0, "y": 0.0, "theta": 0.0}
            def set_velocity(self, v, w): log("base", v, w)
            def translate_by(self, m, v_m=None, a_m=None): log("translate", m)
            def rotate_by(self, r, v_r=None, a_r=None): log("rotate", r)
        class EOA:
            joints = ["wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"]
            status = {"stretch_gripper": {"pos": 0.1, "pos_pct": 20.0, "effort": -35.0}, "wrist_yaw": {"pos": 0.0},
                      "wrist_pitch": {"pos": 0.0}, "wrist_roll": {"pos": 0.0}}
            def move_to(self, j, x, v=None, a=None): log(j, x)
        class Pimu: status = {"runstop_event": False}
        class Robot:
            def __init__(self):
                self.head, self.base, self.lift, self.arm = Head(), Base(), Joint("lift"), Joint("arm")
                self.end_of_arm, self.pimu = EOA(), Pimu()
            def startup(self): return True
            def is_homed(self): return True
            def push_command(self): pass
            def stop(self): log("stop")
    """,
}


def test_robot_helper_teleop_protocol_and_watchdog(tmp_path):
    for name, src in FAKE_STRETCH_BODY.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    log = tmp_path / "calls.jsonl"
    helper = resources.files("stretch_posture_coach").joinpath("robot_helper.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "FAKE_SB_LOG": str(log)}
    proc = subprocess.Popen(
        [sys.executable, str(helper), "--base"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["ready"] and ready["arm"] and ready["homed"] and "stretch_gripper" in ready["joints"]
        send = lambda o: (proc.stdin.write(json.dumps(o) + "\n"), proc.stdin.flush())  # noqa: E731
        send({"cmd": "drive", "v": 5.0, "w": -0.2})  # way above the cap
        send({"cmd": "joints", "lift": 2.0, "arm": 0.2, "wrist_yaw": 0.5})
        send({"cmd": "gripper", "pos": -100})
        st = {}
        for _ in range(20):
            st = json.loads(proc.stdout.readline())
            if "gripper_effort" in st:
                break
        assert st["gripper_effort"] == -35.0 and st["lift_effort"] == 12.0 and st["gripper"] == 20.0
        time.sleep(0.6)  # no new drive command -> watchdog stops the base
        send({"cmd": "quit"})
        proc.wait(timeout=5)
    finally:
        proc.kill()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    base = [c for c in calls if c[1] == "base"]
    assert base[0][2] == pytest.approx(0.30) and base[0][3] == pytest.approx(-0.2)  # clamped to 0.3 m/s
    assert any(c[2] == 0.0 and c[3] == 0.0 for c in base[1:])  # watchdog
    assert base[-1][2:] == [0.0, 0.0]
    stop_t = next(c[0] for c in base[1:] if c[2] == 0.0)
    assert 0.2 < stop_t - base[0][0] < 0.6
    lift = next(c for c in calls if c[1] == "lift")
    assert lift[2] == pytest.approx(1.08)  # clamped
    assert ["stretch_gripper", -100.0] == next(c for c in calls if c[1] == "stretch_gripper")[1:]
    assert any(c[1] == "wrist_yaw" and c[2] == pytest.approx(0.5) for c in calls)
