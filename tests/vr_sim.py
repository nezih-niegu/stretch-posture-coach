"""A simulated Stretch for the VR / world / teleop tests: synthetic RGB-D room (scene_synth), simulated lidar,
a robot whose base and arm respond to teleop commands, and a person walking around.

    uv run python tests/vr_sim.py      # serves the real VR page on https://<this machine>:8443 with the sim
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import scene_synth  # noqa: E402

from stretch_posture_coach.camera import Intrinsics  # noqa: E402
from stretch_posture_coach.lidar import SimLidar  # noqa: E402
from stretch_posture_coach.scene import HeadKinematics, map_T_base  # noqa: E402
from stretch_posture_coach.slam2d import compose  # noqa: E402


def room_segments() -> np.ndarray:
    def box(b):
        (x0, x1), (y0, y1) = b["x"], b["y"]
        return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]

    return np.array(box(scene_synth.ROOM) + box(scene_synth.TABLE), float)


class SimRobot:
    """Duck-types robot.StretchHead (whole-robot mode)."""

    available = True
    has_base = True
    has_arm = True
    homed = True
    runstop = False

    def __init__(self):
        self.true = (0.0, 0.0, 0.0)  # ground-truth base pose (sim world = map at start)
        self.odom = (0.0, 0.0, 0.0)  # drifting wheel odometry
        self._pan, self._tilt = 0.0, -0.35
        self.v = self.w = 0.0
        self.drive_t = 0.0
        self.joints = {
            "lift": 0.6,
            "arm": 0.1,
            "wrist_yaw": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
            "gripper_effort": 2.0,
            "lift_effort": 10.0,
            "arm_effort": 5.0,
        }
        self.targets: dict = {}
        self.gripper_target = 50.0
        self.object_at = -30.0  # an object in the gripper: closing past this squeezes it
        self.log: list = []

    # --- StretchHead API
    @property
    def pan(self):
        return self._pan

    @property
    def tilt(self):
        return self._tilt

    @property
    def base_x(self):
        return self.odom[0]

    @property
    def base_y(self):
        return self.odom[1]

    @property
    def base_theta(self):
        return self.odom[2]

    def track(self, target, now):
        pass

    def look(self, pan, tilt, now):
        self._pan, self._tilt = pan, max(-1.3, min(0.4, tilt))

    def base_rotate(self, rad, compensate_head=True):
        self.true = compose(self.true, (0, 0, rad))
        self.odom = compose(self.odom, (0, 0, rad * 1.05))

    def base_translate(self, m):
        self.true = compose(self.true, (m, 0, 0))
        self.odom = compose(self.odom, (m * 1.05, 0, 0))

    def drive(self, v, w):
        self.v, self.w, self.drive_t = v, w, time.monotonic()
        self.log.append(("drive", v, w))

    def set_joints(self, **t):
        self.targets.update(t)
        self.log.append(("joints", dict(t)))

    def gripper(self, pct):
        self.gripper_target = pct
        self.log.append(("gripper", pct))

    def hold(self):
        self.v = self.w = 0.0

    def close(self):
        pass

    # --- physics
    def step(self, dt: float):
        if time.monotonic() - self.drive_t > 0.3:  # robot_helper watchdog
            self.v = self.w = 0.0
        if self.v or self.w:
            self.true = compose(self.true, (self.v * dt, 0, self.w * dt))
            self.odom = compose(self.odom, (self.v * dt * 1.05, 0.002 * dt, self.w * dt * 1.08))
        for k, tgt in self.targets.items():
            cur = self.joints[k]
            vmax = 0.15 if k in ("lift", "arm") else 1.0
            self.joints[k] = cur + max(-vmax * dt, min(vmax * dt, tgt - cur))
        g = self.joints["gripper"]
        tgt = max(self.gripper_target, self.object_at)  # the object stops the fingers
        self.joints["gripper"] = g + max(-80 * dt, min(80 * dt, tgt - g))
        squeeze = max(0.0, self.object_at - self.gripper_target)
        self.joints["gripper_effort"] = (
            2.0 + min(70.0, squeeze * 0.9) if self.joints["gripper"] <= self.object_at + 1 else 2.0
        )


class Assess:
    """Minimal stand-in for assessment.Assessment."""

    def __init__(self, reba):
        self.present = True
        self.reba = {"score": reba}
        self.rula = {"score": max(1, reba - 2)}
        self.metrics = {"posture": "standing", "neck_flexion": 10 + 3 * reba}


def person_at(x: float, y: float):
    scene_synth.PERSON["x"] = (x - 0.15, x + 0.15)
    scene_synth.PERSON["y"] = (y - 0.2, y + 0.2)


def sim_world(cfg: dict, robot: SimRobot):
    from stretch_posture_coach.world import LiveWorld

    lid = SimLidar(room_segments(), lambda: robot.true, hz=8, cfg=cfg.get("lidar", {}))
    world = LiveWorld(cfg, robot, None, lidar=lid)
    lid.start()
    return world, lid


def frame_at(robot: SimRobot, kin: HeadKinematics, intr: Intrinsics, t: float):
    x, y, th = robot.true
    T = map_T_base(x, y, th) @ kin.base_T_cam(robot.pan, robot.tilt)
    depth, color, _ = scene_synth.render(T, intr, with_person=True)
    pose = scene_synth.person_pose(T, intr)
    return color, depth, pose


def run(port: int = 8443, seconds: float | None = None, sessions="/tmp/vr-sim-sessions", fps: float = 8.0):
    import logging

    from stretch_posture_coach.config import load_config
    from stretch_posture_coach.teleop import TeleopController
    from stretch_posture_coach.vr import VRServer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(
        overrides={"vr": {"port": port, "gripper_camera": False, "nav_camera": False, "map_hz": 3}}
    )
    robot = SimRobot()
    world, lid = sim_world(cfg, robot)
    Path(sessions).mkdir(parents=True, exist_ok=True)
    world.new_session(Path(sessions))
    vr = VRServer({**cfg["vr"], "teleop": cfg["teleop"]}, sessions, "/tmp/vr-sim-certs", cfg["thresholds"])
    vr.attach_world(world)
    tele = TeleopController(cfg["teleop"], robot, world, vr)
    tele.start()
    print(f"VR sim on https://127.0.0.1:{vr.port}/  teleop PIN {vr.pin}", flush=True)
    kin = HeadKinematics(1.3)
    intr = Intrinsics.from_fov(240, 320, 42)
    t0 = time.monotonic()
    k = 0
    events = [
        (3, {"kind": "issue", "issue": "neck_flexion", "detail": "head 32 deg forward/down"}),
        (8, {"kind": "praise", "issue": "neck_flexion"}),
        (12, {"kind": "fall", "state": "check", "reason": "hips dropped 0.6 m in 0.8 s"}),
        (16, {"kind": "fall", "state": "alarm", "reason": "no recovery 12 s after the fall"}),
    ]
    try:
        while seconds is None or time.monotonic() - t0 < seconds:
            t = time.monotonic() - t0
            robot.step(1.0 / fps)
            robot._pan = 0.9 * math.sin(t * 0.35)  # head sweeps the room
            px, py = 1.8 + 0.6 * math.cos(t * 0.2), 0.3 * math.sin(t * 0.2)
            person_at(px, py)
            color, depth, pose = frame_at(robot, kin, intr, t)
            reba = 2 + 7 * (0.5 + 0.5 * math.sin(t * 0.3 + px))
            a = Assess(round(reba))
            wl = world.update(t, color, depth, intr, robot.pan, robot.tilt, pose, 0.5, a)
            for when, ev in list(events):
                if t >= when:
                    world.event({**ev, "wall_time": time.time()}, a.metrics)
                    events.remove((when, ev))
            vr.publish(
                color,
                depth,
                intr,
                robot.true[2] + robot.pan,
                robot.tilt,
                pose,
                {"neck_flexion": 25},
                {
                    "present": True,
                    "posture": "standing",
                    "reba": {"score": a.reba["score"], "risk": "medium"},
                },
                0.5,
                {
                    "T": [round(float(v), 5) for v in wl["T"].reshape(-1)],
                    "registered": "lidar",
                    "person": None if wl["person"] is None else [float(v) for v in wl["person"]],
                    "robot": {
                        "base": list(wl["base"]),
                        "pan": robot.pan,
                        "tilt": robot.tilt,
                        "joints": dict(robot.joints),
                        "has_base": True,
                        "has_arm": True,
                        "homed": True,
                    },
                    "world": world.status(),
                },
            )
            k += 1
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t0 - t)))
    except KeyboardInterrupt:
        pass
    finally:
        tele.close()
        lid.close()
        world.close()
        vr.close()
    return robot, world


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8443)
