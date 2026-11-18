"""Expert teleoperation from the Meta Quest: arm by hand tracking, base by joystick, gripper by A/B with haptic
feedback. Overrides follow mode while active.

Controls (VR page; the expert unlocks it with the PIN printed in the robot's log):
  right controller  hold GRIP = clutch: while held, the gripper follows your hand
                      up/down            -> lift
                      toward/away along the arm's direction -> arm extension
                      twist / tilt / roll the controller    -> wrist yaw / pitch / roll (Dex Wrist 3)
                    A = open gripper, B = close gripper (vibrates with the grip force)
                    TRIGGER (hold) = precision: hand motion scaled down
                    thumbstick click = wrist following on/off
  left controller   thumbstick = drive: forward/back, left/right turns the base (Stretch is differential
                    drive - it can't strafe; move sideways by turning, driving, turning back)
  right thumbstick  also drives (fine: 40 % speed)
  bare hands        right hand moves the arm while the LEFT hand pinches (clutch); right thumb-index distance
                    sets the gripper opening. No joystick and no vibration without controllers.

Safety: only in mode TELEOP (modes.py), one expert at a time (PIN), commands stop after 0.4 s without a fresh
message, base velocity capped and re-checked by robot_helper (0.3 s watchdog), every base command passes the
Control-Barrier-Function filter built from the lidar scan (cbf.py) plus a hard emergency stop, runstop
respected, arm/lift only when the robot is homed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw about local Y, pitch about local X, roll about local Z) - three.js 'YXZ' order."""
    m23 = float(np.clip(R[1, 2], -1, 1))
    x = math.asin(-m23)
    if abs(m23) < 0.9999999:
        y, z = math.atan2(R[0, 2], R[2, 2]), math.atan2(R[1, 0], R[1, 1])
    else:
        y, z = math.atan2(-R[2, 0], R[0, 0]), 0.0
    return y, x, z


def shape_axis(a: float, dead: float = 0.15, expo: float = 2.0) -> float:
    """Joystick deadzone + exponential curve (fine control near the center)."""
    a = float(np.clip(a, -1, 1))
    if abs(a) < dead:
        return 0.0
    m = (abs(a) - dead) / (1 - dead)
    return math.copysign(m**expo, a)


@dataclass
class TeleopCommand:
    t: float  # monotonic receive time
    drive: tuple[float, float] = (0.0, 0.0)  # stick (forward +, turn-left +), -1..1
    hand_p: np.ndarray | None = None  # map frame (m)
    hand_q: tuple | None = None  # map frame quaternion (x, y, z, w)
    clutch: bool = False
    precision: bool = False
    wrist: bool = True
    gripper: str | None = None  # "open" | "close" (button edge) | None
    aperture: float | None = None  # bare hands: 0 closed .. 1 open
    mode: str = "controllers"
    frame: str = "map"  # "base": hand pose already expressed in the robot base frame (VR ride-along views)

    @classmethod
    def from_msg(cls, m: dict, now: float) -> TeleopCommand:
        d = m.get("drive") or [0, 0]
        hand = m.get("hand") or {}
        p = hand.get("p")
        q = hand.get("q")
        ap = m.get("aperture")
        return cls(
            t=now,
            drive=(float(d[0]), float(d[1])),
            hand_p=np.array(p, float) if p is not None and len(p) == 3 else None,
            hand_q=tuple(float(v) for v in q) if q is not None and len(q) == 4 else None,
            clutch=bool(m.get("clutch")),
            precision=bool(m.get("precision")),
            wrist=bool(m.get("wrist", True)),
            gripper=m.get("gripper") if m.get("gripper") in ("open", "close") else None,
            aperture=None if ap is None else float(np.clip(ap, 0, 1)),
            mode=str(m.get("mode", "controllers")),
            frame="base" if m.get("frame") == "base" else "map",
        )


JOINT_LIMITS = {
    "lift": (0.05, 1.05),
    "arm": (0.0, 0.50),
    "wrist_yaw": (-1.7, 3.9),
    "wrist_pitch": (-1.5, 0.5),
    "wrist_roll": (-2.9, 2.9),
}
JOINT_DEFAULTS = {"lift": 0.6, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}


class TeleopMapper:
    """Clutched, relative hand -> joint mapping (no jumps when the clutch engages)."""

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.scale = float(c.get("hand_scale", 1.0))
        self.precision_scale = float(c.get("precision_scale", 0.3))
        self.arm_sign = float(c.get("arm_axis_sign", -1.0))  # Stretch's arm extends along -y of base_link
        g = c.get("wrist_gains", [1.0, 1.0, 1.0])
        self.wrist_gain = [float(v) for v in g]
        self.anchor = None
        self.targets: dict = {}

    def reset(self):
        self.anchor = None

    def update(self, cmd: TeleopCommand, joints: dict, base_theta: float) -> dict | None:
        if not cmd.clutch or cmd.hand_p is None or cmd.hand_q is None:
            self.anchor = None
            return None
        R = quat_to_R(cmd.hand_q)
        key = (cmd.precision, cmd.wrist)
        if self.anchor is None or self.anchor["key"] != key:
            # (re)anchor: hand pose now <-> the robot's current joints (or our last targets, if newer)
            j0 = {}
            for k in JOINT_LIMITS:
                if self.anchor is not None and k in self.targets:
                    j0[k] = self.targets[k]
                elif joints.get(k) is not None:
                    j0[k] = float(joints[k])
                else:
                    j0[k] = self.targets.get(k, JOINT_DEFAULTS[k])
            self.anchor = {"p": cmd.hand_p.copy(), "R": R, "j": j0, "theta": base_theta, "key": key}
        a = self.anchor
        s = self.precision_scale if cmd.precision else self.scale
        d_map = cmd.hand_p - a["p"]
        c, sn = math.cos(a["theta"]), math.sin(a["theta"])
        d_base = np.array([c * d_map[0] + sn * d_map[1], -sn * d_map[0] + c * d_map[1], d_map[2]])
        out = {
            "lift": a["j"]["lift"] + s * d_base[2],
            "arm": a["j"]["arm"] + s * self.arm_sign * d_base[1],
        }
        if cmd.wrist:
            yaw, pitch, roll = euler_yxz(a["R"].T @ R)
            ws = 1.0 if not cmd.precision else 0.5
            out["wrist_yaw"] = a["j"]["wrist_yaw"] + ws * self.wrist_gain[0] * yaw
            out["wrist_pitch"] = a["j"]["wrist_pitch"] + ws * self.wrist_gain[1] * pitch
            out["wrist_roll"] = a["j"]["wrist_roll"] - ws * self.wrist_gain[2] * roll
        for k, v in out.items():
            out[k] = float(np.clip(v, *JOINT_LIMITS[k]))
        self.targets.update(out)
        return out


class LidarGuard:
    """Scales forward/backward speed by the free distance in the lidar scan (base frame points)."""

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.enabled = bool(c.get("lidar_guard", True))
        self.half_len = float(c.get("footprint_half_length_m", 0.20))
        self.half_w = float(c.get("footprint_half_width_m", 0.20)) + 0.05
        self.stop = float(c.get("stop_distance_m", 0.12))  # emergency layer under the CBF filter
        self.slow = float(c.get("slow_distance_m", 0.05))

    def __call__(self, v: float, pts: np.ndarray | None) -> tuple[float, str]:
        if not self.enabled or abs(v) < 1e-6 or pts is None or not len(pts):
            return v, ""
        ahead = pts[:, 0] * math.copysign(1, v)
        lane = (np.abs(pts[:, 1]) < self.half_w) & (ahead > 0)
        if not lane.any():
            return v, ""
        gap = float(ahead[lane].min()) - self.half_len
        f = np.clip((gap - self.stop) / max(self.slow, 1e-3), 0.0, 1.0)
        if f <= 0:
            return 0.0, f"obstacle {gap:.2f} m {'ahead' if v > 0 else 'behind'}: stopped"
        if f < 1:
            return v * f, f"obstacle {gap:.2f} m: slowing"
        return v, ""


class Haptics:
    """Gripper (and arm/lift) motor effort -> vibration level, with a click on first contact."""

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.free = float(c.get("gripper_effort_free", 8.0))
        self.full = float(c.get("gripper_effort_full", 60.0))
        self.arm_free = float(c.get("arm_effort_free", 25.0))
        self.arm_full = float(c.get("arm_effort_full", 70.0))
        self._prev = 0.0

    def __call__(self, joints: dict) -> dict:
        def lvl(v, lo, hi):
            if v is None:
                return 0.0
            return float(np.clip((abs(float(v)) - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

        g = lvl(joints.get("gripper_effort"), self.free, self.full)
        a = max(
            lvl(joints.get("lift_effort"), self.arm_free, self.arm_full),
            lvl(joints.get("arm_effort"), self.arm_free, self.arm_full),
        )
        contact = g >= 0.15 and self._prev < 0.15
        self._prev = g
        return {"gripper": round(g, 3), "arm": round(a, 3), "contact": contact}


class TeleopController(threading.Thread):
    """20 Hz loop: VR command -> base velocity / joint targets / gripper; robot effort -> haptics."""

    def __init__(self, cfg: dict, robot, world=None, server=None, allowed=None):
        super().__init__(daemon=True, name="teleop")
        self.allowed = allowed or (lambda: True)  # e.g. lambda: fsm.mode == Mode.TELEOP
        self.c = cfg
        self.robot, self.world, self.server = robot, world, server
        self.mapper = TeleopMapper(cfg)
        self.guard = LidarGuard(cfg)
        self.haptics = Haptics(cfg)
        self.max_v = float(cfg.get("max_speed_m_s", 0.25))
        self.max_w = float(cfg.get("max_turn_rad_s", 0.6))
        self.timeout = float(cfg.get("command_timeout_s", 0.4))
        self.open_pct = float(cfg.get("gripper_open_pct", 50))
        self.close_pct = float(cfg.get("gripper_close_pct", -100))
        self.active = False
        self.note = ""
        self.running = True
        self._was_driving = False
        self._last_gripper_aperture = None
        self.state = {"active": False, "note": "", "v": 0.0, "w": 0.0}

    def step(self, cmd: TeleopCommand | None, now: float) -> dict:
        """One control cycle (public for tests)."""
        r = self.robot
        fresh = cmd is not None and now - cmd.t <= self.timeout and self.allowed()
        if not fresh:
            if self.active:
                log.info("teleop: no fresh command - holding")
                r.drive(0.0, 0.0)
                self.mapper.reset()
            self.active = False
            self.state = {"active": False, "note": "", "v": 0.0, "w": 0.0}
            return self.state
        if not self.active:
            log.info("teleop: expert control ON (follow mode paused)")
        self.active = True
        notes = []
        if getattr(r, "runstop", None):
            notes.append("runstop pressed")
        # base
        v = self.max_v * shape_axis(cmd.drive[0])
        w = self.max_w * shape_axis(cmd.drive[1])
        pts = None
        if self.world is not None and getattr(self.world, "lidar_ok", False):
            pts = self.world.loc.latest_points
        cbf = None
        if pts is not None and getattr(self.world, "cbf", None) is not None and (v or w):
            cbf = self.world.cbf.filter(v, w, pts)  # minimal change that keeps every lidar barrier h >= 0
            if cbf.modified:
                notes.append(f"barrier: {cbf.v_ref:.2f}->{cbf.v:.2f} m/s, {cbf.w_ref:.2f}->{cbf.w:.2f} rad/s")
            v, w = cbf.v, cbf.w
        v, gnote = self.guard(v, pts)
        if gnote:
            notes.append(gnote)
        if v or w or self._was_driving:
            r.drive(v, w)
        self._was_driving = bool(v or w)
        # arm
        joints = dict(getattr(r, "joints", {}) or {})
        homed = getattr(r, "homed", None)
        theta = 0.0
        if self.world is not None and cmd.frame != "base":
            theta = float(self.world.base_pose()[2])
        targets = None
        if cmd.clutch and getattr(r, "has_arm", False) and homed is False:
            notes.append("arm not homed: run stretch_robot_home.py")
        else:
            targets = self.mapper.update(cmd, joints, theta)
            if targets:
                r.set_joints(**targets)
        # gripper
        if cmd.gripper == "open":
            r.gripper(self.open_pct)
        elif cmd.gripper == "close":
            r.gripper(self.close_pct)
        elif cmd.aperture is not None and cmd.clutch:
            pct = self.close_pct + cmd.aperture * (self.open_pct - self.close_pct)
            if self._last_gripper_aperture is None or abs(pct - self._last_gripper_aperture) > 4:
                r.gripper(pct)
                self._last_gripper_aperture = pct
        hap = self.haptics(joints)
        self.note = "; ".join(notes)
        self.state = {
            "active": True,
            "note": self.note,
            "v": round(v, 3),
            "w": round(w, 3),
            "clutch": cmd.clutch,
            "targets": targets,
            "haptics": hap,
            "mode": cmd.mode,
            "barrier": None if cbf is None else {"active": len(cbf.active), "h_min": cbf.h_min},
        }
        return self.state

    def run(self):
        period = 1.0 / float(self.c.get("rate_hz", 20))
        while self.running:
            t0 = time.monotonic()
            try:
                cmd = self.server.teleop_command() if self.server is not None else None
                st = self.step(cmd, t0)
                if self.server is not None:
                    self.server.set_teleop_feedback(st)
            except Exception as e:  # never die silently with the base moving
                log.error("teleop loop: %s", e)
                try:
                    self.robot.drive(0.0, 0.0)
                except Exception:
                    pass
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def close(self):
        self.running = False
        if self.active:
            try:
                self.robot.drive(0.0, 0.0)
            except Exception:
                pass
