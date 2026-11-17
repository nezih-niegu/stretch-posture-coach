#!/usr/bin/env python3
"""Stretch controller, run with the robot's *system* python (has Hello Robot's stretch_body).

It is launched as a subprocess by ``robot.py`` so the uv environment never shares packages with the robot's Python.

Modes:
  robot_helper.py          head only (stretch_body.head.Head)
  robot_helper.py --base   whole robot (stretch_body.robot.Robot): head, mobile base, lift, arm, wrist, gripper
                           (follow mode and expert VR teleoperation)

Protocol (JSON lines).
  stdin : {"cmd": "move_by", "pan": rad, "tilt": rad}
          {"cmd": "move_to", "pan": rad|null, "tilt": rad|null}
          {"cmd": "base", "rotate": rad} | {"cmd": "base", "translate": m}  follow mode: small RELATIVE steps
          {"cmd": "drive", "v": m/s, "w": rad/s}     teleop velocity; must be repeated (watchdog stops the base
                                                     DRIVE_TIMEOUT_S after the last one)
          {"cmd": "joints", "lift": m, "arm": m, "wrist_yaw": rad, "wrist_pitch": rad, "wrist_roll": rad}
                                                     absolute targets, any subset (teleop)
          {"cmd": "gripper", "pos": pct}             stretch_gripper position (about -100 closed .. 100 open)
          {"cmd": "hold"}                            stop the base, freeze arm/lift/wrist where they are
          {"cmd": "quit"}
  stdout: {"ready": true, "base": bool, "arm": bool, "homed": bool, "joints": [...]} once,
          then status at ~15 Hz: {"pan", "tilt", "runstop", "base_x", ..., "lift", "arm", "wrist_yaw", ...,
          "gripper", "gripper_effort", "lift_effort", "arm_effort"}   or {"error": "..."} / {"warning": "..."}

Every command is clamped here as a second line of defense (the caller clamps too).
This file must only use the standard library + stretch_body (it is NOT run inside the uv venv).
"""

import json
import sys
import threading
import time

try:
    import queue
except ImportError:  # pragma: no cover
    import Queue as queue  # noqa: N813

MAX_TRANSLATE_M = 0.25
MAX_ROTATE_RAD = 0.6
BASE_V_M = 0.08  # m/s, follow-mode steps
BASE_A_M = 0.10
BASE_V_R = 0.30  # rad/s
BASE_A_R = 0.40
PAN_LIM = (-4.0, 1.9)
TILT_LIM = (-1.5, 0.5)

DRIVE_MAX_V = 0.30  # m/s   teleop hard caps
DRIVE_MAX_W = 1.0  # rad/s
DRIVE_TIMEOUT_S = 0.3
LIFT_LIM = (0.02, 1.08)
ARM_LIM = (0.0, 0.52)
WRIST_LIM = {"wrist_yaw": (-1.75, 4.0), "wrist_pitch": (-1.57, 0.56), "wrist_roll": (-2.95, 2.95)}
GRIPPER_LIM = (-100.0, 100.0)
LIFT_V, LIFT_A = 0.15, 0.3  # teleop joint speeds (m/s, m/s^2)
ARM_V, ARM_A = 0.15, 0.3
WRIST_V, WRIST_A = 1.0, 2.0  # rad/s
JOINT_PUSH_S = 1.0 / 15


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def call_with_optional_kwargs(fn, x, **kwargs):
    """stretch_body signatures differ slightly across versions; fall back to the bare positional call."""
    try:
        return fn(x, **kwargs)
    except TypeError:
        return fn(x)


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


class HeadOnly:
    has_base = False
    has_arm = False

    def __init__(self):
        from stretch_body.head import Head

        self.head = Head()
        try:
            ok = self.head.startup(threaded=False)
        except TypeError:
            ok = self.head.startup()
        if ok is False:
            raise RuntimeError(
                "head.startup() failed - is another process (e.g. stretch_driver) using the robot? "
                "Try: stretch_free_robot_process.py"
            )

    def ready_info(self):
        return {"base": False, "arm": False, "homed": None, "joints": ["head_pan", "head_tilt"]}

    def status(self):
        self.head.pull_status()
        return {"pan": self.head.status["head_pan"]["pos"], "tilt": self.head.status["head_tilt"]["pos"]}

    def runstop(self):
        return None

    def tick(self, now):
        pass

    def stop(self):
        self.head.stop()


class WholeRobot:
    has_base = True

    def __init__(self):
        from stretch_body.robot import Robot

        self.robot = Robot()
        if self.robot.startup() is False:
            raise RuntimeError(
                "robot.startup() failed - is another process (e.g. stretch_driver) using the robot? "
                "Try: stretch_free_robot_process.py"
            )
        self.head = self.robot.head
        eoa = getattr(self.robot, "end_of_arm", None)
        self.eoa_joints = list(getattr(eoa, "joints", []) or []) if eoa is not None else []
        self.has_arm = (
            getattr(self.robot, "lift", None) is not None and getattr(self.robot, "arm", None) is not None
        )
        self._drive_t = 0.0
        self._driving = False
        self._joint_targets = {}
        self._joint_pushed = 0.0

    def homed(self):
        for name in ("is_homed", "is_calibrated"):
            fn = getattr(self.robot, name, None)
            if fn is not None:
                try:
                    return bool(fn())
                except Exception:
                    pass
        return None

    def ready_info(self):
        return {
            "base": True,
            "arm": self.has_arm,
            "homed": self.homed(),
            "joints": ["head_pan", "head_tilt", "base"]
            + (["lift", "arm"] if self.has_arm else [])
            + self.eoa_joints,
        }

    def status(self):
        # the Robot's own threads keep status fresh; don't pull from here (not thread-safe)
        out = {"pan": self.head.status["head_pan"]["pos"], "tilt": self.head.status["head_tilt"]["pos"]}
        try:
            b = self.robot.base.status
            out["base_x"], out["base_y"], out["base_theta"] = b.get("x"), b.get("y"), b.get("theta")
        except Exception:
            pass
        if self.has_arm:
            for name in ("lift", "arm"):
                st = getattr(self.robot, name).status
                out[name] = st.get("pos")
                eff = _get(st, "motor", "effort_pct")
                if eff is None:
                    eff = st.get("force")
                out[name + "_effort"] = eff
        eoa = getattr(self.robot, "end_of_arm", None)
        if eoa is not None:
            for j in self.eoa_joints:
                st = eoa.status.get(j, {}) if isinstance(eoa.status, dict) else {}
                key = "gripper" if j == "stretch_gripper" else j
                out[key] = st.get("pos_pct", st.get("pos")) if j == "stretch_gripper" else st.get("pos")
                if j == "stretch_gripper":
                    out["gripper_effort"] = st.get("effort")
                    out["gripper_stalled"] = st.get("stalled")
        return out

    def runstop(self):
        try:
            return bool(self.robot.pimu.status.get("runstop_event"))
        except Exception:
            return None

    # --- follow mode (relative steps)
    def rotate(self, rad):
        rad = clamp(float(rad), -MAX_ROTATE_RAD, MAX_ROTATE_RAD)
        call_with_optional_kwargs(self.robot.base.rotate_by, rad, v_r=BASE_V_R, a_r=BASE_A_R)
        self.robot.push_command()

    def translate(self, m):
        m = clamp(float(m), -MAX_TRANSLATE_M, MAX_TRANSLATE_M)
        call_with_optional_kwargs(self.robot.base.translate_by, m, v_m=BASE_V_M, a_m=BASE_A_M)
        self.robot.push_command()

    # --- teleop
    def drive(self, v, w, now):
        v = clamp(float(v), -DRIVE_MAX_V, DRIVE_MAX_V)
        w = clamp(float(w), -DRIVE_MAX_W, DRIVE_MAX_W)
        self.robot.base.set_velocity(v, w)
        self.robot.push_command()
        self._drive_t = now
        self._driving = abs(v) > 1e-4 or abs(w) > 1e-4

    def joints(self, msg):
        if not self.has_arm:
            return
        for k in ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"):
            if msg.get(k) is not None:
                self._joint_targets[k] = float(msg[k])

    def gripper(self, pct):
        if "stretch_gripper" not in self.eoa_joints:
            return
        self._eoa_move("stretch_gripper", clamp(float(pct), *GRIPPER_LIM))
        self.robot.push_command()

    def _eoa_move(self, joint, x, v=None, a=None):
        try:
            if v is not None:
                self.robot.end_of_arm.move_to(joint, x, v, a)
            else:
                self.robot.end_of_arm.move_to(joint, x)
        except TypeError:
            self.robot.end_of_arm.move_to(joint, x)

    def hold(self):
        self.robot.base.set_velocity(0.0, 0.0)
        self._driving = False
        self._joint_targets = {}
        if self.has_arm:
            for name in ("lift", "arm"):
                pos = getattr(self.robot, name).status.get("pos")
                if pos is not None:
                    getattr(self.robot, name).move_to(pos)
        self.robot.push_command()

    def tick(self, now):
        """Watchdog + rate-limited joint pushes (called every loop)."""
        if self._driving and now - self._drive_t > DRIVE_TIMEOUT_S:
            self.robot.base.set_velocity(0.0, 0.0)
            self.robot.push_command()
            self._driving = False
        if self._joint_targets and now - self._joint_pushed >= JOINT_PUSH_S:
            t = self._joint_targets
            self._joint_targets = {}
            if "lift" in t:
                call_with_optional_kwargs(
                    self.robot.lift.move_to, clamp(t["lift"], *LIFT_LIM), v_m=LIFT_V, a_m=LIFT_A
                )
            if "arm" in t:
                call_with_optional_kwargs(
                    self.robot.arm.move_to, clamp(t["arm"], *ARM_LIM), v_m=ARM_V, a_m=ARM_A
                )
            for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
                if j in t and j in self.eoa_joints:
                    self._eoa_move(j, clamp(t[j], *WRIST_LIM[j]), WRIST_V, WRIST_A)
            self.robot.push_command()
            self._joint_pushed = now

    def stop(self):
        try:
            self.robot.base.set_velocity(0.0, 0.0)
            self.robot.push_command()
        except Exception:
            pass
        self.robot.stop()


def main():
    use_base = "--base" in sys.argv[1:]
    try:
        dev = WholeRobot() if use_base else HeadOnly()
    except ImportError as e:
        emit({"error": "stretch_body not importable: %s" % e})
        return 1
    except Exception as e:
        emit({"error": str(e)})
        return 1
    emit(dict({"ready": True}, **dev.ready_info()))

    head = dev.head
    last_status = 0.0
    # Read stdin on a thread: select() on a buffered text stream misses lines already buffered,
    # which would delay commands that arrive in bursts.
    inbox = queue.Queue()

    def reader():
        for raw in sys.stdin:
            inbox.put(raw)
        inbox.put(None)  # EOF: parent went away

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            try:
                line = inbox.get(timeout=0.02)
            except queue.Empty:
                line = ""
            if line is None:
                break
            now = time.time()
            if line:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                cmd = msg.get("cmd")
                if cmd == "quit":
                    break
                if cmd in ("move_by", "move_to"):
                    st = dev.status()
                    if cmd == "move_by":
                        if msg.get("pan"):
                            head.move_to("head_pan", clamp(st["pan"] + float(msg["pan"]), *PAN_LIM))
                        if msg.get("tilt"):
                            head.move_to("head_tilt", clamp(st["tilt"] + float(msg["tilt"]), *TILT_LIM))
                    else:
                        if msg.get("pan") is not None:
                            head.move_to("head_pan", clamp(float(msg["pan"]), *PAN_LIM))
                        if msg.get("tilt") is not None:
                            head.move_to("head_tilt", clamp(float(msg["tilt"]), *TILT_LIM))
                elif dev.has_base and cmd in ("base", "drive", "joints", "gripper"):
                    if dev.runstop():
                        if cmd != "drive" or msg.get("v") or msg.get("w"):
                            emit({"warning": "runstop active, %s command ignored" % cmd})
                    elif cmd == "base":
                        if msg.get("rotate"):
                            dev.rotate(msg["rotate"])
                        elif msg.get("translate"):
                            dev.translate(msg["translate"])
                    elif cmd == "drive":
                        dev.drive(msg.get("v", 0.0), msg.get("w", 0.0), now)
                    elif cmd == "joints":
                        dev.joints(msg)
                    elif cmd == "gripper" and msg.get("pos") is not None:
                        dev.gripper(msg["pos"])
                elif dev.has_base and cmd == "hold":
                    dev.hold()
            dev.tick(now)
            if now - last_status > 0.066:
                out = dev.status()
                out["runstop"] = dev.runstop()
                emit(out)
                last_status = now
    finally:
        try:
            dev.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
