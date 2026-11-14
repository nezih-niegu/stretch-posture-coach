#!/usr/bin/env python3
"""Stretch head (+ optional base) controller, run with the robot's *system* python (has Hello Robot's stretch_body).

It is launched as a subprocess by ``robot.py`` so the uv environment never shares packages with the robot's Python.

Modes:
  robot_helper.py          head only (stretch_body.head.Head)
  robot_helper.py --base   whole robot (stretch_body.robot.Robot): head + mobile base for follow mode

Protocol (JSON lines).
  stdin : {"cmd": "move_by", "pan": rad, "tilt": rad}
          {"cmd": "move_to", "pan": rad|null, "tilt": rad|null}
          {"cmd": "base", "rotate": rad} | {"cmd": "base", "translate": m}   (--base only; bounded by the caller,
                                         and clamped again here)
          {"cmd": "quit"}
  stdout: {"ready": true, "base": bool} once, then {"pan", "tilt", "runstop"} at ~10 Hz, or {"error": "..."}

Base motions are small RELATIVE position moves (never open-ended velocity commands), so if this process or the
caller dies, the robot moves at most one step. They are clamped here as a second line of defense.
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
BASE_V_M = 0.08  # m/s
BASE_A_M = 0.10
BASE_V_R = 0.30  # rad/s
BASE_A_R = 0.40
PAN_LIM = (-4.0, 1.9)
TILT_LIM = (-1.5, 0.5)


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


class HeadOnly:
    has_base = False

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

    def status(self):
        self.head.pull_status()
        return {"pan": self.head.status["head_pan"]["pos"], "tilt": self.head.status["head_tilt"]["pos"]}

    def runstop(self):
        return None

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

    def status(self):
        # the Robot's own threads keep head status fresh; don't pull from here (not thread-safe)
        out = {"pan": self.head.status["head_pan"]["pos"], "tilt": self.head.status["head_tilt"]["pos"]}
        try:  # base odometry, for `posture-coach follow-test`
            b = self.robot.base.status
            out["base_x"], out["base_theta"] = b.get("x"), b.get("theta")
        except Exception:
            pass
        return out

    def runstop(self):
        try:
            return bool(self.robot.pimu.status.get("runstop_event"))
        except Exception:
            return None

    def rotate(self, rad):
        rad = clamp(float(rad), -MAX_ROTATE_RAD, MAX_ROTATE_RAD)
        call_with_optional_kwargs(self.robot.base.rotate_by, rad, v_r=BASE_V_R, a_r=BASE_A_R)
        self.robot.push_command()

    def translate(self, m):
        m = clamp(float(m), -MAX_TRANSLATE_M, MAX_TRANSLATE_M)
        call_with_optional_kwargs(self.robot.base.translate_by, m, v_m=BASE_V_M, a_m=BASE_A_M)
        self.robot.push_command()

    def stop(self):
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
    emit({"ready": True, "base": dev.has_base})

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
                line = inbox.get(timeout=0.05)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                cmd = msg.get("cmd")
                if cmd == "quit":
                    break
                st = dev.status()
                if cmd == "move_by":
                    if msg.get("pan"):
                        head.move_to("head_pan", clamp(st["pan"] + float(msg["pan"]), *PAN_LIM))
                    if msg.get("tilt"):
                        head.move_to("head_tilt", clamp(st["tilt"] + float(msg["tilt"]), *TILT_LIM))
                elif cmd == "move_to":
                    if msg.get("pan") is not None:
                        head.move_to("head_pan", clamp(float(msg["pan"]), *PAN_LIM))
                    if msg.get("tilt") is not None:
                        head.move_to("head_tilt", clamp(float(msg["tilt"]), *TILT_LIM))
                elif cmd == "base" and dev.has_base:
                    if dev.runstop():
                        emit({"warning": "runstop active, base command ignored"})
                    elif msg.get("rotate"):
                        dev.rotate(msg["rotate"])
                    elif msg.get("translate"):
                        dev.translate(msg["translate"])
            now = time.time()
            if now - last_status > 0.1:
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
