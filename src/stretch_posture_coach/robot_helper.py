#!/usr/bin/env python3
"""Stretch head controller, run with the robot's *system* python (which has Hello Robot's stretch_body).

It is launched as a subprocess by ``robot.py`` so the uv environment never has to share packages with the
robot's Python. Protocol: JSON lines.
  stdin : {"cmd": "move_by", "pan": rad, "tilt": rad} | {"cmd": "move_to", "pan": rad|null, "tilt": rad|null}
          | {"cmd": "quit"}
  stdout: {"ready": true} once, then {"pan": rad, "tilt": rad} at ~10 Hz, or {"error": "..."}
This file must only use the standard library + stretch_body (it is NOT run inside the uv venv).
"""

import json
import select
import sys
import time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        from stretch_body.head import Head
    except Exception as e:
        emit({"error": "stretch_body not importable: %s" % e})
        return 1

    head = Head()
    try:
        ok = head.startup(threaded=False)
    except TypeError:
        ok = head.startup()
    if ok is False:
        emit(
            {
                "error": "head.startup() failed - is another process (e.g. stretch_driver) using the robot? "
                "Try: stretch_free_robot_process.py"
            }
        )
        return 1
    emit({"ready": True})

    pan_lim = (-4.0, 1.7)
    tilt_lim = (-1.5, 0.5)
    last_status = 0.0
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                cmd = msg.get("cmd")
                if cmd == "quit":
                    break
                head.pull_status()
                pan = head.status["head_pan"]["pos"]
                tilt = head.status["head_tilt"]["pos"]
                if cmd == "move_by":
                    if msg.get("pan"):
                        target = min(max(pan + float(msg["pan"]), pan_lim[0]), pan_lim[1])
                        head.move_to("head_pan", target)
                    if msg.get("tilt"):
                        target = min(max(tilt + float(msg["tilt"]), tilt_lim[0]), tilt_lim[1])
                        head.move_to("head_tilt", target)
                elif cmd == "move_to":
                    if msg.get("pan") is not None:
                        head.move_to("head_pan", float(msg["pan"]))
                    if msg.get("tilt") is not None:
                        head.move_to("head_tilt", float(msg["tilt"]))
            now = time.time()
            if now - last_status > 0.1:
                head.pull_status()
                emit({"pan": head.status["head_pan"]["pos"], "tilt": head.status["head_tilt"]["pos"]})
                last_status = now
    finally:
        try:
            head.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
