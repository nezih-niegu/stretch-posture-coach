"""Bridge to the Stretch head (pan/tilt) through ``robot_helper.py`` running on the system python."""

from __future__ import annotations

import collections
import json
import logging
import math
import shutil
import subprocess
import threading
import time
from importlib import resources

log = logging.getLogger(__name__)


def stretch_body_available(python: str = "/usr/bin/python3") -> bool:
    exe = shutil.which(python) or python
    try:
        r = subprocess.run([exe, "-c", "import stretch_body.head"], capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class NullHead:
    """Used when no robot is available. Reports a fixed camera pitch."""

    available = False

    def __init__(self, pitch_deg: float = 0.0):
        self._tilt = math.radians(pitch_deg)

    @property
    def tilt(self) -> float | None:
        return self._tilt

    @property
    def pan(self) -> float | None:
        return None

    def track(self, target_xy, now: float) -> None:
        pass

    def close(self) -> None:
        pass


class StretchHead:
    available = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        helper = resources.files("stretch_posture_coach").joinpath("robot_helper.py")
        python = shutil.which(cfg.get("python", "/usr/bin/python3")) or cfg.get("python", "/usr/bin/python3")
        self._proc = subprocess.Popen(
            [python, str(helper)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._pan: float | None = None
        self._tilt: float | None = None
        self._ready = threading.Event()
        self.error: str | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=20)
        self._reader = threading.Thread(target=self._read, daemon=True, name="head-reader")
        self._reader.start()
        threading.Thread(target=self._drain_stderr, daemon=True, name="head-stderr").start()
        if not self._ready.wait(timeout=25):
            err = self.error or self._stderr_tail() or "timeout waiting for robot_helper"
            self.close()
            raise RuntimeError(f"Stretch head unavailable: {err}")
        if self.error:
            self.close()
            raise RuntimeError(f"Stretch head unavailable: {self.error}")
        self._last_cmd = 0.0
        if cfg.get("initial_pan") is not None or cfg.get("initial_tilt") is not None:
            self._send({"cmd": "move_to", "pan": cfg.get("initial_pan"), "tilt": cfg.get("initial_tilt")})

    def _drain_stderr(self):
        assert self._proc.stderr is not None
        for line in self._proc.stderr:  # keep the pipe from filling up and blocking the helper
            self._stderr.append(line.rstrip())

    def _stderr_tail(self) -> str:
        return " | ".join(list(self._stderr)[-5:])

    def _read(self):
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # stretch_body may print log lines
            if "error" in msg:
                self.error = msg["error"]
                self._ready.set()
            elif msg.get("ready"):
                self._ready.set()
            else:
                self._pan, self._tilt = msg.get("pan"), msg.get("tilt")

    def _send(self, obj: dict):
        if self._proc.poll() is not None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    @property
    def pan(self) -> float | None:
        return self._pan

    @property
    def tilt(self) -> float | None:
        return self._tilt

    def track(self, target_xy: tuple[float, float] | None, now: float) -> None:
        """Proportional pan/tilt step toward ``target_xy`` (normalized image coords of the upright image)."""
        c = self.cfg
        if target_xy is None or not c.get("head_tracking", True):
            return
        if now - self._last_cmd < 1.0 / float(c.get("track_hz", 4)):
            return
        ex, ey = target_xy[0] - 0.5, target_xy[1] - 0.5
        db = float(c.get("deadband", 0.08))
        gain, max_step = float(c.get("gain", 0.5)), float(c.get("max_step_rad", 0.12))
        dpan = dtilt = 0.0
        if abs(ex) > db:
            dpan = float(c.get("pan_sign", -1)) * gain * ex * math.radians(float(c.get("hfov_deg", 42)))
        if abs(ey) > db:
            dtilt = float(c.get("tilt_sign", -1)) * gain * ey * math.radians(float(c.get("vfov_deg", 69)))
        dpan = max(-max_step, min(max_step, dpan))
        dtilt = max(-max_step, min(max_step, dtilt))
        if dpan or dtilt:
            self._send({"cmd": "move_by", "pan": dpan, "tilt": dtilt})
            self._last_cmd = now

    def close(self) -> None:
        self._send({"cmd": "quit"})
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def make_head(robot_cfg: dict, camera_pitch_deg: float = 0.0):
    enabled = robot_cfg.get("enabled", "auto")
    if enabled is False or str(enabled).lower() == "false":
        return NullHead(camera_pitch_deg)
    python = robot_cfg.get("python", "/usr/bin/python3")
    if str(enabled).lower() == "auto" and not stretch_body_available(python):
        log.info("stretch_body not found for %s; running without robot head control", python)
        return NullHead(camera_pitch_deg)
    try:
        return StretchHead(robot_cfg)
    except Exception as e:
        if enabled is True or str(enabled).lower() == "true":
            raise
        log.warning("%s; continuing without head control", e)
        return NullHead(camera_pitch_deg)


def wait_for_tilt(head, timeout_s: float = 2.0) -> float | None:
    end = time.monotonic() + timeout_s
    while head.tilt is None and time.monotonic() < end:
        time.sleep(0.05)
    return head.tilt
