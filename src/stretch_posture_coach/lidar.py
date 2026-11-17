"""Stretch 3 base lidar (Slamtec RPLidar A1) without ROS: a small pyserial driver.

Protocol (RPLidar interface protocol, legacy SCAN):
  request   0xA5 <cmd> [payload]          STOP 0x25, RESET 0x40, SCAN 0x20, GET_HEALTH 0x52
  response  0xA5 0x5A <u30 len | u2 mode> <u8 type>   (7-byte descriptor)
  scan node 5 bytes: b0 = quality<<2 | !S<<1 | S   (S = first node of a new 360 deg revolution)
                     b1 = angle_q6[6:0]<<1 | 1     b2 = angle_q6[14:7]      (angle = q6 / 64 deg, clockwise)
                     b3,b4 = distance_q2 (little endian, mm * 4; 0 = no return)
  The A1's motor is switched by the USB adapter's DTR line (DTR low = spin).

Angles are clockwise seen from above; ``to_base`` converts a revolution into base-frame points (x forward,
y left) using the mount yaw from the config (Stretch's laser frame is turned 180 deg).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SYNC, SYNC2 = 0xA5, 0x5A
CMD_STOP, CMD_RESET, CMD_SCAN, CMD_HEALTH = 0x25, 0x40, 0x20, 0x52
SCAN_DESCRIPTOR = bytes([0xA5, 0x5A, 0x05, 0x00, 0x00, 0x40, 0x81])


@dataclass
class Scan:
    t: float  # monotonic time at the end of the revolution
    angle_deg: np.ndarray  # clockwise, as reported by the sensor
    range_m: np.ndarray
    quality: np.ndarray


def parse_nodes(buf: bytearray) -> tuple[list[tuple[bool, int, float, float]], int]:
    """Decode as many 5-byte scan nodes as possible from ``buf``.

    Returns ([(start, quality, angle_deg, dist_m)], bytes_consumed). Misaligned bytes are skipped one at a time
    until the two check bits line up again (resync after noise or a dropped byte)."""
    out = []
    i, n = 0, len(buf)
    while n - i >= 5:
        b0, b1 = buf[i], buf[i + 1]
        s, ns = b0 & 1, (b0 >> 1) & 1
        if s == ns or not (b1 & 1):
            i += 1  # resync
            continue
        angle = ((b1 >> 1) | (buf[i + 2] << 7)) / 64.0
        dist = (buf[i + 3] | (buf[i + 4] << 8)) / 4000.0
        out.append((bool(s), b0 >> 2, angle, dist))
        i += 5
    return out, i


def encode_node(start: bool, quality: int, angle_deg: float, dist_m: float) -> bytes:
    """Inverse of ``parse_nodes`` (used by tests and the simulator)."""
    q6 = int(round((angle_deg % 360.0) * 64)) & 0x7FFF
    d4 = int(round(dist_m * 4000)) & 0xFFFF
    b0 = ((quality & 0x3F) << 2) | ((0 if start else 1) << 1) | (1 if start else 0)
    return bytes([b0, ((q6 & 0x7F) << 1) | 1, q6 >> 7, d4 & 0xFF, d4 >> 8])


def to_base(
    scan: Scan,
    yaw_offset_deg: float = 180.0,
    clockwise: bool = True,
    x: float = 0.0,
    y: float = 0.0,
    min_range: float = 0.25,
    max_range: float = 10.0,
) -> np.ndarray:
    """Valid returns of one revolution as (N, 2) base-frame points (x forward, y left)."""
    r = scan.range_m
    ok = (r > min_range) & (r < max_range)
    a = np.radians(scan.angle_deg[ok])
    th = math.radians(yaw_offset_deg) + (-a if clockwise else a)
    return np.stack([x + r[ok] * np.cos(th), y + r[ok] * np.sin(th)], axis=1)


class _ScanAssembler:
    """Collects nodes into complete revolutions."""

    def __init__(self, min_points: int = 60):
        self.min_points = min_points
        self._cur: list[tuple[int, float, float]] = []
        self.latest: Scan | None = None
        self.count = 0

    def feed(self, nodes, now: float) -> Scan | None:
        done = None
        for start, q, ang, dist in nodes:
            if start and self._cur:
                if len(self._cur) >= self.min_points:
                    arr = np.array(self._cur, np.float64)
                    done = Scan(now, arr[:, 1], arr[:, 2], arr[:, 0])
                    self.latest = done
                    self.count += 1
                self._cur = []
            self._cur.append((q, ang, dist))
        return done


class RPLidar(threading.Thread):
    """Background reader. ``latest`` is the most recent complete revolution; ``on_scan`` is called per scan."""

    def __init__(self, port: str = "/dev/hello-lrf", baud: int = 115200, on_scan=None):
        super().__init__(daemon=True, name="lidar")
        import serial  # pyserial

        self.port = port
        self._ser = serial.Serial(port, baud, timeout=0.1)
        self.on_scan = on_scan
        self._asm = _ScanAssembler()
        self.running = True
        self.error: str | None = None
        self.health: tuple[int, int] | None = None

    @property
    def latest(self) -> Scan | None:
        return self._asm.latest

    @property
    def scans(self) -> int:
        return self._asm.count

    def _cmd(self, cmd: int):
        self._ser.write(bytes([SYNC, cmd]))

    def _read_descriptor(self, timeout: float = 1.0) -> bytes:
        end = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < end:
            buf += self._ser.read(7 - len(buf)) if len(buf) < 7 else b""
            k = buf.find(bytes([SYNC, SYNC2]))
            if k > 0:
                buf = buf[k:]
            if len(buf) >= 7 and buf[:2] == bytes([SYNC, SYNC2]):
                return buf[:7]
        raise TimeoutError("no response descriptor from the lidar")

    def get_health(self) -> tuple[int, int]:
        """(status, error_code): status 0 = good, 1 = warning, 2 = error."""
        self._ser.reset_input_buffer()
        self._cmd(CMD_HEALTH)
        self._read_descriptor()
        data = self._ser.read(3)
        if len(data) < 3:
            raise TimeoutError("short health response")
        return data[0], data[1] | (data[2] << 8)

    def start_motor(self):
        self._ser.dtr = False

    def stop_motor(self):
        self._ser.dtr = True

    def run(self):
        try:
            self._cmd(CMD_STOP)
            time.sleep(0.05)
            self._ser.reset_input_buffer()
            try:
                self.health = self.get_health()
                if self.health[0] == 2:
                    log.warning("lidar reports an error state (code %d): resetting", self.health[1])
                    self._cmd(CMD_RESET)
                    time.sleep(1.0)
                    self._ser.reset_input_buffer()
            except TimeoutError as e:
                log.debug("lidar health: %s", e)
            self.start_motor()
            time.sleep(0.5)  # spin-up
            self._ser.reset_input_buffer()
            self._cmd(CMD_SCAN)
            self._read_descriptor(2.0)
            buf = bytearray()
            while self.running:
                chunk = self._ser.read(self._ser.in_waiting or 5)
                if not chunk:
                    continue
                buf += chunk
                nodes, used = parse_nodes(buf)
                del buf[:used]
                scan = self._asm.feed(nodes, time.monotonic())
                if scan is not None and self.on_scan is not None:
                    try:
                        self.on_scan(scan)
                    except Exception as e:  # never kill the reader
                        log.debug("on_scan: %s", e)
        except Exception as e:
            self.error = str(e)
            log.warning("lidar stopped: %s", e)
        finally:
            try:
                self._cmd(CMD_STOP)
                self.stop_motor()
                self._ser.close()
            except Exception:
                pass

    def close(self):
        self.running = False
        self.join(timeout=2)


class SimLidar(threading.Thread):
    """Simulated lidar for tests / demos: ray-casts a 2-D polygon world from a pose callback."""

    def __init__(self, segments: np.ndarray, pose_fn, hz: float = 8.0, n: int = 360, on_scan=None, cfg=None):
        super().__init__(daemon=True, name="sim-lidar")
        self.segments, self.pose_fn, self.hz, self.n = segments, pose_fn, hz, n
        self.on_scan = on_scan
        self.cfg = cfg or {}
        self.latest: Scan | None = None
        self.scans = 0
        self.running = True
        self.error = None

    def scan_once(self) -> Scan:
        x, y, th = self.pose_fn()
        ang_cw = np.arange(self.n) * 360.0 / self.n
        yaw0 = float(self.cfg.get("yaw_offset_deg", 180.0))
        world_th = th + np.radians(yaw0) - np.radians(ang_cw)
        r = raycast(self.segments, np.array([x, y]), world_th)
        return Scan(time.monotonic(), ang_cw, r, np.full(self.n, 40.0))

    def run(self):
        while self.running:
            s = self.scan_once()
            self.latest = s
            self.scans += 1
            if self.on_scan:
                self.on_scan(s)
            time.sleep(1.0 / self.hz)

    def close(self):
        self.running = False


def raycast(segments: np.ndarray, origin: np.ndarray, angles: np.ndarray, max_r: float = 12.0) -> np.ndarray:
    """Distance along each ray to the nearest of ``segments`` (M, 4: x1 y1 x2 y2); 0 = no hit."""
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N,2)
    p1, p2 = segments[:, :2], segments[:, 2:]
    e = p2 - p1  # (M,2)
    w = p1 - origin  # (M,2)
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (N,M)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
        u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / den
    hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1) & (t < max_r)
    t = np.where(hit, t, np.inf).min(axis=1)
    return np.where(np.isfinite(t), t, 0.0)


def open_lidar(cfg: dict, on_scan=None):
    """Start the base lidar if configured/present. Returns the reader thread or None."""
    enabled = cfg.get("enabled", "auto")
    if enabled is False or str(enabled).lower() in ("false", "off"):
        return None
    port = str(cfg.get("port", "/dev/hello-lrf"))
    from pathlib import Path

    if str(enabled).lower() == "auto" and not Path(port).exists():
        log.info("No lidar at %s: maps use wheel odometry only (set lidar.port if it lives elsewhere)", port)
        return None
    try:
        lid = RPLidar(port, int(cfg.get("baud", 115200)), on_scan=on_scan)
    except Exception as e:
        if str(enabled).lower() == "true":
            raise
        log.warning("Lidar %s unavailable (%s): maps use wheel odometry only", port, e)
        return None
    lid.start()
    log.info("Lidar %s started (2-D SLAM registers the VR scan and the scene map)", port)
    return lid
