"""WebXR telepresence server for the Meta Quest browser (https://<robot>:8443/).

Why this is immersive (and not a flat stereo video):
  The head D435i gives color + aligned depth. We stream both; the Quest rebuilds a textured 3-D surface of the
  scene and renders it natively per eye -> true stereo AND head-motion parallax at the headset's frame rate
  (72-120 Hz), independent of the network rate. Each surface is placed with the robot head pose it was captured
  with, so when you turn your head the robot head follows (telepresence look-control) while what you see stays
  world-locked (no VR sickness from lag). Older views are kept as a dim "memory" around you, the gripper D405
  is a floating panel, the wide-angle nav camera is a peripheral backdrop, and your posture skeleton/HUD float
  in the scene.

Wire format (WebSocket, little-endian), server -> client:
  binary  [u8 type][u32 json_len][json]...  type 1 = RGB-D: [...][u32 jpeg_len][jpeg][u16 depth mm, dw*dh]
                                            type 2 = gripper jpeg, type 3 = nav camera jpeg: [...][jpeg]
  text    {"type": "state", "state": "running" | "ended", "report": "/report"}
client -> server (text): {"type": "look", "yaw": rad, "pitch": rad, "active": bool}
                         {"type": "ack", "t": msg_type, "sent": server_time}   after each binary frame is shown

Flow control: at most MAX_IN_FLIGHT[type] un-acknowledged frames per client. Without it, frames pile up in
socket buffers when Wi-Fi or the headset is slower than the robot (seconds of lag = VR sickness); with it,
stale frames are simply skipped and latency stays at one or two frames.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from . import web
from .camera import Intrinsics

log = logging.getLogger(__name__)

T_RGBD, T_GRIPPER, T_NAV = 1, 2, 3
MAX_IN_FLIGHT = {T_RGBD: 2, T_GRIPPER: 1, T_NAV: 1}
ACK_TIMEOUT_S = 3.0
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    "": "text/plain; charset=utf-8",
}


def pack(msg_type: int, meta: dict, *chunks: bytes, length_prefixed: int = 0) -> bytes:
    """[u8 type][u32 len][json] then chunks; the first ``length_prefixed`` chunks get a u32 length prefix."""
    js = json.dumps(meta, separators=(",", ":"), default=float).encode()
    parts = [struct.pack("<BI", msg_type, len(js)), js]
    for i, c in enumerate(chunks):
        if i < length_prefixed:
            parts.append(struct.pack("<I", len(c)))
        parts.append(c)
    return b"".join(parts)


def _with_rtt(pkt: bytes, rtt_ms: float) -> bytes:
    """Return the packet with ``rtt_ms`` added to its JSON header (round trip measured from acks)."""
    t, n = struct.unpack_from("<BI", pkt, 0)
    meta = json.loads(pkt[5 : 5 + n])
    meta["rtt_ms"] = round(rtt_ms)
    js = json.dumps(meta, separators=(",", ":")).encode()
    return struct.pack("<BI", t, len(js)) + js + pkt[5 + n :]


def unpack(buf: bytes) -> tuple[int, dict, bytes]:
    t, n = struct.unpack_from("<BI", buf, 0)
    meta = json.loads(buf[5 : 5 + n])
    return t, meta, buf[5 + n :]


SKELETON = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (0, 7),
    (0, 8),
    (7, 11),
    (8, 12),
]


def landmarks_3d(
    img_xy: np.ndarray, vis: np.ndarray, depth_m: np.ndarray | None, intr: Intrinsics, min_vis: float = 0.5
) -> list | None:
    """Back-project 2-D landmarks with the depth image -> camera-frame points [x right, y down, z fwd] (m)."""
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    out = []
    for (u, v), vi in zip(img_xy, vis, strict=True):
        if vi < min_vis:
            out.append(None)
            continue
        px, py = int(round(u * (w - 1))), int(round(v * (h - 1)))
        if not (0 <= px < w and 0 <= py < h):
            out.append(None)
            continue
        win = depth_m[max(0, py - 3) : py + 4, max(0, px - 3) : px + 4]
        valid = win[(win > 0.2) & (win < 8.0)]
        if valid.size < 4:
            out.append(None)
            continue
        z = float(np.median(valid))
        sx, sy = intr.w / w, intr.h / h  # depth and intrinsics may be different sizes
        x = (px * sx - intr.cx) / intr.fx * z
        y = (py * sy - intr.cy) / intr.fy * z
        out.append([round(x, 3), round(y, 3), round(z, 3)])
    return out


def region_levels(m: dict, th: dict) -> dict:
    def lvl(v, t):
        if v is None or not t:
            return -1
        r = abs(v) / t
        return 0 if r < 0.7 else (1 if r < 1.0 else 2)

    return {
        "neck": lvl(m.get("neck_flexion"), th.get("neck_flexion")),
        "trunk": lvl(m.get("trunk_flexion"), th.get("trunk_flexion")),
        "l_arm": lvl(m.get("l_upper_arm_elevation"), th.get("upper_arm_elevation")),
        "r_arm": lvl(m.get("r_upper_arm_elevation"), th.get("upper_arm_elevation")),
    }


# ------------------------------------------------------------------------------------------- aux cameras
def find_v4l2_device(name_hint: str) -> str | None:
    """First capture node (/dev/videoN, index 0) whose V4L2 name contains any of the '|'-separated hints."""
    hints = [h.strip().lower() for h in name_hint.split("|") if h.strip()]
    base = Path("/sys/class/video4linux")
    if not base.is_dir():
        return None
    cands = []
    for d in base.iterdir():
        try:
            name = (d / "name").read_text().strip().lower()
            idx = int((d / "index").read_text().strip()) if (d / "index").exists() else 0
        except (OSError, ValueError):
            continue
        if idx == 0 and "realsense" not in name and any(h in name for h in hints):
            cands.append(int(d.name.replace("video", "")))
    return f"/dev/video{min(cands)}" if cands else None


class AuxCamera(threading.Thread):
    """Grabs JPEG frames from the gripper D405 (pyrealsense2) or the head wide-angle nav camera (V4L2)."""

    def __init__(self, kind: str, cfg: dict, on_frame):
        super().__init__(daemon=True, name=f"aux-{kind}")
        self.kind, self.cfg, self.on_frame = kind, cfg, on_frame
        self.running = True

    def run(self):
        try:
            if self.kind == "gripper":
                self._run_realsense()
            else:
                self._run_v4l2()
        except Exception as e:
            log.warning("%s camera unavailable: %s", self.kind, e)

    def _emit(self, frame, meta):
        rot = {
            "cw90": cv2.ROTATE_90_CLOCKWISE,
            "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "180": cv2.ROTATE_180,
        }.get(str(self.cfg.get(f"{self.kind}_rotate", "none")))
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.get("jpeg_quality", 70))]
        )
        if ok:
            h, w = frame.shape[:2]
            self.on_frame(
                T_GRIPPER if self.kind == "gripper" else T_NAV,
                pack(T_GRIPPER if self.kind == "gripper" else T_NAV, {"w": w, "h": h, **meta}, buf.tobytes()),
            )

    def _run_realsense(self):
        import pyrealsense2 as rs

        want = str(self.cfg.get("gripper_name", "D405"))
        devs = [
            (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            for d in rs.context().query_devices()
        ]
        match = [d for d in devs if want.lower() in d[0].lower()]
        if not match:
            raise RuntimeError(f"no RealSense matching {want!r} (found {devs})")
        pipe, c = rs.pipeline(), rs.config()
        c.enable_device(match[0][1])
        c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        pipe.start(c)
        log.info("VR: gripper camera %s streaming", match[0][0])
        period = 1.0 / float(self.cfg.get("aux_fps", 10))
        last = 0.0
        try:
            while self.running:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                col = frames.get_color_frame()
                if col and time.monotonic() - last >= period:
                    last = time.monotonic()
                    self._emit(np.asanyarray(col.get_data()), {"name": "gripper"})
        finally:
            pipe.stop()

    def _run_v4l2(self):
        dev = self.cfg.get("nav_camera", "auto")
        if dev in (True, "auto", None):
            dev = find_v4l2_device(str(self.cfg.get("nav_name", "vitade|navigation|wide|arducam")))
        if not dev:
            raise RuntimeError("no wide-angle navigation camera found (set vr.nav_camera: /dev/videoN)")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("nav_width", 800)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("nav_height", 600)))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {dev}")
        log.info("VR: navigation camera %s streaming", dev)
        period = 1.0 / float(self.cfg.get("aux_fps", 10))
        meta = {
            "name": "nav",
            "hfov": float(self.cfg.get("nav_hfov_deg", 120)),
            "vfov": float(self.cfg.get("nav_vfov_deg", 90)),
        }
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                self._emit(frame, meta)
                time.sleep(max(0.0, period - 0.01))
        finally:
            cap.release()


# ------------------------------------------------------------------------------------------- server
@dataclass(eq=False)  # identity hash: clients live in a set
class _Client:
    conn: object
    event: asyncio.Event
    sent: dict = field(default_factory=dict)
    in_flight: dict = field(default_factory=dict)  # type -> list of send times (monotonic)
    rtt_ms: float = 0.0


class VRServer:
    def __init__(
        self,
        cfg: dict,
        sessions_dir: str | Path,
        certs_dir: str | Path | None = None,
        thresholds: dict | None = None,
    ):
        self.c = cfg
        self.sessions_dir = Path(sessions_dir)
        self.th = thresholds or {}
        self.fps = float(cfg.get("fps", 15))
        self.decim = max(1, int(cfg.get("depth_decimation", 4)))
        self.quality = int(cfg.get("jpeg_quality", 70))
        self.scheme = "https" if cfg.get("https", True) else "http"
        self._web_root = resources.files("stretch_posture_coach").joinpath("web")
        self._latest: dict | None = None
        self._new = threading.Event()
        self._packets: dict[int, bytes] = {}
        self._clients: set[_Client] = set()
        self._state = {"type": "state", "state": "running", "report": None}
        self.report_path: Path | None = None
        self._look: dict | None = None
        self._alarm = {"type": "alarm", "state": "ok", "active": False, "reason": ""}
        self.on_ack = None  # callable(who): a viewer acknowledged the fall alarm
        self.snapshot_provider = None  # callable() -> jpeg bytes | None
        self._aux: list[AuxCamera] = []
        self._aux_started = False
        self._running = True

        self._ssl = None
        if self.scheme == "https":
            from .certs import ensure_cert

            cert, key = ensure_cert(certs_dir or "certs")
            self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl.load_cert_chain(cert, key)

        self.loop = asyncio.new_event_loop()
        started = threading.Event()
        self._start_error: Exception | None = None
        self._thread = threading.Thread(target=self._run_loop, args=(started,), daemon=True, name="vr-ws")
        self._thread.start()
        started.wait(10)
        if self._start_error:
            raise self._start_error
        self._encoder = threading.Thread(target=self._encode_loop, daemon=True, name="vr-encode")
        self._encoder.start()

    # ---------------------------------------------------------------- public (called from the pipeline thread)
    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def publish(
        self,
        frame: np.ndarray,
        depth_m: np.ndarray | None,
        intr: Intrinsics,
        yaw: float,
        pitch: float,
        pose=None,
        metrics: dict | None = None,
        hud: dict | None = None,
        min_vis: float = 0.5,
    ) -> None:
        """Cheap: stores references; encoding happens on the encoder thread only when someone is watching."""
        if not self._clients:
            return
        self._latest = {
            "frame": frame,
            "depth": depth_m,
            "intr": intr,
            "yaw": yaw,
            "pitch": pitch,
            "pose": pose,
            "metrics": metrics or {},
            "hud": hud or {},
            "min_vis": min_vis,
            "t": time.time(),
        }
        self._new.set()

    def look_target(self, max_age_s: float = 1.0) -> tuple[float, float] | None:
        """Latest head direction requested by the VR viewer (yaw, pitch in rad), if active and fresh."""
        lk = self._look
        if not lk or not lk.get("active") or time.monotonic() - lk["t"] > max_age_s:
            return None
        return float(lk["yaw"]), float(lk["pitch"])

    def set_alarm(self, st: dict) -> None:
        """Fall-detector state; pushed to viewers only when it changes."""
        new = {"type": "alarm", **st}
        if new != self._alarm:
            self._alarm = new
            self.loop.call_soon_threadsafe(self._wake_all)

    def set_ended(self, report_path: Path | None) -> None:
        self.report_path = report_path
        self._state = {"type": "state", "state": "ended", "report": "/report" if report_path else "/sessions"}
        self.loop.call_soon_threadsafe(self._wake_all)

    def close(self) -> None:
        self._running = False
        self._new.set()
        for cam in self._aux:
            cam.running = False

        async def _shutdown():
            self._server.close()
            await self._server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)

    # ---------------------------------------------------------------- encoding
    def _encode_loop(self):
        period = 1.0 / max(self.fps, 1.0)
        last = 0.0
        while self._running:
            self._new.wait(timeout=1.0)
            self._new.clear()
            if not self._running:
                return
            item = self._latest
            if item is None or not self._clients:
                continue
            wait = period - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
                item = self._latest or item
            last = time.monotonic()
            try:
                pkt = self._encode(item)
            except Exception as e:  # never kill the encoder
                log.debug("VR encode failed: %s", e)
                continue
            self.loop.call_soon_threadsafe(self._set_packet, T_RGBD, pkt)

    def _encode(self, it: dict) -> bytes:
        frame, intr = it["frame"], it["intr"]
        h, w = frame.shape[:2]
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        dw, dh = max(2, w // self.decim), max(2, h // self.decim)
        depth = it["depth"]
        if depth is not None and depth.shape[:2] == (h, w):
            d = cv2.resize(depth, (dw, dh), interpolation=cv2.INTER_NEAREST)
            dmm = np.clip(np.nan_to_num(d * 1000.0), 0, 65535).astype("<u2")
            has_depth = True
        else:  # no depth: show the image as a flat screen 2.5 m away
            dmm = np.full((dh, dw), 2500, "<u2")
            has_depth = False
        skel = None
        pose = it["pose"]
        if pose is not None and has_depth:
            skel = landmarks_3d(pose.image, pose.visibility, depth, intr, it["min_vis"])
        meta = {
            "t": it["t"],
            "w": w,
            "h": h,
            "dw": dw,
            "dh": dh,
            "has_depth": has_depth,
            "intr": [intr.fx / intr.w, intr.fy / intr.h, (intr.cx + 0.5) / intr.w, (intr.cy + 0.5) / intr.h],
            "yaw": it["yaw"],
            "pitch": it["pitch"],
            "skeleton": skel,
            "edges": SKELETON,
            "levels": region_levels(it["metrics"], self.th) if skel else None,
            "hud": it["hud"],
            "max_range": float(self.c.get("max_range_m", 6.0)),
            "camera_height": float(self.c.get("camera_height_m", 1.3)),
            "head_control_default": bool(self.c.get("head_control", True)),
        }
        return pack(T_RGBD, meta, jpg.tobytes(), dmm.tobytes(), length_prefixed=1)

    # ---------------------------------------------------------------- asyncio side
    def _set_packet(self, t: int, pkt: bytes):
        self._packets[t] = pkt
        self._wake_all()

    def _wake_all(self):
        for c in self._clients:
            c.event.set()

    def _on_aux(self, t: int, pkt: bytes):
        self.loop.call_soon_threadsafe(self._set_packet, t, pkt)

    def _start_aux(self):
        if self._aux_started:
            return
        self._aux_started = True
        if self.c.get("gripper_camera", "auto") not in (False, "false", "off"):
            self._aux.append(AuxCamera("gripper", self.c, self._on_aux))
        if self.c.get("nav_camera", "auto") not in (False, "false", "off"):
            self._aux.append(AuxCamera("nav", self.c, self._on_aux))
        for cam in self._aux:
            cam.start()

    def _run_loop(self, started: threading.Event):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start())
        except Exception as e:
            self._start_error = e
            started.set()
            return
        started.set()
        self.loop.run_forever()

    async def _start(self):
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._handler,
            self.c.get("host", "0.0.0.0"),
            int(self.c.get("port", 8443)),
            ssl=self._ssl,
            process_request=self._http,
            compression=None,
            max_size=1 << 16,
            ping_interval=10,
            ping_timeout=20,
        )
        self.port = list(self._server.sockets)[0].getsockname()[1]

    def _response(self, status: int, ctype: str, body: bytes):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        reason = {200: "OK", 204: "No Content", 302: "Found", 404: "Not Found"}.get(status, "OK")
        # Connection: close - websockets closes plain-HTTP connections after the response; without this header
        # browsers try to reuse the socket for the next page (e.g. the report) and hang until it times out.
        headers = Headers(
            [
                ("Content-Type", ctype),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
                ("Connection", "close"),
            ]
        )
        return Response(status, reason, headers, body)

    def _http(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None  # proceed with the WebSocket handshake
        if path == "/alarm/snapshot.jpg":
            snap = self.snapshot_provider() if self.snapshot_provider else None
            return (
                self._response(200, "image/jpeg", snap)
                if snap
                else self._response(404, "text/plain", b"none")
            )
        if path == "/favicon.ico":
            return self._response(204, "image/x-icon", b"")
        if path in ("/", "/index.html"):
            return self._response(
                200, CONTENT_TYPES[".html"], self._web_root.joinpath("vr.html").read_bytes()
            )
        if path == "/vr.js":
            return self._response(200, CONTENT_TYPES[".js"], self._web_root.joinpath("vr.js").read_bytes())
        if path.startswith("/vendor/"):
            name = path[len("/vendor/") :]
            if "/" not in name and not name.startswith("."):
                f = self._web_root.joinpath("vendor", name)
                if f.is_file():
                    return self._response(
                        200, CONTENT_TYPES.get(Path(name).suffix, CONTENT_TYPES[""]), f.read_bytes()
                    )
        res = web.route(path, self.sessions_dir, self.report_path, '<a href="/">VR view</a>')
        if res is not None:
            return self._response(*res)
        return self._response(404, "text/plain", b"not found")

    async def _handler(self, conn):
        client = _Client(conn, asyncio.Event())
        self._clients.add(client)
        self._start_aux()
        log.info("VR viewer connected (%d)", len(self._clients))
        client.event.set()
        sender = asyncio.ensure_future(self._sender(client))
        try:
            async for msg in conn:
                if isinstance(msg, str):
                    try:
                        m = json.loads(msg)
                    except ValueError:
                        continue
                    if m.get("type") == "alarm_ack":
                        if self.on_ack:
                            self.on_ack("VR viewer")
                    elif m.get("type") == "ack":
                        q = client.in_flight.get(int(m.get("t", 0)), [])
                        if q:
                            client.rtt_ms = 0.8 * client.rtt_ms + 0.2 * (time.monotonic() - q.pop(0)) * 1000
                        client.event.set()
                    elif m.get("type") == "look":
                        self._look = {
                            "yaw": float(m.get("yaw", 0)),
                            "pitch": float(m.get("pitch", 0)),
                            "active": bool(m.get("active")),
                            "t": time.monotonic(),
                        }
        except Exception:
            pass
        finally:
            sender.cancel()
            self._clients.discard(client)
            if not self._clients:
                self._look = None
            log.info("VR viewer disconnected (%d)", len(self._clients))

    async def _sender(self, client: _Client):
        conn = client.conn
        try:
            while True:
                await client.event.wait()
                client.event.clear()
                if client.sent.get("state") != self._state:
                    await conn.send(json.dumps(self._state))
                    client.sent["state"] = dict(self._state)
                if client.sent.get("alarm") != self._alarm:
                    await conn.send(json.dumps(self._alarm))
                    client.sent["alarm"] = dict(self._alarm)
                now = time.monotonic()
                for t, pkt in list(self._packets.items()):
                    if client.sent.get(t) is pkt:
                        continue
                    q = client.in_flight.setdefault(t, [])
                    if q and now - q[0] > ACK_TIMEOUT_S:  # lost ack: don't stall forever
                        q.clear()
                    if len(q) >= MAX_IN_FLIGHT.get(t, 1):
                        continue  # wait for the headset to catch up; newer frames replace this one
                    client.sent[t] = pkt  # remember the original object (identity check above)
                    q.append(time.monotonic())
                    await conn.send(_with_rtt(pkt, client.rtt_ms) if t == T_RGBD and client.rtt_ms else pkt)
        except Exception:
            return
