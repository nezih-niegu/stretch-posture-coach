"""Frame sources: Stretch head RealSense (D435i), any V4L2 webcam, or a video file."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

_ROTATIONS = {
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


class CameraError(RuntimeError):
    pass


class FrameSource:
    """Yields ``(bgr_frame, timestamp_s)``; frames are already rotated upright."""

    is_live = True
    last_depth: np.ndarray | None = (
        None  # meters (float32), aligned to the color frame and rotated the same way
    )

    def __init__(self, rotate: str = "none"):
        self._rot = _ROTATIONS.get(str(rotate).lower())

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        return cv2.rotate(frame, self._rot) if self._rot is not None else frame

    def read(self) -> tuple[np.ndarray | None, float]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RealSenseSource(FrameSource):
    def __init__(
        self,
        name_filter: str = "D435",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        rotate: str = "cw90",
        depth: bool = True,
    ):
        super().__init__(rotate)
        try:
            import pyrealsense2 as rs
        except ImportError as e:  # pragma: no cover - hardware path
            raise CameraError("pyrealsense2 is not installed") from e
        self._rs = rs
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise CameraError(
                "No RealSense camera found. If ROS 2 (stretch_driver / realsense node) is running it holds the "
                "camera - stop it first (e.g. `stretch_free_robot_process.py`)."
            )
        names = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in devices]
        match = [n for n in names if name_filter.lower() in n[0].lower()]
        if not match:
            raise CameraError(f"No RealSense matching {name_filter!r}. Found: {names}")
        self.device_name, serial = match[0]
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._depth = bool(depth)
        if self._depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = self._pipe.start(cfg)
        except RuntimeError as e:  # pragma: no cover - hardware path
            raise CameraError(
                f"Could not start {self.device_name} ({serial}): {e}. Is another process using it?"
            ) from e
        if self._depth:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self._align = rs.align(rs.stream.color)

    def read(self):
        frames = self._pipe.wait_for_frames(timeout_ms=5000)
        if self._depth:
            frames = self._align.process(frames)
            d = frames.get_depth_frame()
            if d:
                depth_m = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale
                self.last_depth = self._rotate(depth_m)
        color = frames.get_color_frame()
        if not color:
            return None, time.monotonic()
        return self._rotate(np.asanyarray(color.get_data()).copy()), time.monotonic()

    def close(self):
        try:
            self._pipe.stop()
        except Exception:
            pass


class OpenCVSource(FrameSource):
    def __init__(
        self, src: int | str, width: int = 640, height: int = 480, fps: int = 30, rotate: str = "none"
    ):
        super().__init__(rotate)
        self.is_live = not isinstance(src, str)
        self._cap = cv2.VideoCapture(src)
        if self.is_live:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open video source {src!r}")
        self._file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = 0
        self.device_name = f"video:{src}"

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None, time.monotonic()
        self._n += 1
        ts = time.monotonic() if self.is_live else self._n / self._file_fps
        return self._rotate(frame), ts

    def close(self):
        self._cap.release()


def open_source(cam_cfg: dict, source: str | None = None) -> FrameSource:
    """Create a frame source from the ``camera`` config section (``source`` overrides cam_cfg['source'])."""
    src = str(source or cam_cfg.get("source", "realsense"))
    w, h, fps = int(cam_cfg.get("width", 640)), int(cam_cfg.get("height", 480)), int(cam_cfg.get("fps", 30))
    rotate = cam_cfg.get("rotate", "none")
    if src == "realsense":
        return RealSenseSource(
            cam_cfg.get("realsense_name", "D435"), w, h, fps, rotate, bool(cam_cfg.get("depth", True))
        )
    if src == "webcam" or src.startswith("webcam:") or src.isdigit():
        idx = (
            int(src.split(":", 1)[1])
            if ":" in src
            else (int(src) if src.isdigit() else int(cam_cfg.get("webcam_index", 0)))
        )
        # webcams are not mounted sideways; only rotate if the user explicitly configured it for this source
        return OpenCVSource(idx, w, h, fps, cam_cfg.get("webcam_rotate", "none"))
    if Path(src).exists():
        return OpenCVSource(src, rotate=cam_cfg.get("file_rotate", "none"))
    raise CameraError(
        f"Unknown camera source {src!r} (use realsense, webcam, webcam:N, or a video file path)"
    )
