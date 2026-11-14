"""MediaPipe Pose Landmarker (Tasks API) wrapper returning numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)

# Skeleton edges used for drawing (MediaPipe 33-landmark topology, body only).
POSE_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (15, 19),
    (17, 19),
    (16, 18),
    (16, 20),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (27, 31),
    (28, 30),
    (30, 32),
    (28, 32),
    (0, 7),
    (0, 8),
    (9, 10),
]


def model_path(models_dir: str | Path, variant: str) -> Path:
    return Path(models_dir) / f"pose_landmarker_{variant}.task"


@dataclass
class PoseResult:
    image: np.ndarray  # (33, 2) normalized image coords (x, y in 0..1)
    world: np.ndarray  # (33, 3) meters, origin at hip center
    visibility: np.ndarray  # (33,)


class PoseEstimator:
    def __init__(
        self,
        models_dir: str | Path = "models",
        variant: str = "full",
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = model_path(models_dir, variant)
        if not path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {path}. Run `posture-coach download-models` (or ./install.sh)."
            )
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, bgr: np.ndarray, timestamp_s: float) -> PoseResult | None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(int(timestamp_s * 1000), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        res = self._landmarker.detect_for_video(image, ts)
        if not res.pose_landmarks or not res.pose_world_landmarks:
            return None
        lms, wlms = res.pose_landmarks[0], res.pose_world_landmarks[0]
        img = np.array([[lm.x, lm.y] for lm in lms], dtype=float)
        vis = np.array([lm.visibility if lm.visibility is not None else 1.0 for lm in lms], dtype=float)
        world = np.array([[lm.x, lm.y, lm.z] for lm in wlms], dtype=float)
        return PoseResult(image=img, world=world, visibility=vis)

    def close(self):
        self._landmarker.close()


def download_models(
    models_dir: str | Path, variants=("lite", "full", "heavy"), force: bool = False
) -> list[Path]:
    import requests

    out = []
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for v in variants:
        dest = model_path(models_dir, v)
        if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
            out.append(dest)
            continue
        url = MODEL_URL.format(v=v)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        out.append(dest)
    return out
