"""Main loop: camera -> MediaPipe -> biomechanics -> assessment -> coach (LLM + speech) / head tracking / UI / log."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

from . import biomech
from .assessment import Assessor
from .biomech import Calibration, Subject, compute_metrics, gravity_from_pitch
from .camera import open_source
from .coach import Coach
from .llm import OllamaClient, OllamaError
from .logger import SessionLogger
from .pose import PoseEstimator
from .robot import make_head, wait_for_tilt
from .speech import Speaker

log = logging.getLogger(__name__)

TORSO_IDX = [11, 12, 23, 24]


def load_calibration(path: str | Path) -> Calibration:
    p = Path(path)
    if p.exists():
        try:
            return Calibration.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError) as e:
            log.warning("Ignoring unreadable calibration %s: %s", p, e)
    return Calibration()


def subject_from_cfg(cfg: dict) -> Subject:
    s = cfg.get("subject", {})
    return Subject(
        float(s.get("body_mass_kg", 70)),
        float(s.get("hand_load_kg", 0)),
        bool(s.get("arms_supported", False)),
    )


def make_estimator(cfg: dict) -> PoseEstimator:
    p = cfg["pose"]
    return PoseEstimator(
        p.get("models_dir", "models"),
        p.get("model", "full"),
        float(p["min_detection_confidence"]),
        float(p["min_presence_confidence"]),
        float(p["min_tracking_confidence"]),
    )


def make_llm(cfg: dict) -> OllamaClient | None:
    if not cfg["llm"].get("enabled", True):
        return None
    client = OllamaClient.from_config(cfg["llm"])
    try:
        if not client.has_model():
            log.warning(
                "Ollama is up but model %s is not pulled (run: ollama pull %s). Using rule-based messages.",
                client.model,
                client.model,
            )
            return None
        t0 = time.monotonic()
        client.warmup()
        log.info("LLM %s ready (warm-up %.1f s)", client.model, time.monotonic() - t0)
    except OllamaError as e:
        log.warning("%s. Using rule-based messages.", e)
        return None
    return client


def _want_window(cfg: dict) -> bool:
    w = cfg["display"].get("window", "auto")
    if str(w).lower() == "auto":
        return bool(os.environ.get("DISPLAY"))
    return bool(w) and str(w).lower() != "false"


def _torso_target(pose, min_vis: float):
    if pose is None:
        return None
    vis = pose.visibility[TORSO_IDX]
    if (vis >= min_vis).sum() < 2:
        return None
    pts = pose.image[TORSO_IDX][vis >= min_vis]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _gravity(head, cfg: dict) -> np.ndarray:
    tilt = head.tilt
    pitch = math.degrees(tilt) if tilt is not None else float(cfg["camera"].get("pitch_deg", 0.0))
    return gravity_from_pitch(pitch)


class Pipeline:
    def __init__(
        self,
        cfg: dict,
        source=None,
        estimator=None,
        head=None,
        speaker=None,
        llm="auto",
        window: bool | None = None,
        stream: bool | None = None,
        log_session: bool | None = None,
    ):
        self.cfg = cfg
        self.min_vis = float(cfg["pose"].get("min_visibility", 0.5))
        self.source = source if source is not None else open_source(cfg["camera"])
        self.estimator = estimator if estimator is not None else make_estimator(cfg)
        self.head = (
            head if head is not None else make_head(cfg["robot"], float(cfg["camera"].get("pitch_deg", 0)))
        )
        self.speaker = speaker if speaker is not None else Speaker(cfg["speech"])
        self.llm = make_llm(cfg) if llm == "auto" else llm
        self.subject = subject_from_cfg(cfg)
        self.calibration = load_calibration(cfg["analysis"].get("calibration_file", "calibration.json"))
        self.assessor = Assessor(cfg)
        self.logger = None
        if cfg["logging"].get("enabled", True) if log_session is None else log_session:
            self.logger = SessionLogger(
                cfg["logging"].get("sessions_dir", "sessions"), cfg, float(cfg["logging"].get("log_hz", 5))
            )
        self.coach = Coach(
            cfg, self.speaker, self.llm, on_event=self._on_event, issues=self.assessor.issues.values()
        )
        self.window = _want_window(cfg) if window is None else window
        self.streamer = None
        if cfg["display"].get("stream", True) if stream is None else stream:
            from .stream import FrameStreamer

            try:
                d = cfg["display"]
                self.streamer = FrameStreamer(
                    d.get("stream_host", "0.0.0.0"), d.get("stream_port", 8080), d.get("jpeg_quality", 75)
                )
                log.info("Live view: http://%s:%d/", os.uname().nodename, self.streamer.port)
            except OSError as e:
                log.warning("Could not start web viewer: %s", e)
        self.fps = 0.0
        self.frames = 0
        self.last_assessment = None

    def _on_event(self, ev: dict):
        if self.logger:
            self.logger.event(ev)

    def step(self, frame: np.ndarray, ts: float):
        pose = self.estimator.process(frame, ts)
        raw = None
        if pose is not None:
            raw = compute_metrics(
                pose.world,
                pose.visibility,
                _gravity(self.head, self.cfg),
                self.subject,
                self.calibration,
                self.min_vis,
            )
        a = self.assessor.update(raw, ts)
        self.coach.update(a, ts)
        self.head.track(_torso_target(pose, self.min_vis) if a.present else None, ts)
        if self.logger:
            self.logger.log(a, self.head.tilt)
        self.last_assessment = a
        return pose, a

    def run(self, duration_s: float | None = None, max_frames: int | None = None) -> Path | None:
        import cv2

        from .visualize import draw

        start = time.monotonic()
        last_print = 0.0
        prev = time.monotonic()
        if self.head.available:
            wait_for_tilt(self.head)
        self.coach.greet()
        log.info("Running. Press Ctrl+C%s to stop.", " (or q in the window)" if self.window else "")
        try:
            while True:
                if duration_s and time.monotonic() - start > duration_s:
                    break
                if max_frames and self.frames >= max_frames:
                    break
                frame, ts = self.source.read()
                if frame is None:
                    if not self.source.is_live:
                        break
                    continue
                pose, a = self.step(frame, ts)
                self.frames += 1
                now = time.monotonic()
                dt = now - prev
                prev = now
                if dt > 0:
                    self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt

                if self.window or self.streamer:
                    tilt = self.head.tilt
                    status = {
                        "fps": self.fps,
                        "last_message": self.coach.last_message,
                        "tilt_deg": "-" if tilt is None else f"{math.degrees(tilt):.0f}",
                    }
                    vis = draw(frame, pose, a, self.cfg["thresholds"], status, self.min_vis)
                    if self.streamer:
                        self.streamer.publish(vis, self._summary(a))
                    if self.window:
                        cv2.imshow("Stretch Posture Coach", vis)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                if now - last_print > 5:
                    last_print = now
                    log.info(self._status_line(a))
        except KeyboardInterrupt:
            log.info("Stopping...")
        finally:
            self.close()
        return self.logger.dir if self.logger else None

    def _summary(self, a) -> dict:
        if not a.present:
            return {"present": False, "fps": round(self.fps, 1)}
        keys = [
            "posture",
            "neck_flexion",
            "neck_side_bend",
            "trunk_flexion",
            "trunk_side_bend",
            "trunk_twist",
            "shoulder_tilt_cm",
            "max_upper_arm_elevation",
            "max_elbow_flexion",
            "max_knee_flexion",
            "l5s1_compression_n",
            "l5s1_shear_n",
            "neck_load_kg",
            "max_shoulder_moment_nm",
        ]
        return {
            "present": True,
            "fps": round(self.fps, 1),
            **{k: a.metrics.get(k) for k in keys},
            "rula": a.rula,
            "reba": a.reba,
            "static_s": round(a.static_for_s),
            "issues": [s.detail for s in a.active],
            "last_message": self.coach.last_message,
        }

    def _status_line(self, a) -> str:
        if not a.present:
            return f"[{self.fps:4.1f} fps] no person in view"
        m = a.metrics
        issues = ",".join(s.rule.id for s in a.active) or "none"
        return (
            f"[{self.fps:4.1f} fps] {m.get('posture')}: neck {m.get('neck_flexion')} trunk {m.get('trunk_flexion')}"
            f" | RULA {a.rula and a.rula['score']} REBA {a.reba and a.reba['score']}"
            f" | L5/S1 {m.get('l5s1_compression_n')} N | issues: {issues}"
        )

    def close(self):
        import cv2

        for fn in (self.speaker.close, self.head.close, self.source.close, self.estimator.close):
            try:
                fn()
            except Exception as e:
                log.debug("close: %s", e)
        if self.streamer:
            self.streamer.close()
        if self.logger:
            self.logger.close()
        if self.window:
            cv2.destroyAllWindows()


def calibrate(
    cfg: dict, seconds: float = 8.0, source=None, estimator=None, head=None, speaker=None
) -> Calibration:
    """Record a neutral posture and save offsets to ``analysis.calibration_file``."""
    source = source if source is not None else open_source(cfg["camera"])
    estimator = estimator if estimator is not None else make_estimator(cfg)
    head = head if head is not None else make_head(cfg["robot"], float(cfg["camera"].get("pitch_deg", 0)))
    speaker = speaker if speaker is not None else Speaker(cfg["speech"])
    min_vis = float(cfg["pose"].get("min_visibility", 0.5))
    subject = subject_from_cfg(cfg)
    spanish = cfg["coach"].get("language", "English").lower().startswith("span")
    try:
        if head.available:
            wait_for_tilt(head)
        speaker.say_blocking(
            "Calibración. Siéntate o párate derecho, con los hombros relajados y la vista al frente. Quédate quieto."
            if spanish
            else "Calibration. Sit or stand tall in your best neutral posture, shoulders relaxed, looking straight ahead. Hold still."
        )
        samples: list[dict] = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end or len(samples) < 5:
            if time.monotonic() > t_end + 20:
                break
            frame, ts = source.read()
            if frame is None:
                if not source.is_live:
                    break
                continue
            pose = estimator.process(frame, ts)
            if pose is None:
                continue
            m = compute_metrics(
                pose.world,
                pose.visibility,
                _gravity(head, cfg),
                subject,
                None,
                min_vis,
                apply_calibration=False,
            )
            if m.get("torso_visible") and m.get("neck_flexion") is not None:
                samples.append(m)
            head.track(_torso_target(pose, min_vis), ts)
        cal = biomech.calibration_from_samples(samples)
        if not cal.offsets:
            speaker.say_blocking(
                "No pude verte bien. Intenta de nuevo."
                if spanish
                else "I couldn't see you well enough. Please try again with your whole body in view."
            )
            raise RuntimeError(f"Calibration failed: only {len(samples)} usable frames")
        path = Path(cfg["analysis"].get("calibration_file", "calibration.json"))
        path.write_text(
            json.dumps({**cal.to_dict(), "samples": len(samples), "time": time.time()}, indent=2),
            encoding="utf-8",
        )
        speaker.say_blocking("Listo, calibración guardada." if spanish else "Done. Calibration saved.")
        log.info("Calibration saved to %s: %s", path, cal.to_dict())
        return cal
    finally:
        for fn in (speaker.close, head.close, source.close, estimator.close):
            try:
                fn()
            except Exception:
                pass
