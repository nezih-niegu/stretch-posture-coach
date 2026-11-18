"""Main loop: camera -> MediaPipe -> biomechanics -> assessment -> coach (LLM + speech) / head tracking / UI / log."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import biomech
from .alarm import Alarm
from .assessment import Assessor
from .biomech import Calibration, Subject, compute_metrics, gravity_from_pitch
from .camera import Intrinsics, open_source
from .cbf import SafeBase
from .coach import Coach
from .fall import ALARM, CHECK, IMPACT, OK, FallDetector, features_from, hip_height_from_depth
from .follow import Follower, person_distance
from .llm import OllamaClient, OllamaError
from .logger import SessionLogger
from .modes import POLICIES, Ev, Mode, ModeMachine
from .pose import PoseEstimator
from .robot import make_head, wait_for_tilt
from .scene import SceneRecorder
from .speech import Speaker
from .survey import Survey
from .world import LiveWorld

log = logging.getLogger(__name__)
_REPORT_LOCK = threading.Lock()  # matplotlib report building, one at a time

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
        vr: bool | None = None,
        keep_servers: bool = False,
        world=None,
    ):
        self.cfg = cfg
        self.keep_servers = keep_servers
        self.min_vis = float(cfg["pose"].get("min_visibility", 0.5))
        self.source = source if source is not None else open_source(cfg["camera"])
        self.estimator = estimator if estimator is not None else make_estimator(cfg)
        want_follow = bool(cfg.get("follow", {}).get("enabled", False))
        vcfg = cfg.get("vr", {})
        want_vr = vcfg.get("enabled", True) if vr is None else vr
        tcfg = cfg.get("teleop", {})
        want_teleop = bool(want_vr and tcfg.get("enabled", True))
        self.head = (
            head
            if head is not None
            else make_head(
                cfg["robot"], float(cfg["camera"].get("pitch_deg", 0)), base=want_follow or want_teleop
            )
        )
        # follow mode's base steps go through the Control-Barrier-Function filter (lidar obstacles)
        self.safe_base = SafeBase(self.head)
        self.follower = Follower(cfg.get("follow", {}), cfg["robot"], self.safe_base)
        self.front_pan = float(cfg.get("follow", {}).get("forward_pan_rad", 0.0))
        self.front_tilt = float(cfg["robot"].get("front_tilt_rad", -0.15))
        if hasattr(self.head, "look"):  # the camera always starts looking to the front
            self.head.look(self.front_pan, self.front_tilt, 0.0)
        if want_follow and not self.follower.enabled:
            log.warning(
                "Follow mode requested but the mobile base is not available - only the head will track."
            )
        elif self.follower.enabled:
            log.info(
                "FOLLOW MODE ON: keeping %.1f-%.1f m. Keep the floor clear and the runstop in reach.",
                self.follower._p("min_m", 2.0),
                self.follower._p("max_m", 3.0),
            )
        self.follow_state = None
        if not want_follow:
            log.info("Follow mode OFF: the base will not move (use --follow, or set follow.enabled: true)")
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
        self.fall = FallDetector(cfg.get("fall", {}))
        self._fall_ack: str | None = None  # set from the web/VR threads
        self.alarm = Alarm(
            cfg.get("alarm", {}),
            self.speaker,
            self.coach.templates,
            live_url=f"http://{_host_ip()}:{cfg['display'].get('stream_port', 8080)}/",
            snapshot_dir=self.logger.dir if self.logger else None,
            spanish=self.coach.persona.spanish,
        )
        self.window = _want_window(cfg) if window is None else window
        sessions_dir = cfg["logging"].get("sessions_dir", "sessions")
        # registered world model: lidar SLAM + head joints + floor plane -> live map, heat map, captions
        self.world = world if world is not None else LiveWorld(cfg, self.head, self.follower)
        self.safe_base.world = self.world
        self.world.new_session(self.logger.dir if self.logger else None)
        # who is in charge (survey / autonomous coaching / expert teleop / fall response / runstop)
        self.fsm = ModeMachine(Mode.AUTONOMOUS)
        self.fsm.on_change(self._on_mode)
        self.survey = Survey(
            cfg, self.head, self.world, say=lambda key: self.coach.announce(key, time.monotonic(), 0)
        )
        self._survey_start = False
        self._runstop = None
        self.vr = None
        if want_vr:
            from .vr import VRServer

            try:
                self.vr = VRServer(
                    {**vcfg, "teleop": {**tcfg, "enabled": want_teleop}},
                    sessions_dir,
                    vcfg.get("certs_dir", "certs"),
                    cfg["thresholds"],
                )
                log.info(
                    "VR view (Meta Quest browser): %s://%s:%d/  (accept the certificate warning once)",
                    self.vr.scheme,
                    _host_ip(),
                    self.vr.port,
                )
            except Exception as e:
                log.warning("Could not start VR server: %s", e)
        self.teleop = None
        if self.vr is not None:
            self.vr.attach_world(self.world)
            if want_teleop:
                from .teleop import TeleopController

                self.teleop = TeleopController(
                    tcfg, self.head, self.world, self.vr, allowed=lambda: self.fsm.mode == Mode.TELEOP
                )
                self.teleop.start()
                if not getattr(self.head, "has_base", False):
                    log.info("Expert teleop: no mobile base/arm available (head-only or no robot)")
                if self.vr.require_pin:
                    log.warning(
                        "EXPERT TELEOP PIN: %s  (enter it on the VR page to drive the base and arm)",
                        self.vr.pin,
                    )
        self.vr_control = False
        self.voice = None
        if self.vr is not None:
            self.vr.status_provider = self._status_for_vr
            if cfg.get("voice", {}).get("enabled", True):
                from .voice import VoiceRelay

                self.voice = VoiceRelay(
                    cfg,
                    speak=self._expert_say,
                    llm=self.llm,
                    notify=self.vr.voice_notify,
                    context=lambda: self.last_assessment,
                    log_event=self._on_event,
                )
                self.vr.on_voice = self.voice
                self.voice.preload()
        self.streamer = None
        if cfg["display"].get("stream", True) if stream is None else stream:
            from .stream import FrameStreamer

            try:
                d = cfg["display"]
                self.streamer = FrameStreamer(
                    d.get("stream_host", "0.0.0.0"),
                    d.get("stream_port", 8080),
                    d.get("jpeg_quality", 75),
                    sessions_dir=sessions_dir,
                    vr_port=self.vr.port if self.vr else None,
                )
                log.info("Live view: http://%s:%d/", _host_ip(), self.streamer.port)
            except OSError as e:
                log.warning("Could not start web viewer: %s", e)
        self.scene_rec = self._new_scene_recorder()
        for srv in (self.streamer, self.vr):  # fall alarm: "I'm OK" buttons + snapshot; End-session button
            if srv is not None:
                srv.on_ack = self.request_fall_ack
                srv.on_end_session = self.request_end_session
                srv.snapshot_provider = lambda: self.alarm.snapshot
        self.fps = 0.0
        self.frames = 0
        self.last_assessment = None
        self._new_session_mode()

    # ------------------------------------------------------------------ modes
    def _new_session_mode(self) -> None:
        """Every session starts with a 360° survey that re-calibrates the map (if there is a robot to move)."""
        sc = self.cfg.get("survey", {})
        if sc.get("enabled", True) and getattr(self.head, "available", False):
            self.fsm.dispatch(Ev.NEW_SESSION, session=self.logger.dir.name if self.logger else None)
        else:
            log.info(
                "360° survey skipped (%s)", "disabled" if not sc.get("enabled", True) else "no robot head"
            )

    def _on_mode(self, old: Mode, new: Mode, event: Ev, info: dict) -> None:
        if old == Mode.SURVEY and new != Mode.SURVEY:
            self.survey.abort()
        if new == Mode.SURVEY:
            self._survey_start = True  # started from step() with the frame clock
        if new == Mode.TELEOP:
            self.coach.announce("teleop_on", time.monotonic(), 0)
        if POLICIES[new].coaching == "speak" and POLICIES[old].coaching != "speak":
            self.coach.mute(getattr(self.coach, "_t", 0.0), 5.0)  # short grace before tips resume
        if old == Mode.TELEOP and hasattr(self.head, "drive"):
            self.head.drive(0.0, 0.0)  # leaving expert control: make sure the base is stopped
        if self.logger:
            self.logger.event(
                {
                    "kind": "mode",
                    "from": old.value,
                    "to": new.value,
                    "event": event.value,
                    "wall_time": time.time(),
                    **info,
                }
            )

    def _status_for_vr(self) -> dict:
        return {
            "mode": self.fsm.status(),
            "survey": self.survey.status(),
            "voice": None
            if self.voice is None
            else {"mode": self.voice.mode, "language": self.voice.language},
        }

    def _expert_say(self, text: str) -> None:
        self.coach.last_message = text
        self.speaker.say(text)

    def _update_modes(self, ts: float) -> None:
        rs = getattr(self.head, "runstop", None)
        if rs is not None and rs != self._runstop:
            self.fsm.dispatch(Ev.RUNSTOP if rs else Ev.RUNSTOP_CLEARED)
        self._runstop = rs
        engaged = self.vr is not None and self.teleop is not None and self.vr.teleop_engaged
        if engaged and not self.fsm.is_(Mode.TELEOP, Mode.SAFE_STOP):
            self.fsm.dispatch(Ev.TELEOP_ON)
        elif not engaged and self.fsm.is_(Mode.TELEOP):
            self.fsm.dispatch(Ev.TELEOP_OFF)
        if self._survey_start and self.fsm.is_(Mode.SURVEY):
            self._survey_start = False
            self.survey.start(ts, self.logger.dir if self.logger else None)
        if self.fsm.is_(Mode.SURVEY):
            r = self.survey.step(ts)
            if r == "done":
                self.fsm.dispatch(Ev.SURVEY_DONE)
            elif r == "failed":
                self.fsm.dispatch(Ev.SURVEY_FAILED, error=(self.survey.result or {}).get("error"))

    def _on_event(self, ev: dict):
        if self.logger:
            self.logger.event(ev)
        try:  # caption it in the SLAM view, where the person is
            la = self.last_assessment
            self.world.event(ev, la.metrics if la is not None and la.present else None)
        except Exception as e:
            log.debug("caption: %s", e)

    def step(self, frame: np.ndarray, ts: float):
        who = getattr(self, "_end_session_request", None)
        if who and self.logger is not None:
            self._end_session_request = None
            self._rotate_session(who)
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
        depth = getattr(self.source, "last_depth", None)
        h_img, w_img = frame.shape[:2]
        intr = getattr(self.source, "intrinsics", None) or Intrinsics.from_fov(
            w_img, h_img, float(self.cfg.get("vr", {}).get("fallback_hfov_deg", 60))
        )
        wl = self.world.update(
            ts,
            frame,
            depth,
            intr,
            self.head.pan,
            self.head.tilt,
            pose if a.present else None,
            self.min_vis,
            a,
        )
        self._update_fall(frame, pose, raw, depth, intr, ts)
        self._update_modes(ts)
        pol = self.fsm.policy
        self.coach.scene_text = self.world.scene_text()
        if pol.coaching == "speak":
            self.coach.update(a, ts)  # (muted by _update_fall while someone is down)
        # "background" (survey, expert teleop): posture is still assessed, logged and mapped, just not spoken
        target = _torso_target(pose, self.min_vis) if a.present else None
        forward_pan = self.front_pan
        look = self.vr.look_target() if self.vr and pol.head in ("track", "vr") else None
        if look is not None and not self.vr_control:
            log.info("VR head control ON: the robot head starts at the front and follows the headset")
        elif look is None and self.vr_control:
            log.info("VR head control OFF")
        self.vr_control = look is not None
        if self.vr_control:  # telepresence: headset rotation RELATIVE to when VR started; 0, 0 = front
            yaw, pitch = look
            vc = self.cfg.get("vr", {})
            yaw *= float(vc.get("look_yaw_sign", 1))
            pitch *= float(vc.get("look_pitch_sign", 1))
            self.head.look(self.front_pan + yaw, self.front_tilt + pitch, ts)
        elif pol.head == "track":
            self.head.track(target, ts)
        # "vr" without look control: the head holds still for the expert; "survey": survey.py drives it
        dist, src = person_distance(
            pose if a.present else None,
            depth,
            self.min_vis,
            float(self.cfg["robot"].get("vfov_deg", 69)),
        )
        if pol.base != "follow" or self.vr_control or self.fall.active:
            # follow only in AUTONOMOUS, and never while someone is in VR or may have fallen
            fs = self.follower.state
            fs.distance, fs.source, fs.action = dist, src, "paused"
            fs.note = (
                "fall check / alarm"
                if self.fall.active
                else ("VR head control" if pol.base == "follow" else f"mode {self.fsm.mode.value}")
            )
        else:
            fs = self.follower.update(a.present, dist, src, target[0] if target else None, ts)
        self.follow_state = fs
        self._record_scene(ts, frame, depth, intr, pose, raw, a.present, wl)
        if self.follower.enabled:
            key = (fs.action, fs.note.split(" (")[0] if fs.action == "hold" else fs.note)
            if key != getattr(
                self, "_last_follow_key", None
            ):  # log every change of decision, with the reason
                self._last_follow_key = key
                dist_txt = "-" if fs.distance is None else f"{fs.distance:.2f} m ({fs.source})"
                log.info(
                    "follow: %-8s %s | distance %s | bearing %s deg",
                    fs.action,
                    fs.note,
                    dist_txt,
                    "-" if fs.bearing_deg is None else f"{fs.bearing_deg:+.0f}",
                )
        if self.vr is not None and self.vr.n_clients:
            pan = self.head.pan
            h, w = frame.shape[:2]
            hud = {**self._summary(a), "vr_control": self.vr_control}
            h = self.head
            extra = {
                "T": [round(float(v), 5) for v in wl["T"].reshape(-1)],
                "registered": "lidar" if self.world.lidar_ok else "odometry",
                "person": None if wl["person"] is None else [round(float(v), 3) for v in wl["person"]],
                "robot": {
                    "base": [round(float(v), 4) for v in wl["base"]],
                    "pan": h.pan,
                    "tilt": h.tilt,
                    "joints": dict(getattr(h, "joints", {}) or {}),
                    "has_base": bool(getattr(h, "has_base", False)),
                    "has_arm": bool(getattr(h, "has_arm", False)),
                    "homed": getattr(h, "homed", None),
                    "runstop": getattr(h, "runstop", None),
                },
                "world": self.world.status(),
                "mode": self.fsm.status(),
                "front": [self.front_pan, self.front_tilt],
                "teleop_state": None
                if self.teleop is None
                else {
                    k: v
                    for k, v in self.teleop.state.items()
                    if k in ("active", "note", "v", "w", "clutch", "mode")
                },
            }
            self.vr.publish(
                frame,
                depth,
                intr,
                self.follower.theta + ((pan - forward_pan) if pan is not None else 0.0),
                self.head.tilt or 0.0,
                pose if a.present else None,
                a.metrics,
                hud,
                self.min_vis,
                extra,
            )
        if self.follower.enabled and self.cfg["follow"].get("announce", True):
            if fs.action == "back_off":
                self.coach.announce("follow_back", ts)
            elif fs.action == "blocked" and "back-up" in fs.note:
                self.coach.announce("follow_limit", ts)
        fkey = (fs.action, fs.note)
        if (
            fs.action in ("rotate", "approach", "back_off")
            and self.logger
            and fkey != getattr(self, "_last_follow_ev", None)
        ):
            self.logger.event(
                {
                    "kind": "follow",
                    "action": fs.action,
                    "note": fs.note,
                    "distance_m": fs.distance,
                    "bearing_deg": fs.bearing_deg,
                    "wall_time": time.time(),
                }
            )
        self._last_follow_ev = fkey  # one event per decision change, not one per frame
        if self.logger:
            self.logger.log(a, self.head.tilt, fs)
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
                        "follow": self.follow_state,
                        "follow_enabled": self.follower.enabled,
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
        fs = self.follow_state
        follow = (
            None
            if fs is None
            else {
                "distance_m": fs.distance,
                "source": fs.source,
                "action": fs.action,
                "bearing_deg": fs.bearing_deg,
                "enabled": self.follower.enabled,
            }
        )
        if not a.present:
            return {"present": False, "fps": round(self.fps, 1), "follow": follow}
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
            "follow": follow,
        }

    def _status_line(self, a) -> str:
        fs = self.follow_state
        dist = "" if fs is None or fs.distance is None else f" | {fs.distance:.2f} m ({fs.source})"
        if self.follower.enabled and fs is not None:
            dist += f" follow:{fs.action}"
        if not a.present:
            return f"[{self.fps:4.1f} fps] no person in view{dist}"
        m = a.metrics
        issues = ",".join(s.rule.id for s in a.active) or "none"
        return (
            f"[{self.fps:4.1f} fps] {m.get('posture')}: neck {m.get('neck_flexion')} trunk {m.get('trunk_flexion')}"
            f" | RULA {a.rula and a.rula['score']} REBA {a.reba and a.reba['score']}"
            f" | L5/S1 {m.get('l5s1_compression_n')} N | issues: {issues}{dist}"
        )

    def close(self):
        import cv2

        self.alarm.stop("session ended")
        if self.teleop is not None:
            self.teleop.close()
        if self.scene_rec:
            self.scene_rec.close()
        if self.logger:
            self.world.save_occupancy(self.logger.dir)
            self.world.save_scene_graph(self.logger.dir)
        self.world.close()
        for fn in (self.speaker.close, self.head.close, self.source.close, self.estimator.close):
            try:
                fn()
            except Exception as e:
                log.debug("close: %s", e)
        if not self.keep_servers:
            self.close_servers()
        if self.logger:
            self.logger.close()
        if self.window:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ sessions
    def request_end_session(self, who: str = "web") -> dict:
        """Called from the web/VR threads: close the current session and start a new one at the next frame.
        The camera, robot and web services keep running; the report is built in the background."""
        if self.logger is None:
            return {"ok": False, "error": "session logging is disabled"}
        self._end_session_request = who
        return {"ok": True, "ending": self.logger.dir.name}

    def _rotate_session(self, who: str) -> None:
        old = self.logger
        old.event({"kind": "session_end", "by": who, "wall_time": time.time()})
        old.close()
        if self.scene_rec:
            self.scene_rec.close()
        self.logger = SessionLogger(
            self.cfg["logging"].get("sessions_dir", "sessions"),
            self.cfg,
            float(self.cfg["logging"].get("log_hz", 5)),
        )
        self.alarm.snapshot_dir = self.logger.dir
        self.world.new_session(self.logger.dir, old.dir)
        self.scene_rec = self._new_scene_recorder()
        log.info("Session %s ended (%s); new session %s started", old.dir.name, who, self.logger.dir.name)
        self._new_session_mode()
        self._set_last_report(
            {"name": old.dir.name, "status": "building", "url": f"/sessions/{old.dir.name}/"}
        )
        threading.Thread(target=self._build_report, args=(old.dir,), daemon=True, name="report").start()

    def _new_scene_recorder(self):
        sc = self.cfg.get("scene", {})
        if self.logger is None or not sc.get("record", True):
            return None
        return SceneRecorder(self.logger.dir, sc)

    def _record_scene(self, ts, frame, depth, intr, pose, raw, present, wl=None) -> None:
        if self.scene_rec is None:
            return
        h = self.head
        base = wl["base"] if wl is not None else self.world.base_pose()
        extra = {"fall": self.fall.state}
        if wl is not None:  # registered camera pose (lidar SLAM + joints + floor plane)
            extra["T_map_cam"] = [round(float(v), 5) for v in wl["T"].reshape(-1)]
            extra["registration"] = "lidar" if self.world.lidar_ok else "odometry"
        self.scene_rec.maybe_record(
            ts,
            frame,
            depth,
            intr,
            h.pan,
            h.tilt,
            base,
            pose if present else None,
            (raw or {}).get("posture"),
            self.min_vis,
            extra,
        )

    def _build_report(self, session_dir: Path) -> None:
        import multiprocessing as mp

        from .report import build_report_job, summarize

        with _REPORT_LOCK:
            try:
                # separate process: matplotlib + map fusion must not stall the camera loop or the VR stream
                proc = mp.get_context("spawn").Process(
                    target=build_report_job, args=(str(session_dir),), name="report", daemon=True
                )
                proc.start()
                proc.join(timeout=900)
                if proc.is_alive():
                    proc.terminate()
                    raise RuntimeError("report build timed out")
                if proc.exitcode != 0 or not (session_dir / "report.html").exists():
                    raise RuntimeError(f"report process exited with {proc.exitcode}")
                sm = summarize(session_dir)
                self._set_last_report(
                    {
                        "name": session_dir.name,
                        "status": "ready",
                        "time": time.time(),
                        "url": f"/sessions/{session_dir.name}/",
                        "in_view_min": sm.get("in_view_min"),
                        "rula_mean": sm.get("rula_mean"),
                        "messages": sm.get("coaching_messages"),
                    }
                )
            except Exception as e:
                log.error("report for %s failed: %s", session_dir.name, e)
                self._set_last_report({"name": session_dir.name, "status": "failed", "url": "/sessions"})

    def _set_last_report(self, info: dict) -> None:
        self.last_report = info
        for srv in (self.streamer, self.vr):
            if srv is not None:
                srv.last_report = info
                if hasattr(srv, "notify_state"):
                    srv.notify_state()

    # ------------------------------------------------------------------ fall detection
    def request_fall_ack(self, who: str) -> None:
        """Called from the web/VR server threads ("I'm OK" / caregiver acknowledged)."""
        self._fall_ack = who

    def _update_fall(self, frame, pose, raw, depth, intr, ts) -> None:
        if not self.fall.enabled:
            return
        present = bool(raw and raw.get("torso_visible"))
        hip_h = None
        if present and depth is not None:
            hip_h = hip_height_from_depth(
                pose,
                depth,
                intr,
                _gravity(self.head, self.cfg),
                float(self.cfg["camera"].get("height_m", 1.3)),
                self.min_vis,
            )
        feats = features_from(ts, present, pose, raw, hip_h, self.min_vis)
        if present:  # scene graph: what is under the person (sofa / bed is not the floor)
            feats.support = (self.world.last.get("scene") or {}).get("support")
        change = self.fall.update(feats)
        # never give posture tips to someone who is down: while lying, during a fall check/alarm, and for a
        # quiet period after it clears (the "are you OK?" check and the alarm still speak)
        fcfg = self.cfg.get("fall", {})
        if self.fall.state != OK:
            self.coach.mute(ts, float(fcfg.get("quiet_after_s", 30)))
        elif self.fall.is_down(feats):
            self.coach.mute(ts, float(fcfg.get("quiet_lying_s", 10)))
        if self._fall_ack:
            who, self._fall_ack = self._fall_ack, None
            change = self.fall.acknowledge(ts, who) or change
        if change:
            self._on_fall_change(*change, frame=frame, present=present, ts=ts)
        self._publish_alarm()

    def _on_fall_change(self, state: str, reason: str, frame, present: bool, ts: float) -> None:
        t = self.coach.templates
        if state in (CHECK, ALARM):  # name the place from the scene graph ("near table 3, hidden behind ...")
            where = self.world.fall_where()
            if where and where not in reason:
                reason = f"{reason} ({where})"
        text = ""
        if state == IMPACT:
            log.info("fall? rapid descent: %s", reason)
        elif state == CHECK:
            self.fsm.dispatch(Ev.FALL, state="check")
            text = t.get("fall_check", "")
            self.speaker.say(text)
            self.alarm.save_snapshot(frame, "check")
            if self.cfg.get("alarm", {}).get("notify_on_check", True):
                self.alarm.notify_async("check", reason)
                self._fall_notified = True
            if not present and self.head.tilt is not None:  # look down where they were
                self.head.look(self.head.pan or 0.0, max(-1.2, self.head.tilt - 0.5), ts)
            log.warning("FALL CHECK: %s", reason)
        elif state == ALARM:
            self.fsm.dispatch(Ev.FALL, state="alarm")
            text = t.get("fall_alarm", "")
            self.alarm.save_snapshot(frame, "alarm")
            self.alarm.start(reason)  # also sends the urgent notification
            self._fall_notified = True
        elif state == OK:
            self.fsm.dispatch(Ev.FALL_CLEARED)
            was_alarm = self.alarm.active
            self.coach.mute(ts, float(self.cfg.get("fall", {}).get("quiet_after_s", 30)))
            self.alarm.stop(reason)
            if was_alarm or getattr(self, "_fall_notified", False):  # tell the phone it's over
                self.alarm.notify_async("resolved", reason)
            self._fall_notified = False
            if "sat down" not in reason:  # false start (sat down fast): say nothing
                text = t.get("fall_ok", "")
                self.speaker.say(text)
            log.info("fall %s: %s", "alarm cleared" if was_alarm else "check cleared", reason)
        if state != IMPACT:
            self._on_event(
                {
                    "kind": "fall",
                    "state": state,
                    "reason": reason,
                    "text": text,
                    "source": "fall-detector",
                    "wall_time": time.time(),
                }
            )
        elif self.logger:
            self.logger.event(
                {
                    "kind": "fall",
                    "state": state,
                    "reason": reason,
                    "source": "fall-detector",
                    "wall_time": time.time(),
                }
            )

    def _publish_alarm(self) -> None:
        st = {
            "state": self.fall.state,
            "reason": self.fall.reason,
            "active": self.fall.active,
            "siren": self.alarm.active,
        }
        if self.streamer:
            self.streamer.alarm = st
        if self.vr:
            self.vr.set_alarm(st)

    def show_report(self, report_path: Path | None) -> None:
        """Session over: web viewer and VR page switch to the report."""
        if self.streamer:
            self.streamer.set_ended(report_path)
        if self.vr:
            self.vr.set_ended(report_path)

    def close_servers(self) -> None:
        for srv in (self.streamer, self.vr):
            if srv is not None:
                try:
                    srv.close()
                except Exception as e:
                    log.debug("close server: %s", e)
        self.streamer = self.vr = None


def _host_ip() -> str:
    from .certs import local_ips

    ips = [ip for ip in local_ips() if not ip.startswith("127.")]
    return ips[0] if ips else os.uname().nodename


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
    spanish = (
        cfg["coach"].get("language", "English").lower().startswith(("span", "espa"))
        or cfg["coach"].get("persona") == "norteno"
    )
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
