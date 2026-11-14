"""Fall alarm: siren + spoken alert through the Stretch speaker, phone push (ntfy) / webhook, snapshot."""

from __future__ import annotations

import io
import json
import logging
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

from .speech import Speaker, _play_wav

log = logging.getLogger(__name__)


TITLES_EN = {
    "check": ("Stretch: possible fall", "high", "warning"),
    "alarm": ("Stretch: FALL DETECTED - no response", "urgent", "rotating_light"),
    "resolved": ("Stretch: fall alert cleared", "default", "white_check_mark"),
    "test": ("Stretch: test notification", "default", "bell"),
}
TITLES_ES = {
    "check": ("Stretch: posible caida", "high", "warning"),
    "alarm": ("Stretch: CAIDA DETECTADA - sin respuesta", "urgent", "rotating_light"),
    "resolved": ("Stretch: alerta de caida resuelta", "default", "white_check_mark"),
    "test": ("Stretch: notificacion de prueba", "default", "bell"),
}  # ntfy titles are HTTP headers: keep them ASCII (the message body can be any UTF-8)


def force_ipv4() -> None:
    """Make requests/urllib3 use IPv4 only. On networks with broken IPv6 (like the robot's) Python otherwise
    tries IPv6 first and the notification can stall until it times out."""
    import socket

    import urllib3.util.connection as uc

    uc.allowed_gai_family = lambda: socket.AF_INET


def siren_wav(seconds: float = 2.4, rate: int = 22050) -> bytes:
    """Two-tone (960/770 Hz) alarm, 0.3 s per tone, square-ish for loudness."""
    t = np.arange(int(seconds * rate)) / rate
    freq = np.where((t // 0.3) % 2 == 0, 960.0, 770.0)
    phase = 2 * np.pi * np.cumsum(freq) / rate
    sig = np.tanh(3 * np.sin(phase)) * 0.9
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    pcm = (sig * fade * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Alarm:
    def __init__(
        self,
        cfg: dict,
        speaker: Speaker | None,
        templates: dict,
        live_url: str = "",
        snapshot_dir: str | Path | None = None,
        spanish: bool = False,
    ):
        self.c = cfg
        self.spanish = spanish
        if cfg.get("force_ipv4", True):
            force_ipv4()
        self.speaker = speaker
        self.templates = templates
        self.live_url = live_url
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot: bytes | None = None
        self.active = False
        self.reason = ""
        self.started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent_notifications: list[dict] = []  # for tests / the report

    # ------------------------------------------------------------------ snapshot
    def save_snapshot(self, frame_bgr: np.ndarray | None, tag: str) -> Path | None:
        if frame_bgr is None:
            return None
        import cv2

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        self.snapshot = buf.tobytes()
        if self.snapshot_dir is None:
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"fall_{time.strftime('%H%M%S')}_{tag}.jpg"
        path.write_bytes(self.snapshot)
        return path

    # ------------------------------------------------------------------ alarm on/off
    def start(self, reason: str) -> None:
        if self.active:
            return
        self.active, self.reason, self.started_at = True, reason, time.time()
        self._stop.clear()
        log.warning("FALL ALARM: %s", reason)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="alarm")
        self._thread.start()
        self.notify_async("alarm", reason)

    def stop(self, why: str = "") -> None:
        if not self.active:
            return
        self.active = False
        self._stop.set()
        log.warning("Fall alarm cleared: %s", why)

    def _loop(self):
        wav_path = None
        if self.c.get("siren", True):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(siren_wav())
                wav_path = f.name
        deadline = time.time() + float(self.c.get("max_duration_s", 600))
        try:
            while not self._stop.is_set() and time.time() < deadline:
                if wav_path:
                    try:
                        _play_wav(wav_path, float(self.c.get("volume", 1.5)))
                    except Exception as e:  # no audio player: keep the spoken/visual alarm going
                        log.error("siren failed: %s", e)
                        wav_path = None
                if self._stop.is_set():
                    break
                if self.speaker is not None:
                    self.speaker.say_blocking(self.templates.get("fall_alarm", "Alert! A fall was detected."))
                self._stop.wait(float(self.c.get("repeat_s", 4)))
        finally:
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------ notifications
    def notify_async(self, kind: str, reason: str) -> None:
        threading.Thread(target=self.notify, args=(kind, reason), daemon=True, name="alarm-notify").start()

    def notify(self, kind: str, reason: str) -> None:
        """kind: "check" (possible fall, asking), "alarm" (no response), "resolved"."""
        titles = TITLES_ES if self.spanish else TITLES_EN
        title, priority, tags = titles.get(kind, titles["resolved"])
        when = time.strftime("%H:%M:%S")
        live = "Vista en vivo" if self.spanish else "Live view"
        text = f"{when} - {reason}." + (f"\n{live}: {self.live_url}" if self.live_url else "")
        topic = str(self.c.get("ntfy_topic") or "").strip()
        if not topic and not str(self.c.get("webhook_url") or "").strip() and kind != "resolved":
            log.warning(
                "Fall %s: no phone notification sent - set alarm.ntfy_topic in config.yaml "
                "(see README: Fall detection)",
                kind,
            )
        if topic:
            url = f"{str(self.c.get('ntfy_server', 'https://ntfy.sh')).rstrip('/')}/{topic}"
            headers = {
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            }
            if self.live_url:
                headers["Click"] = self.live_url
            self._post(url, text.encode("utf-8"), headers, "ntfy")
        hook = str(self.c.get("webhook_url") or "").strip()
        if hook:
            body = json.dumps(
                {"event": f"fall_{kind}", "reason": reason, "time": time.time(), "live_url": self.live_url}
            ).encode()
            self._post(hook, body, {"Content-Type": "application/json"}, "webhook")

    def _post(self, url: str, body: bytes, headers: dict, name: str, attempts: int = 2) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(url, data=body, headers=headers, timeout=8)
                self.sent_notifications.append({"to": name, "status": r.status_code})
                if r.status_code < 400:
                    log.info("%s notification sent: %s", name, headers.get("Title", ""))
                    return True
                log.error("%s notification failed: HTTP %s %s", name, r.status_code, r.text[:200])
            except requests.RequestException as e:
                self.sent_notifications.append({"to": name, "error": str(e)})
                log.error("%s notification failed (attempt %d): %s", name, attempt, e)
            if attempt < attempts:
                time.sleep(2)
        return False

    def status(self) -> dict:
        return {"active": self.active, "reason": self.reason, "since": self.started_at}
