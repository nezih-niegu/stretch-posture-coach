"""Text-to-speech through the Stretch speaker: Piper (neural, offline) with espeak-ng fallback."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

log = logging.getLogger(__name__)


def audio_env() -> dict:
    """Over SSH, XDG_RUNTIME_DIR is often unset, so PulseAudio/PipeWire can't be found. Fix that."""
    env = dict(os.environ)
    if "XDG_RUNTIME_DIR" not in env:
        candidate = f"/run/user/{os.getuid()}"
        if os.path.isdir(candidate):
            env["XDG_RUNTIME_DIR"] = candidate
    return env


def _play_wav(path: str, volume: float = 1.0) -> None:
    env = audio_env()
    if shutil.which("paplay"):
        cmd = ["paplay", f"--volume={int(max(0.0, min(volume, 1.5)) * 65536)}", path]
    elif shutil.which("pw-play"):
        cmd = ["pw-play", f"--volume={volume}", path]
    elif shutil.which("aplay"):
        cmd = ["aplay", "-q", path]
    else:
        raise RuntimeError("No audio player found (install pulseaudio-utils or alsa-utils)")
    subprocess.run(cmd, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _Engine:
    name = "base"

    def say(self, text: str) -> None:
        raise NotImplementedError


class PrintEngine(_Engine):
    name = "print"

    def say(self, text: str) -> None:
        print(f"[{self.name}] {text}", flush=True)


class EspeakEngine(_Engine):
    name = "espeak"

    def __init__(self, voice: str = "en-us", rate_wpm: int = 165, volume: float = 1.0):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise RuntimeError("espeak-ng not installed (sudo apt install espeak-ng)")
        self.voice, self.rate, self.amp = voice, rate_wpm, int(max(0, min(volume, 2.0)) * 100)

    def say(self, text: str) -> None:
        subprocess.run(
            [self.exe, "-v", self.voice, "-s", str(self.rate), "-a", str(self.amp), text],
            env=audio_env(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class PiperEngine(_Engine):
    name = "piper"

    def __init__(
        self, voice: str, voices_dir: str | Path = "models/piper", rate_wpm: int = 165, volume: float = 1.0
    ):
        from piper import PiperVoice

        model = Path(voices_dir) / f"{voice}.onnx"
        if not model.exists():
            raise RuntimeError(f"Piper voice not found: {model} (run `posture-coach download-models`)")
        self.voice = PiperVoice.load(str(model))
        self.volume = volume
        # piper's length_scale > 1 is slower; map words-per-minute around a 165 wpm baseline
        self.length_scale = max(0.6, min(1.6, 165.0 / max(rate_wpm, 60)))

    def say(self, text: str) -> None:
        syn_config = None
        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=self.length_scale)
        except Exception:
            pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as wf:
                if hasattr(self.voice, "synthesize_wav"):
                    self.voice.synthesize_wav(text, wf, syn_config=syn_config)
                else:  # piper-tts < 1.3
                    self.voice.synthesize(text, wf)
            _play_wav(path, self.volume)
        finally:
            Path(path).unlink(missing_ok=True)


def make_engine(cfg: dict) -> _Engine:
    engine = str(cfg.get("engine", "auto")).lower()
    rate, vol = int(cfg.get("rate_wpm", 165)), float(cfg.get("volume", 1.0))
    if engine == "print":
        return PrintEngine()
    errors = []
    if engine in ("auto", "piper"):
        try:
            return PiperEngine(
                cfg.get("piper_voice", "en_US-lessac-medium"),
                cfg.get("voices_dir", "models/piper"),
                rate,
                vol,
            )
        except Exception as e:
            errors.append(f"piper: {e}")
            if engine == "piper":
                raise
    if engine in ("auto", "espeak"):
        try:
            return EspeakEngine(cfg.get("espeak_voice", "en-us"), rate, vol)
        except Exception as e:
            errors.append(f"espeak: {e}")
            if engine == "espeak":
                raise
    log.warning("No TTS engine available (%s); printing messages instead", "; ".join(errors))
    return PrintEngine()


class Speaker:
    """Non-blocking speech queue. Only the newest pending message is kept."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.engine: _Engine = make_engine(cfg) if self.enabled else PrintEngine()
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def say(self, text: str) -> None:
        if not text:
            return
        try:
            self._q.get_nowait()  # drop stale message
        except queue.Empty:
            pass
        self._q.put_nowait(text)

    def say_blocking(self, text: str) -> None:
        self._busy.set()
        try:
            self.engine.say(text)
        finally:
            self._busy.clear()

    def _run(self):
        while True:
            text = self._q.get()
            if text is None:
                return
            self._busy.set()
            try:
                log.info("speaking: %s", text)
                self.engine.say(text)
            except Exception as e:  # never kill the thread
                log.error("TTS failed: %s", e)
            finally:
                self._busy.clear()

    def close(self, wait_s: float = 8.0):
        import time

        deadline = time.monotonic() + wait_s
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.1)
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
