"""Talk through Stretch from the headset: Quest microphone -> speech-to-text -> LLM -> the robot's voice.

The expert holds the LEFT TRIGGER (push-to-talk) in VR (or the "Hold to talk" button on the page). The
browser streams 16 kHz mono PCM over the VR WebSocket; this module

  1. transcribes it offline with faster-whisper (CPU, int8) - partial transcripts every ~1 s while talking,
     shown live in the headset, and the final transcript on release;
  2. turns the expert's words into what Stretch says to the person, with the local LLM (Ollama) in Stretch's
     persona and language (``voice.mode: relay``) - or speaks them as said (``verbatim``);
  3. speaks it with the robot's own TTS voice (Piper) on the robot speaker, and logs it to the session
     transcript (kind "expert").

Only the PIN-authenticated expert can use it. Nothing leaves the robot.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import personas
from .llm import clean_spoken

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000


class WhisperTranscriber:
    """faster-whisper on the CPU. The model is downloaded once to <models>/whisper (``download-models``)."""

    def __init__(self, model: str = "base", models_dir: str | Path = "models", threads: int = 4):
        self.model_name, self.models_dir, self.threads = model, Path(models_dir) / "whisper", threads
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                t0 = time.monotonic()
                self._model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.threads,
                    download_root=str(self.models_dir),
                )
                log.info(
                    "speech-to-text: whisper %s loaded in %.1f s", self.model_name, time.monotonic() - t0
                )
        return self._model

    def __call__(self, audio: np.ndarray, language: str | None) -> str:
        m = self.load()
        segs, _ = m.transcribe(
            audio, language=language, beam_size=1, vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segs).strip()


RELAY_EN = """You are the voice of {robot}, a robot posture coach. A remote specialist is talking to the person
through you. Turn what the specialist said into what you say to the person, in {language}:
- Keep the meaning exactly. Speak directly to the person ("you"). Do not add advice or facts of your own.
- If the specialist asks the person something, ask it. If they give an instruction, give it kindly.
- If they ask you to report the person's posture, use the measurements below.
- One or two short sentences, plain spoken words, no quotes, no emoji.
Current measurements: {context}"""

RELAY_ES = """Eres la voz de {robot}, un robot que ayuda con la postura. Un especialista habla con la persona
a través de ti. Convierte lo que dijo el especialista en lo que tú le dices a la persona, en español:
- Conserva el significado exacto. Háblale directamente a la persona ("tú"). No agregues consejos ni datos propios.
- Si el especialista pregunta algo, pregúntalo. Si da una indicación, dila con amabilidad.
- Si pide reportar la postura, usa las mediciones de abajo.
- Una o dos frases cortas, palabras habladas sencillas, sin comillas ni emojis.{style}
Mediciones actuales: {context}"""


# Worked examples for the LLM (small models otherwise answer the specialist instead of relaying)
RELAY_SHOTS_EN = [
    ("Ask him how he has been feeling lately.", "How have you been feeling lately?"),
    ("Tell her to relax her shoulders.", "Please relax your shoulders."),
    ("Can you tell the patient to sit up straight?", "Please sit up straight."),
    ("Hello, good afternoon.", "Hello, good afternoon."),
    ("Tell him his neck looks much better now.", "Your neck looks much better now."),
]
RELAY_SHOTS_ES = [
    ("Pregúntale cómo se ha sentido últimamente.", "¿Cómo te has sentido últimamente?"),
    ("Dile que relaje los hombros.", "Por favor, relaja los hombros."),
    ("¿Le puedes decir que se siente derecho?", "Por favor, siéntate derecho."),
    ("Hola, buenas tardes.", "Hola, buenas tardes."),
]
# replies that talk back to the specialist / the robot instead of relaying -> rejected
_BAD = re.compile(
    r"\b(specialist|especialista|assist you|help you today|i'll make sure|i will make sure|i can help|"
    r"yes,? i can|thanks for the instruction|claro que sí,? puedo|te ayudo)\b|^\s*(hello,? )?stretch\b",
    re.I,
)
_TELL_EN = re.compile(
    r"^(?:(?:hello|hi|hey),?\s*)?(?:stretch[,.]?\s*)?(?:(?:can|could|would) you\s+|please\s+)?"
    r"(tell|ask|asked)\s+(?:the\s+)?(?:patient|person|him|her|them|user)\s+(?:to\s+)?(.+?)[.?!]*$",
    re.I,
)
_TELL_ES = re.compile(
    r"^(?:(?:hola|oye),?\s*)?(?:stretch[,.]?\s*)?¿?\s*(?:(?:le\s+)?puedes\s+)?"
    r"(dile|pídele|decirle|pedirle|pregúntale|preguntarle)\s+(?:al paciente\s+|a la paciente\s+)?(?:que\s+)?"
    r"(.+?)[.?!]*$",
    re.I,
)
_ASK_WORDS = (
    "how",
    "what",
    "if",
    "whether",
    "when",
    "where",
    "why",
    "do",
    "does",
    "are",
    "is",
    "has",
    "have",
    "can",
)
_PRON = [
    (r"\bhas he\b", "have you"),
    (r"\bhas she\b", "have you"),
    (r"\bis he\b", "are you"),
    (r"\bis she\b", "are you"),
    (r"\bdoes he\b", "do you"),
    (r"\bdoes she\b", "do you"),
    (r"\bhe has\b", "you have"),
    (r"\bshe has\b", "you have"),
    (r"\bhe is\b", "you are"),
    (r"\bshe is\b", "you are"),
    (r"\bhe feels\b", "you feel"),
    (r"\bshe feels\b", "you feel"),
    (r"\bhe needs\b", "you need"),
    (r"\bshe needs\b", "you need"),
    (r"\bhe\b", "you"),
    (r"\bshe\b", "you"),
    (r"\bhis\b", "your"),
    (r"\bher\b", "your"),
    (r"\bhim\b", "you"),
    (r"\bhimself\b", "yourself"),
    (r"\bherself\b", "yourself"),
    (r"\bthey are\b", "you are"),
    (r"\btheir\b", "your"),
    (r"\bthem\b", "you"),
    (r"\bI need\b", "the specialist needs"),
    (r"\bI want\b", "the specialist wants"),
    (r"\bI\b", "the specialist"),
    (r"\bme\b", "the specialist"),
    (r"\bmy\b", "the specialist's"),
]


def rule_relay(heard: str, spanish: bool) -> str | None:
    """Deterministic rewrite of clear relay commands ("tell the patient to X", "ask him how he is").
    Returns None when the sentence isn't one of those (then the LLM handles it)."""
    t = heard.strip().strip('"')
    if spanish:
        m = _TELL_ES.match(t)
        if not m:
            return None
        verb, rest = m.group(1).lower(), m.group(2).strip()
        if verb.startswith(
            ("pregunt", "pregúnt")
        ):  # "pregúntale cómo se siente" -> "¿Cómo se siente?" (usted)
            return f"¿{rest[0].upper()}{rest[1:]}?"
        return None  # "dile que se siente" needs conjugation (-> "siéntate"): the LLM does that
    m = _TELL_EN.match(t)
    if not m:
        return None
    verb, rest = m.group(1).lower(), m.group(2).strip()
    for a, b in _PRON:
        rest = re.sub(a, b, rest, flags=re.I)
    first = rest.split(" ", 1)[0].lower()
    if verb.startswith("ask") and first in _ASK_WORDS:
        words = rest.split(" ")
        if words[0].lower() in ("if", "whether"):
            words = words[1:]
            q = " ".join(words)  # "if you feel pain" -> "do you feel pain"
            q = re.sub(r"^you (are|have|can)\b", lambda mm: f"{mm.group(1)} you", q, flags=re.I)
            q = re.sub(r"^you (feel|need|want)\b", r"do you \1", q, flags=re.I)
        else:
            q = " ".join(words)  # "how you have been" -> "how have you been"
            q = re.sub(
                r"^(how|what|when|where|why) you (have|are|were|do|did)\b", r"\1 \2 you", q, flags=re.I
            )
            q = re.sub(r"^(how|what|when|where|why) you (feel|felt|need)\b", r"\1 do you \2", q, flags=re.I)
        return q[0].upper() + q[1:] + "?"
    return f"Please {rest[0].lower()}{rest[1:]}."


def is_bad_relay(text: str) -> bool:
    return bool(_BAD.search(text))


def posture_context(a) -> str:
    if a is None or not getattr(a, "present", False):
        return "the person is not in view"
    m = a.metrics
    parts = [f"posture {m.get('posture', '?')}"]
    for k, lab in (("neck_flexion", "neck"), ("trunk_flexion", "trunk")):
        if m.get(k) is not None:
            parts.append(f"{lab} {m[k]:.0f} deg")
    if a.reba:
        parts.append(f"REBA {a.reba.get('score')} ({a.reba.get('risk')})")
    return ", ".join(parts)


class VoiceRelay:
    """Push-to-talk session state machine: begin() -> feed(pcm16 bytes)* -> end()."""

    def __init__(
        self, cfg: dict, speak, llm=None, transcriber=None, notify=None, context=None, log_event=None
    ):
        c = cfg.get("voice", {})
        self.c = c
        self.coach_cfg = cfg.get("coach", {})
        self.persona = personas.resolve(self.coach_cfg)
        lang = str(c.get("language", "auto")).lower()
        self.language = ("es" if self.persona.spanish else "en") if lang == "auto" else lang
        self.mode = str(c.get("mode", "relay"))
        self.partial_s = float(c.get("partial_every_s", 1.0))
        self.max_s = float(c.get("max_utterance_s", 30))
        self.speak = speak  # (text) -> None, the robot's TTS
        self.llm = llm
        self.stt = transcriber or WhisperTranscriber(
            str(c.get("stt_model", "base")),
            cfg.get("pose", {}).get("models_dir", "models"),
            int(c.get("threads", 4)),
        )
        self.notify = notify or (lambda msg: None)  # (dict) -> None, to the headset
        self.context = context or (lambda: None)
        self.log_event = log_event or (lambda ev: None)
        self._buf = bytearray()
        self._talking = False
        self._lock = threading.Lock()
        self._busy = threading.Lock()  # one transcription at a time
        self._t0 = 0.0
        self._partial_thread: threading.Thread | None = None
        self.last: dict = {}

    def preload(self) -> None:
        """Load the STT model in the background so the first utterance is quick."""

        def _load():
            try:
                self.stt.load() if hasattr(self.stt, "load") else None
            except Exception as e:
                log.warning("speech-to-text unavailable (%s): run `posture-coach download-models`", e)

        threading.Thread(target=_load, daemon=True, name="stt-load").start()

    # ------------------------------------------------------------------ audio in
    def begin(self) -> None:
        with self._lock:
            self._buf = bytearray()
            self._talking = True
            self._t0 = time.monotonic()
        self.notify({"type": "voice", "state": "listening"})
        self._partial_thread = threading.Thread(target=self._partials, daemon=True, name="stt-partial")
        self._partial_thread.start()

    def feed(self, pcm16: bytes) -> None:
        with self._lock:
            if not self._talking:
                return
            if len(self._buf) < self.max_s * SAMPLE_RATE * 2:
                self._buf += pcm16

    def end(self) -> None:
        with self._lock:
            if not self._talking:
                return
            self._talking = False
            audio = bytes(self._buf)
        threading.Thread(target=self._final, args=(audio,), daemon=True, name="stt-final").start()

    def _audio(self, raw: bytes) -> np.ndarray:
        return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768.0

    def _partials(self):
        while True:
            time.sleep(self.partial_s)
            with self._lock:
                if not self._talking:
                    return
                raw = bytes(self._buf)
            if len(raw) < SAMPLE_RATE * 2 * 0.6 or not self._busy.acquire(blocking=False):
                continue
            try:
                text = self.stt(self._audio(raw), self.language)
                if text:
                    self.notify({"type": "voice", "state": "listening", "partial": text})
            except Exception as e:
                log.debug("partial STT: %s", e)
            finally:
                self._busy.release()

    # ------------------------------------------------------------------ text out
    def _final(self, raw: bytes):
        dur = len(raw) / 2 / SAMPLE_RATE
        if dur < 0.4:
            self.notify(
                {"type": "voice", "state": "idle", "error": "too short - hold the trigger while you talk"}
            )
            return
        self.notify({"type": "voice", "state": "transcribing"})
        t0 = time.monotonic()
        with self._busy:
            try:
                heard = self.stt(self._audio(raw), self.language)
            except Exception as e:
                log.error("speech-to-text failed: %s", e)
                self.notify({"type": "voice", "state": "idle", "error": f"speech-to-text failed: {e}"})
                return
        t_stt = time.monotonic() - t0
        if not heard:
            self.notify({"type": "voice", "state": "idle", "error": "didn't catch that"})
            return
        self.notify({"type": "voice", "state": "thinking", "heard": heard})
        said, source = self.compose(heard)
        self.speak(said)
        self.last = {
            "heard": heard,
            "said": said,
            "source": source,
            "stt_s": round(t_stt, 2),
            "audio_s": round(dur, 1),
            "total_s": round(time.monotonic() - t0, 2),
        }
        log.info(
            "expert voice: heard %r -> saying %r (%s, %.1f s)", heard, said, source, self.last["total_s"]
        )
        self.notify({"type": "voice", "state": "spoken", **self.last})
        self.log_event(
            {"kind": "expert", "source": source, "heard": heard, "text": said, "wall_time": time.time()}
        )

    def compose(self, heard: str) -> tuple[str, str]:
        """What Stretch says for the expert's words: (text, "llm" | "verbatim")."""
        if self.mode == "verbatim":
            return clean_spoken(heard, 400), "verbatim"
        ruled = rule_relay(heard, self.persona.spanish)
        if ruled:  # "tell the patient to ..." - no need for the LLM (and no chance for it to talk back)
            return clean_spoken(ruled, 400), "rule"
        if self.llm is None:
            return clean_spoken(heard, 400), "verbatim"
        robot = self.coach_cfg.get("robot_name", "Stretch")
        ctx = posture_context(self.context())
        if self.persona.spanish:
            style = ""
            if self.persona.key == "norteno":
                style = "\n- Habla como norteño de la sierra, con cariño y respeto, sin exagerar."
            sys_prompt = RELAY_ES.format(robot=robot, style=style, context=ctx)
        else:
            sys_prompt = RELAY_EN.format(
                robot=robot, language=self.coach_cfg.get("language", "English"), context=ctx
            )
        shots = RELAY_SHOTS_ES if self.persona.spanish else RELAY_SHOTS_EN
        msgs = [{"role": "system", "content": sys_prompt}]
        for u, a in shots:
            msgs += [{"role": "user", "content": f"Specialist: {u}"}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": f"Specialist: {heard}"})
        try:
            out = self.llm.chat(msgs, max_tokens=int(self.c.get("max_tokens", 90)))
            out = re.sub(r"^\s*(stretch|robot|you|tú|yo)\s*:\s*", "", out.strip(), flags=re.I)
            out = clean_spoken(out, 400)
            if out and not is_bad_relay(out) and len(out) <= 2.5 * len(heard) + 40:
                return out, "llm"
            if out:
                log.info(
                    "voice relay: rejected LLM reply %r (talks back / adds content): speaking the words", out
                )
        except Exception as e:
            log.warning("voice relay LLM failed (%s): speaking the transcript", e)
        return clean_spoken(heard, 400), "verbatim"
