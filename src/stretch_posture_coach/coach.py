"""Decides *when* Stretch should speak and *what* to say (LLM via Ollama, with rule-based fallback)."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable

from . import personas
from .assessment import Assessment, IssueState
from .llm import OllamaClient, OllamaError, clean_spoken
from .speech import Speaker

log = logging.getLogger(__name__)

RISK_ES = {
    "acceptable": "aceptable",
    "investigate": "medio",
    "change soon": "alto",
    "change now": "muy alto",
    "negligible": "bajo",
    "low": "bajo",
    "medium": "medio",
    "high": "alto",
    "very high": "muy alto",
}


def _risk_phrase(a: Assessment, spanish: bool = False) -> str:
    risk = (a.reba or a.rula or {}).get("risk")
    if spanish:
        return RISK_ES.get(risk, "desconocido") if risk else "desconocido"
    parts = []
    if a.rula:
        parts.append(f"RULA {a.rula['score']} ({a.rula['risk']})")
    if a.reba:
        parts.append(f"REBA {a.reba['score']} ({a.reba['risk']})")
    return ", ".join(parts) or "unknown"


def _observation_en(kind: str, a: Assessment, issues: list[IssueState], now: float) -> str:
    posture = a.metrics.get("posture", "unknown")
    who = f"the user is {posture}" if posture != "unknown" else "the user is in view"
    if kind == "break":
        mins = int(a.static_for_s // 60) if a.static else None
        return (
            f"Observation: {who} and has been in front of you for a long time"
            + (f", holding the same posture for {mins} minutes" if mins else "")
            + ". Suggest a short movement break (stand up, roll the shoulders, walk). What do you say?"
        )
    main = issues[0]
    obs = f"Observation: {who}. Main problem: {main.rule.label} ({main.detail}) for {main.duration(now):.0f} seconds."
    if len(issues) > 1:
        obs += " Also: " + "; ".join(i.rule.label for i in issues[1:3]) + "."
    load = a.metrics.get("l5s1_compression_n")
    if load:
        obs += f" Estimated low-back compression {load:.0f} N."
    return (
        obs + f" Overall ergonomic risk: {_risk_phrase(a)}. Focus only on the main problem. What do you say?"
    )


def _observation_es(kind: str, a: Assessment, issues: list[IssueState], now: float) -> str:
    posture = personas.POSTURE_ES.get(a.metrics.get("posture", "unknown"), "frente a ti")
    who = f"la persona está {posture}"
    if kind == "break":
        mins = int(a.static_for_s // 60) if a.static else None
        return (
            f"Observación: {who} y lleva mucho tiempo frente a ti"
            + (f", en la misma postura por {mins} minutos" if mins else "")
            + ". Sugiere una pausa corta para moverse (pararse, mover los hombros, caminar). ¿Qué le dices?"
        )
    main = issues[0]
    label = personas.LABELS_ES.get(main.rule.id, main.rule.label)
    obs = f"Observación: {who}. Problema principal: {label} durante {main.duration(now):.0f} segundos."
    if len(issues) > 1:
        obs += (
            " También: "
            + "; ".join(personas.LABELS_ES.get(i.rule.id, i.rule.label) for i in issues[1:3])
            + "."
        )
    return (
        obs
        + f" Riesgo ergonómico: {_risk_phrase(a, True)}. Enfócate solo en el problema principal. ¿Qué le dices?"
    )


def build_prompt(
    kind: str, a: Assessment, issues: list[IssueState], last_messages: list[str], cfg: dict, now: float
) -> list[dict]:
    c = cfg["coach"]
    p = personas.resolve(c)
    obs = (_observation_es if p.spanish else _observation_en)(kind, a, issues, now)
    if last_messages:
        obs += (
            f' No repitas tu mensaje anterior: "{last_messages[-1]}".'
            if p.spanish
            else f' Do not repeat your previous message: "{last_messages[-1]}".'
        )
    msgs = [{"role": "system", "content": personas.system_prompt(p, c)}]
    for user, assistant in p.examples:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    msgs.append({"role": "user", "content": obs})
    return msgs


class Coach:
    def __init__(
        self,
        cfg: dict,
        speaker: Speaker,
        llm: OllamaClient | None,
        on_event: Callable[[dict], None] | None = None,
        issues=(),
    ):
        self.cfg = cfg
        self.c = cfg["coach"]
        self.enabled = bool(self.c.get("enabled", True))
        self.speaker = speaker
        self.llm = llm
        self.on_event = on_event or (lambda e: None)
        lang = self.c.get("language", "English")
        self.persona = personas.resolve(self.c)
        self.templates = self.persona.fallback
        self.allow_cjk = lang.lower() in ("chinese", "japanese", "korean", "mandarin")
        self._announced: dict[str, float] = {}
        self.last_spoken = -1e9
        self.last_break = None
        self.present_since: float | None = None
        self.absent_since: float | None = None
        self.history: deque[str] = deque(maxlen=4)
        self.last_message = ""
        self._busy = threading.Event()
        self._praised: set[tuple[str, float]] = set()
        self._issues = list(issues)  # all IssueState objects owned by the Assessor
        self._t = 0.0  # latest timestamp seen (same clock as the frame timestamps)

    # ------------------------------------------------------------------ public
    def greet(self):
        if not (self.enabled and self.c.get("greet", True)):
            return
        name = (self.c.get("user_name") or "").strip()
        text = self.templates["greet"].format(
            name=f", {name}" if name else "", robot=self.c.get("robot_name", "Stretch")
        )
        self._speak(text, {"kind": "greet", "source": "template"})

    def announce(self, key: str, now: float, min_gap_s: float = 45.0) -> bool:
        """Speak a short fixed line (e.g. follow-mode notices) unless said recently or already talking."""
        last = self._announced.get(key)
        if (last is not None and now - last < min_gap_s) or self.speaker.busy or key not in self.templates:
            return False
        self._announced[key] = now
        self._speak(self.templates[key], {"kind": "announce", "key": key, "source": "template"})
        return True

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or self.speaker.busy

    def mute(self, now: float, seconds: float) -> None:
        """No posture tips / praise / break reminders until ``now + seconds`` (e.g. someone is on the floor)."""
        self.mute_until = max(getattr(self, "mute_until", 0.0), now + seconds)

    def muted(self, now: float | None = None) -> bool:
        return (self._t if now is None else now) < getattr(self, "mute_until", -1e9)

    def update(self, a: Assessment, now: float) -> None:
        self._t = now
        self._update_presence(a, now)
        if self.busy:
            self.last_spoken = now  # measure the quiet interval from the END of the last message
            return
        if not self.enabled or not a.present or self.muted(now):
            return
        since_last = now - self.last_spoken

        # 1) praise a fix that happened shortly after we coached it
        if self.c.get("praise", True) and since_last > 4:
            for st in self._issues:
                if (
                    st.last_coached
                    and st.cleared_at
                    and st.cleared_at > st.last_coached
                    and st.cleared_at - st.last_coached < float(self.c.get("praise_window_s", 20))
                    and (st.rule.id, st.last_coached) not in self._praised
                    and not st.active
                ):
                    self._praised.add((st.rule.id, st.last_coached))
                    self._speak(
                        random.choice(self.templates["praise"]),
                        {"kind": "praise", "issue": st.rule.id, "source": "template"},
                    )
                    return

        if since_last < float(self.c.get("min_interval_s", 25)):
            return

        # 2) coach the most important sustained issue not coached recently
        repeat = float(self.c.get("repeat_interval_s", 75))
        candidates = [s for s in a.active if s.last_coached is None or now - s.last_coached > repeat]
        if candidates:
            main = candidates[0]
            main.last_coached = now
            others = [s for s in a.active if s is not main]
            self._submit("issue", a, [main, *others], now)
            return

        # 3) movement break
        brk = float(self.c.get("break_reminder_min", 30)) * 60
        if brk > 0 and self.present_since is not None and now - self.present_since > brk:
            if self.last_break is None or now - self.last_break > brk:
                self.last_break = now
                self._submit("break", a, [], now)

    # ----------------------------------------------------------------- internals
    def _update_presence(self, a: Assessment, now: float):
        if a.present:
            self.absent_since = None
            if self.present_since is None:
                self.present_since = now
        else:
            if self.absent_since is None:
                self.absent_since = now
            if now - self.absent_since > 60:  # stepped away for a minute = a break
                self.present_since = None

    def _submit(self, kind: str, a: Assessment, issues: list[IssueState], now: float):
        self._busy.set()
        self.last_spoken = now  # reserve the slot now so we don't double-submit
        threading.Thread(target=self._work, args=(kind, a, issues, now), daemon=True, name="coach").start()

    def _work(self, kind: str, a: Assessment, issues: list[IssueState], now: float):
        text, source, latency = "", "template", 0.0
        try:
            if self.llm is not None:
                t0 = time.monotonic()
                try:
                    raw = self.llm.chat(build_prompt(kind, a, issues, list(self.history), self.cfg, now))
                    text = clean_spoken(raw, allow_cjk=self.allow_cjk)
                    source = "llm"
                except OllamaError as e:
                    log.warning("LLM failed (%s); using fallback", e)
                latency = time.monotonic() - t0
            if not text:
                source = "template"
                key = "break" if kind == "break" else issues[0].rule.id
                text = self.templates.get(key, self.templates["break"])
            event = {"kind": kind, "source": source, "llm_latency_s": round(latency, 2)}
            if issues:
                event.update(
                    {
                        "issue": issues[0].rule.id,
                        "detail": issues[0].detail,
                        "others": [i.rule.id for i in issues[1:]],
                    }
                )
            if a.rula:
                event["rula"] = a.rula["score"]
            if a.reba:
                event["reba"] = a.reba["score"]
            self._speak(text, event)
        finally:
            self._busy.clear()

    def _speak(self, text: str, event: dict):
        if self.muted() and event.get("kind") in ("issue", "praise", "break"):
            # e.g. an LLM tip that was being generated when the person fell: drop it
            log.info("coach: dropped %s message while muted: %s", event.get("kind"), text)
            return
        self.last_spoken = max(self.last_spoken, self._t)
        self.history.append(text)
        self.last_message = text
        self.speaker.say(text)
        event = {**event, "text": text, "wall_time": time.time()}
        log.info("coach [%s/%s]: %s", event.get("kind"), event.get("source"), text)
        try:
            self.on_event(event)
        except Exception as e:
            log.error("event callback failed: %s", e)
