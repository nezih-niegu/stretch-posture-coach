"""Decides *when* Stretch should speak and *what* to say (LLM via Ollama, with rule-based fallback)."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable

from .assessment import Assessment, IssueState
from .llm import OllamaClient, OllamaError, clean_spoken
from .speech import Speaker

log = logging.getLogger(__name__)

FALLBACK = {
    "English": {
        "neck_flexion": "Let's lift your head a little. Bring your chin back so your ears line up over your shoulders.",
        "neck_extension": "Your head is tilted back. Gently lower your chin and keep your neck long.",
        "neck_side_bend": "Your head is tilting to one side. Level it out and relax your neck.",
        "neck_twist": "You've been turning your head for a while. Try facing your work straight on.",
        "trunk_flexion": "You're leaning forward. Sit back, let your chair support you, and lengthen your spine.",
        "trunk_extension": "You're leaning back. Stack your shoulders over your hips and engage your core.",
        "trunk_side_bend": "You're leaning to one side. Center your weight evenly on both hips.",
        "trunk_twist": "Your torso is twisted. Turn your whole body to face what you're working on.",
        "shoulder_tilt": "One shoulder is higher than the other. Let both shoulders drop and relax evenly.",
        "shoulders_raised": "Your shoulders are creeping up. Take a breath and let them drop away from your ears.",
        "arms_raised": "Your arms have been raised for a while. Lower them and bring your work closer if you can.",
        "lumbar_load": "That position loads your lower back. Bend your knees and keep the load close to your body.",
        "break": "You've been at it for a while. How about standing up and stretching for a minute? Trust me, I'm a Stretch.",
        "praise": [
            "Much better, nice work!",
            "That's it, great posture.",
            "Perfect, keep it like that.",
            "Nice correction!",
        ],
        "greet": "Hi{name}! I'm {robot}. I'll keep an eye on your posture and let you know if anything needs adjusting.",
    },
    "Spanish": {
        "neck_flexion": "Levanta un poco la cabeza. Lleva la barbilla hacia atrás para que tus orejas queden sobre tus hombros.",
        "neck_extension": "Tienes la cabeza inclinada hacia atrás. Baja suavemente la barbilla y alarga el cuello.",
        "neck_side_bend": "Tu cabeza se inclina hacia un lado. Nivélala y relaja el cuello.",
        "neck_twist": "Llevas rato con la cabeza girada. Intenta ponerte de frente a tu trabajo.",
        "trunk_flexion": "Te estás inclinando hacia adelante. Recárgate en la silla y alarga la espalda.",
        "trunk_extension": "Te estás inclinando hacia atrás. Alinea los hombros sobre la cadera.",
        "trunk_side_bend": "Te inclinas hacia un lado. Reparte tu peso por igual en ambas caderas.",
        "trunk_twist": "Tienes el torso girado. Gira todo el cuerpo hacia lo que estás haciendo.",
        "shoulder_tilt": "Un hombro está más alto que el otro. Deja caer y relaja ambos hombros.",
        "shoulders_raised": "Tus hombros están subiendo. Respira y bájalos, lejos de las orejas.",
        "arms_raised": "Llevas rato con los brazos levantados. Bájalos y acerca tu trabajo si puedes.",
        "lumbar_load": "Esa postura carga tu espalda baja. Dobla las rodillas y mantén la carga cerca del cuerpo.",
        "break": "Llevas un buen rato así. ¿Qué tal si te levantas y te estiras un minuto?",
        "praise": ["¡Mucho mejor, bien hecho!", "Eso es, excelente postura.", "Perfecto, mantente así."],
        "greet": "¡Hola{name}! Soy {robot}. Voy a cuidar tu postura y te aviso si algo necesita ajuste.",
    },
}

SYSTEM_PROMPT = """You are {robot}, a friendly Hello Robot Stretch 3 robot working as a posture coach. \
You watch the user with your head camera and measure their full-body posture.
Rules for every reply:
- Speak in {language}. One or two short sentences, at most 30 words total. Plain spoken words only.
- Give ONE specific physical correction the user can do right now, phrased positively.
- Be warm and encouraging, never scold. No lists, no emojis, no markdown, no medical diagnosis.
- Do not mention numbers, degrees or scores.{name_rule}"""

EXAMPLE_USER = (
    "Observation: the user is sitting. Main problem: head forward / looking down (head 28 deg forward/down "
    "from the trunk) for 12 seconds. Overall ergonomic risk: medium. What do you say?"
)
EXAMPLE_ASSISTANT = (
    "Let's give your neck a break. Bring your chin back so your ears sit right over your shoulders."
)


def _risk_phrase(a: Assessment) -> str:
    parts = []
    if a.rula:
        parts.append(f"RULA {a.rula['score']} ({a.rula['risk']})")
    if a.reba:
        parts.append(f"REBA {a.reba['score']} ({a.reba['risk']})")
    return ", ".join(parts) or "unknown"


def build_prompt(
    kind: str, a: Assessment, issues: list[IssueState], last_messages: list[str], cfg: dict, now: float
) -> list[dict]:
    c = cfg["coach"]
    name = (c.get("user_name") or "").strip()
    sys = SYSTEM_PROMPT.format(
        robot=c.get("robot_name", "Stretch"),
        language=c.get("language", "English"),
        name_rule=f"\n- The user's name is {name}; use it at most occasionally." if name else "",
    )
    posture = a.metrics.get("posture", "unknown")
    who = f"the user is {posture}" if posture != "unknown" else "the user is in view"
    if kind == "break":
        mins = int(a.static_for_s // 60) if a.static else None
        obs = (
            f"Observation: {who} and has been in front of you for a long time"
            + (f", holding the same posture for {mins} minutes" if mins else "")
            + ". Suggest a short movement break (stand up, roll the shoulders, walk). What do you say?"
        )
    else:
        main = issues[0]
        obs = (
            f"Observation: {who}. Main problem: {main.rule.label} ({main.detail}) for "
            f"{main.duration(now):.0f} seconds."
        )
        if len(issues) > 1:
            obs += " Also: " + "; ".join(i.rule.label for i in issues[1:3]) + "."
        load = a.metrics.get("l5s1_compression_n")
        if load:
            obs += f" Estimated low-back compression {load:.0f} N."
        obs += f" Overall ergonomic risk: {_risk_phrase(a)}."
        obs += " Focus only on the main problem. What do you say?"
    if last_messages:
        obs += f' Do not repeat your previous message: "{last_messages[-1]}".'
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT},
        {"role": "user", "content": obs},
    ]


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
        self.templates = FALLBACK.get(lang, FALLBACK["English"])
        self.allow_cjk = lang.lower() in ("chinese", "japanese", "korean", "mandarin")
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

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or self.speaker.busy

    def update(self, a: Assessment, now: float) -> None:
        self._t = now
        self._update_presence(a, now)
        if self.busy:
            self.last_spoken = now  # measure the quiet interval from the END of the last message
            return
        if not self.enabled or not a.present:
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
