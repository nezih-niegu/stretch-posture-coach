"""Who is in charge of the robot: a small finite-state machine.

States (one at a time) and what each one owns:

  state          head                      base              arm      posture coaching
  ─────────────  ────────────────────────  ────────────────  ───────  ───────────────────────────────
  SURVEY         360° sweep (survey.py)    survey spin       -        background (assess + log, quiet)
  AUTONOMOUS     track the person / VR     follow mode       -        speaks
  TELEOP         expert's headset (VR)     expert joysticks  expert   background (assess + log, quiet)
  FALL_RESPONSE  track the person          stopped           -        off (only the fall check/alarm)
  SAFE_STOP      hold                      stopped           -        background

Events and transitions (anything not listed is ignored and logged at debug level):

  NEW_SESSION      AUTONOMOUS -> SURVEY;   in TELEOP / FALL_RESPONSE / SAFE_STOP the survey is queued
  SURVEY_DONE      SURVEY -> AUTONOMOUS    (SURVEY_FAILED too: the session still runs, the report says so)
  TELEOP_ON        SURVEY / AUTONOMOUS / FALL_RESPONSE -> TELEOP   (the expert overrides the routine;
                   an interrupted survey is queued again)
  TELEOP_OFF       TELEOP -> SURVEY if one is queued, else AUTONOMOUS
  FALL             SURVEY / AUTONOMOUS -> FALL_RESPONSE;  in TELEOP the expert stays in charge (alarm still runs)
  FALL_CLEARED     FALL_RESPONSE -> SURVEY if queued, else AUTONOMOUS
  RUNSTOP          any -> SAFE_STOP (remembers where it was)
  RUNSTOP_CLEARED  SAFE_STOP -> the remembered state (a running survey restarts)

To add a capability (e.g. "fetch an object"): add a state, its Policy, and its rows in TRANSITIONS.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum

log = logging.getLogger(__name__)


class Mode(str, Enum):
    SURVEY = "survey"
    AUTONOMOUS = "autonomous"
    TELEOP = "teleop"
    FALL_RESPONSE = "fall_response"
    SAFE_STOP = "safe_stop"


class Ev(str, Enum):
    NEW_SESSION = "new_session"
    SURVEY_DONE = "survey_done"
    SURVEY_FAILED = "survey_failed"
    TELEOP_ON = "teleop_on"
    TELEOP_OFF = "teleop_off"
    FALL = "fall"
    FALL_CLEARED = "fall_cleared"
    RUNSTOP = "runstop"
    RUNSTOP_CLEARED = "runstop_cleared"


@dataclass(frozen=True)
class Policy:
    head: str  # "track" (person, or the VR viewer if look control is on) | "vr" | "survey" | "hold"
    base: str  # "follow" | "teleop" | "survey" | "none"
    arm: str  # "teleop" | "none"
    coaching: str  # "speak" | "background" | "off"
    label: str


POLICIES: dict[Mode, Policy] = {
    Mode.SURVEY: Policy("survey", "survey", "none", "background", "Mapping the room (360°)"),
    Mode.AUTONOMOUS: Policy("track", "follow", "none", "speak", "Coaching"),
    Mode.TELEOP: Policy("vr", "teleop", "teleop", "background", "Expert control"),
    Mode.FALL_RESPONSE: Policy("track", "none", "none", "off", "Fall response"),
    Mode.SAFE_STOP: Policy("hold", "none", "none", "background", "Runstop"),
}

_ANY = "*"
# (state, event) -> next state;  "pending" = queue a survey and stay;  "resume" = back to the remembered state
TRANSITIONS: dict[tuple, object] = {
    (Mode.AUTONOMOUS, Ev.NEW_SESSION): Mode.SURVEY,
    (Mode.SURVEY, Ev.NEW_SESSION): Mode.SURVEY,  # restart the sweep for the new session
    (Mode.TELEOP, Ev.NEW_SESSION): "pending",
    (Mode.FALL_RESPONSE, Ev.NEW_SESSION): "pending",
    (Mode.SAFE_STOP, Ev.NEW_SESSION): "pending",
    (Mode.SURVEY, Ev.SURVEY_DONE): Mode.AUTONOMOUS,
    (Mode.SURVEY, Ev.SURVEY_FAILED): Mode.AUTONOMOUS,
    (Mode.SURVEY, Ev.TELEOP_ON): Mode.TELEOP,
    (Mode.AUTONOMOUS, Ev.TELEOP_ON): Mode.TELEOP,
    (Mode.FALL_RESPONSE, Ev.TELEOP_ON): Mode.TELEOP,
    (Mode.TELEOP, Ev.TELEOP_OFF): "after_override",
    (Mode.SURVEY, Ev.FALL): Mode.FALL_RESPONSE,
    (Mode.AUTONOMOUS, Ev.FALL): Mode.FALL_RESPONSE,
    (Mode.FALL_RESPONSE, Ev.FALL_CLEARED): "after_override",
    (_ANY, Ev.RUNSTOP): Mode.SAFE_STOP,
    (Mode.SAFE_STOP, Ev.RUNSTOP_CLEARED): "resume",
}


class ModeMachine:
    """Thread-safe. ``dispatch`` returns True if the state changed; listeners get (old, new, event, info)."""

    def __init__(self, initial: Mode = Mode.AUTONOMOUS, clock=time.monotonic):
        self.mode = initial
        self.since = clock()
        self.clock = clock
        self.survey_pending = False
        self._resume: Mode | None = None
        self._lock = threading.RLock()
        self._listeners: list = []
        self.history: list[dict] = []

    @property
    def policy(self) -> Policy:
        return POLICIES[self.mode]

    def on_change(self, fn) -> None:
        self._listeners.append(fn)

    def is_(self, *modes: Mode) -> bool:
        return self.mode in modes

    def dispatch(self, event: Ev, **info) -> bool:
        with self._lock:
            old = self.mode
            nxt = TRANSITIONS.get((old, event), TRANSITIONS.get((_ANY, event)))
            if nxt is None:
                log.debug("mode: %s ignores %s", old.value, event.value)
                return False
            if old == Mode.SURVEY and event not in (Ev.SURVEY_DONE, Ev.SURVEY_FAILED, Ev.NEW_SESSION):
                self.survey_pending = True  # interrupted (expert, fall, runstop): redo the sweep afterwards
            if nxt == "pending":
                self.survey_pending = True
                log.info("mode: survey queued (in %s)", old.value)
                return False
            if nxt == "after_override":
                nxt = Mode.SURVEY if self.survey_pending else Mode.AUTONOMOUS
            elif nxt == "resume":
                nxt = self._resume or Mode.AUTONOMOUS
                if self.survey_pending and nxt == Mode.AUTONOMOUS:
                    nxt = Mode.SURVEY
            if nxt == Mode.SAFE_STOP and old != Mode.SAFE_STOP:
                self._resume = old
            if nxt == Mode.SURVEY:
                self.survey_pending = False
            if nxt == old and event != Ev.NEW_SESSION:
                return False
            self.mode = nxt
            self.since = self.clock()
            rec = {"from": old.value, "to": nxt.value, "event": event.value, "wall_time": time.time(), **info}
            self.history.append(rec)
            del self.history[:-200]
        log.info("MODE %s -> %s (%s)%s", old.value, nxt.value, event.value, f" {info}" if info else "")
        for fn in list(self._listeners):
            try:
                fn(old, nxt, event, info)
            except Exception as e:
                log.error("mode listener: %s", e)
        return True

    def status(self) -> dict:
        return {
            "mode": self.mode.value,
            "label": self.policy.label,
            "since_s": round(self.clock() - self.since, 1),
            "survey_pending": self.survey_pending,
            "policy": asdict(self.policy),
        }
