"""Temporal layer: smoothing, static-posture detection, RULA/REBA, and sustained-issue tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from . import ergonomics


@dataclass
class Rule:
    id: str
    label: str  # short human description, used in prompts and fallback speech
    # returns (ratio, value, detail) where ratio >= 1 means violated; None when not measurable
    check: Callable[[dict, dict], tuple[float, float, str] | None]
    priority: float = 1.0


def _ratio(value: float | None, threshold: float) -> float | None:
    if value is None or threshold <= 0:
        return None
    return value / threshold


def _side_word(v: float) -> str:
    return "left" if v > 0 else "right"


def _rule_neck_flexion(m, th):
    v = m.get("neck_flexion")
    r = _ratio(v, th["neck_flexion"])
    return None if r is None else (r, v, f"head {v:.0f} deg forward/down from the trunk")


def _rule_neck_extension(m, th):
    v = m.get("neck_flexion")
    r = _ratio(None if v is None else -v, th["neck_extension"])
    return None if r is None else (r, v, f"head tilted back {abs(v):.0f} deg")


def _rule_neck_side(m, th):
    v = m.get("neck_side_bend")
    r = _ratio(None if v is None else abs(v), th["neck_side_bend"])
    return None if r is None else (r, v, f"head tilted {abs(v):.0f} deg toward the {_side_word(v)} shoulder")


def _rule_neck_twist(m, th):
    v = m.get("neck_twist")
    r = _ratio(None if v is None else abs(v), th["neck_twist"])
    return None if r is None else (r, v, f"head turned {abs(v):.0f} deg to the {_side_word(v)}")


def _rule_trunk_flexion(m, th):
    v = m.get("trunk_flexion")
    r = _ratio(v, th["trunk_flexion"])
    what = "slouching / leaning forward in the chair" if m.get("posture") == "sitting" else "bending forward"
    return None if r is None else (r, v, f"{what} ({v:.0f} deg)")


def _rule_trunk_extension(m, th):
    if m.get("posture") == "sitting":  # reclining against a backrest is fine
        return None
    v = m.get("trunk_flexion")
    r = _ratio(None if v is None else -v, th["trunk_extension"])
    return None if r is None else (r, v, f"leaning back {abs(v or 0):.0f} deg while standing")


def _rule_trunk_side(m, th):
    v = m.get("trunk_side_bend")
    r = _ratio(None if v is None else abs(v), th["trunk_side_bend"])
    return None if r is None else (r, v, f"trunk leaning {abs(v):.0f} deg to the {_side_word(v)}")


def _rule_trunk_twist(m, th):
    v = m.get("trunk_twist")
    r = _ratio(None if v is None else abs(v), th["trunk_twist"])
    return None if r is None else (r, v, f"upper body twisted {abs(v):.0f} deg relative to the hips")


def _rule_shoulder_tilt(m, th):
    v = m.get("shoulder_tilt_cm")
    r = _ratio(None if v is None else abs(v), th["shoulder_tilt_cm"])
    higher = "left" if (v or 0) > 0 else "right"
    return None if r is None else (r, v, f"{higher} shoulder {abs(v or 0):.0f} cm higher than the other")


def _rule_shoulders_raised(m, th):
    if "l_shoulder_raised" not in m and "r_shoulder_raised" not in m:
        return None
    sides = [s for s in ("l", "r") if m.get(f"{s}_shoulder_raised")]
    if not sides:
        return (0.0, 0.0, "")
    which = (
        "both shoulders" if len(sides) == 2 else ("left shoulder" if sides[0] == "l" else "right shoulder")
    )
    return (1.2, float(len(sides)), f"{which} shrugged up toward the ears")


def _rule_arms_raised(m, th):
    v = m.get("max_upper_arm_elevation")
    r = _ratio(v, th["upper_arm_elevation"])
    return None if r is None else (r, v, f"arm raised {v:.0f} deg away from the body")


def _rule_lumbar(m, th):
    v = m.get("l5s1_compression_n")
    r = _ratio(v, th["l5s1_compression_n"])
    return None if r is None else (r, v, f"estimated low-back compression {v:.0f} N")


RULES: list[Rule] = [
    Rule("neck_flexion", "head forward / looking down", _rule_neck_flexion, 1.2),
    Rule("neck_extension", "head tilted back", _rule_neck_extension, 0.9),
    Rule("neck_side_bend", "head tilted sideways", _rule_neck_side, 0.9),
    Rule("neck_twist", "head turned to one side", _rule_neck_twist, 0.7),
    Rule("trunk_flexion", "slouching / leaning forward", _rule_trunk_flexion, 1.3),
    Rule("trunk_extension", "leaning back", _rule_trunk_extension, 0.8),
    Rule("trunk_side_bend", "leaning sideways", _rule_trunk_side, 1.0),
    Rule("trunk_twist", "twisted torso", _rule_trunk_twist, 0.9),
    Rule("shoulder_tilt", "uneven shoulders", _rule_shoulder_tilt, 0.8),
    Rule("shoulders_raised", "shrugged shoulders", _rule_shoulders_raised, 1.0),
    Rule("arms_raised", "arms raised for a long time", _rule_arms_raised, 0.9),
    Rule("lumbar_load", "high low-back load", _rule_lumbar, 1.5),
]


class Smoother:
    """EMA over numeric metrics; non-numeric values pass through."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.state: dict[str, float] = {}

    def reset(self):
        self.state.clear()

    def __call__(self, m: dict) -> dict:
        out = dict(m)
        for k, v in m.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            prev = self.state.get(k)
            sm = v if prev is None else prev + self.alpha * (v - prev)
            self.state[k] = sm
            out[k] = round(sm, 1)
        return out


@dataclass
class IssueState:
    rule: Rule
    since: float | None = None  # when the current violation started
    clear_since: float | None = None
    active: bool = False
    ratio: float = 0.0
    value: float = 0.0
    detail: str = ""
    last_coached: float | None = None
    activated_at: float | None = None
    cleared_at: float | None = None

    def duration(self, now: float) -> float:
        return 0.0 if self.since is None else now - self.since


@dataclass
class Assessment:
    t: float
    present: bool
    metrics: dict = field(default_factory=dict)
    rula: dict | None = None
    reba: dict | None = None
    static: bool = False
    static_for_s: float = 0.0
    active: list[IssueState] = field(default_factory=list)  # sorted, most important first


class Assessor:
    STATIC_KEYS = ("trunk_flexion", "neck_flexion", "max_upper_arm_elevation")
    STATIC_TOL_DEG = 12.0

    def __init__(self, cfg: dict):
        self.th = dict(cfg["thresholds"])
        self.hold_s = float(self.th.get("hold_s", 6))
        a = cfg.get("analysis", {})
        s = cfg.get("subject", {})
        self.static_s = float(a.get("static_posture_s", 60))
        self.arms_supported = bool(s.get("arms_supported", False))
        self.load_kg = float(s.get("hand_load_kg", 0.0)) * 2
        self.smoother = Smoother(float(a.get("smoothing_alpha", 0.35)))
        self.issues = {r.id: IssueState(r) for r in RULES}
        self._static_ref: dict | None = None
        self._static_since: float | None = None
        self._last_seen: float | None = None

    def update(self, raw: dict | None, now: float) -> Assessment:
        if not raw or not raw.get("torso_visible"):
            if self._last_seen is None or now - self._last_seen > 1.5:
                self.smoother.reset()
                self._static_ref = None
                for st in self.issues.values():
                    st.since, st.active, st.clear_since = None, False, None
            return Assessment(t=now, present=False)
        self._last_seen = now
        m = self.smoother(raw)

        static_for = self._update_static(m, now)
        static = static_for >= self.static_s
        rula = ergonomics.rula(m, self.arms_supported, static, self.load_kg)
        reba = ergonomics.reba(m, self.arms_supported, static, self.load_kg)

        for st in self.issues.values():
            res = st.rule.check(m, self.th)
            self._update_issue(st, res, now)

        active = [st for st in self.issues.values() if st.active]
        active.sort(
            key=lambda s: s.rule.priority * min(s.ratio, 3.0) * (1 + min(s.duration(now), 120) / 60),
            reverse=True,
        )
        return Assessment(
            t=now,
            present=True,
            metrics=m,
            rula=rula,
            reba=reba,
            static=static,
            static_for_s=static_for,
            active=active,
        )

    def _update_issue(self, st: IssueState, res, now: float):
        if res is None:
            st.since, st.active, st.clear_since = None, False, None
            return
        ratio, value, detail = res
        st.ratio, st.value = ratio, value
        if detail:
            st.detail = detail
        if ratio >= 1.0:
            st.clear_since = None
            if st.since is None:
                st.since = now
            if not st.active and now - st.since >= self.hold_s:
                st.active, st.activated_at = True, now
        else:
            if st.active and ratio >= 0.85:  # hysteresis band: stay active
                return
            # tolerate a short dip below threshold (1 s grace) before clearing
            if st.clear_since is None:
                st.clear_since = now
            if now - st.clear_since >= 1.0:
                if st.active:
                    st.cleared_at = now
                st.since, st.active = None, False

    def _update_static(self, m: dict, now: float) -> float:
        cur = {k: m.get(k) for k in self.STATIC_KEYS if m.get(k) is not None}
        if self._static_ref is None or any(
            abs(cur[k] - self._static_ref.get(k, cur[k])) > self.STATIC_TOL_DEG for k in cur
        ):
            self._static_ref, self._static_since = cur, now
        return now - (self._static_since or now)
