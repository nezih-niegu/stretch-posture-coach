"""End-to-end: synthetic camera + synthetic pose -> assessment -> coach (fake LLM) -> log -> report."""

import json

import numpy as np
import pytest
from synth import skeleton

from stretch_posture_coach.assessment import Assessor
from stretch_posture_coach.biomech import compute_metrics, gravity_from_pitch
from stretch_posture_coach.config import load_config
from stretch_posture_coach.llm import OllamaError, clean_spoken
from stretch_posture_coach.pipeline import Pipeline
from stretch_posture_coach.report import summarize, write_report
from stretch_posture_coach.robot import NullHead


class FakeSource:
    """Plays a scripted posture timeline at 10 Hz of video time."""

    is_live = False
    device_name = "fake"

    def __init__(self, script):
        self.script = script  # list of (seconds, skeleton kwargs)
        self.t = 0.0
        self.total = sum(s for s, _ in script)

    def current(self):
        acc = 0.0
        for dur, kw in self.script:
            acc += dur
            if self.t < acc:
                return kw
        return None

    def read(self):
        if self.t >= self.total:
            return None, self.t
        self.t += 0.1
        return np.zeros((64, 48, 3), np.uint8), self.t

    def close(self):
        pass


class FakeEstimator:
    def __init__(self, source):
        self.source = source

    def process(self, frame, ts):
        kw = self.source.current()
        return None if kw is None or kw == "absent" else skeleton(**kw)

    def close(self):
        pass


class FakeSpeaker:
    busy = False

    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)

    def close(self):
        pass


class FakeLLM:
    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def chat(self, messages, max_tokens=None):
        self.prompts.append(messages)
        if self.fail:
            raise OllamaError("down")
        return "**Stretch:** Sit back in your chair and let it support your spine. 😊 Great job so far!"


@pytest.fixture
def cfg(tmp_path):
    c = load_config(
        overrides={
            "logging": {"sessions_dir": str(tmp_path / "sessions")},
            "analysis": {"calibration_file": str(tmp_path / "cal.json")},
            "coach": {"min_interval_s": 5, "greet": False},
            "thresholds": {"hold_s": 3},
        }
    )
    return c


def run_pipeline(cfg, script, llm):
    src = FakeSource(script)
    speaker = FakeSpeaker()
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=NullHead(),
        speaker=speaker,
        llm=llm,
        window=False,
        stream=False,
        vr=False,
    )
    session = p.run()
    # coach threads are daemon threads; give them a moment
    import time

    for _ in range(50):
        if not p.coach.busy:
            break
        time.sleep(0.02)
    return p, speaker, session


def test_end_to_end_slouch_coached_then_praised(cfg):
    script = [
        (5, {"sitting": True, "elbow_flex": 80}),
        (8, {"sitting": True, "trunk_flex": 35, "elbow_flex": 80}),
        (6, {"sitting": True, "elbow_flex": 80}),
    ]
    llm = FakeLLM()
    p, speaker, session = run_pipeline(cfg, script, llm)
    assert llm.prompts, "LLM should have been asked"
    prompt = llm.prompts[0][-1]["content"]
    assert "slouching" in prompt and "sitting" in prompt
    assert speaker.said[0] == "Sit back in your chair and let it support your spine. Great job so far!"
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert "issue" in kinds and "praise" in kinds
    assert events[kinds.index("issue")]["issue"] == "trunk_flexion"

    s = summarize(session)
    assert s["samples"] > 50 and s["coaching_messages"] >= 1 and s["llm_messages"] >= 1
    out = write_report(session)
    assert out.exists() and "RULA" in out.read_text()


def test_fallback_when_llm_down(cfg):
    script = [(10, {"neck_flex": 40, "elbow_flex": 80})]
    _, speaker, _ = run_pipeline(cfg, script, FakeLLM(fail=True))
    assert speaker.said and "chin" in speaker.said[0].lower()


def test_no_coaching_for_short_violation(cfg):
    script = [(3, {}), (1.5, {"trunk_flex": 40}), (3, {})]
    llm = FakeLLM()
    _, speaker, _ = run_pipeline(cfg, script, llm)
    assert not llm.prompts and not speaker.said


def test_absent_person(cfg):
    _, speaker, session = run_pipeline(cfg, [(3, "absent")], FakeLLM())
    assert not speaker.said
    assert summarize(session)["in_view_min"] == 0


def test_assessor_issue_hysteresis(cfg):
    a = Assessor(cfg)
    g = gravity_from_pitch(0)
    t = 0.0

    def feed(trunk, secs):
        nonlocal t
        res = None
        for _ in range(int(secs * 10)):
            t += 0.1
            p = skeleton(trunk_flex=trunk)
            res = a.update(compute_metrics(p.world, p.visibility, g), t)
        return res

    assert not feed(35, 2).active
    assert [s.rule.id for s in feed(35, 2).active] == ["trunk_flexion"]
    assert feed(18, 3).active  # within hysteresis band (>= 0.85 * 20)
    assert not feed(5, 3).active


def test_clean_spoken():
    assert (
        clean_spoken('"Sit up tall. Relax your shoulders. Breathe. Smile."')
        == "Sit up tall. Relax your shoulders."
    )
    assert clean_spoken("请坐直") == ""
    assert clean_spoken("<think>hmm</think>Lift your chin.") == "Lift your chin."
