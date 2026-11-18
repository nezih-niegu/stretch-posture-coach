"""Mode FSM, Control Barrier Functions, the 360° survey and the expert voice relay."""

import math
import time

import numpy as np
import pytest
from vr_sim import SimRobot, sim_world

from stretch_posture_coach.cbf import CBFSafety, SafeBase
from stretch_posture_coach.config import load_config
from stretch_posture_coach.modes import POLICIES, Ev, Mode, ModeMachine


def test_fsm_teleop_overrides_routine_and_survey_resumes():
    m = ModeMachine(Mode.AUTONOMOUS)
    seen = []
    m.on_change(lambda old, new, ev, info: seen.append((old, new, ev)))
    assert m.dispatch(Ev.NEW_SESSION) and m.mode == Mode.SURVEY
    assert m.dispatch(Ev.TELEOP_ON) and m.mode == Mode.TELEOP  # the expert overrides the survey...
    assert m.survey_pending  # ...which is queued
    assert m.policy.coaching == "background" and m.policy.base == "teleop" and m.policy.arm == "teleop"
    assert not m.dispatch(Ev.NEW_SESSION)  # still the expert's
    assert m.dispatch(Ev.TELEOP_OFF) and m.mode == Mode.SURVEY  # back to the queued survey
    assert m.dispatch(Ev.SURVEY_DONE) and m.mode == Mode.AUTONOMOUS and not m.survey_pending
    assert POLICIES[m.mode].coaching == "speak"
    assert [s[2] for s in seen] == [Ev.NEW_SESSION, Ev.TELEOP_ON, Ev.TELEOP_OFF, Ev.SURVEY_DONE]


def test_fsm_fall_and_runstop():
    m = ModeMachine(Mode.AUTONOMOUS)
    m.dispatch(Ev.FALL)
    assert m.mode == Mode.FALL_RESPONSE and m.policy.base == "none" and m.policy.coaching == "off"
    m.dispatch(Ev.TELEOP_ON)  # an expert can take over to help
    assert m.mode == Mode.TELEOP
    assert not m.dispatch(Ev.FALL)  # expert stays in charge during a fall
    m.dispatch(Ev.RUNSTOP)
    assert m.mode == Mode.SAFE_STOP and m.policy.head == "hold"
    m.dispatch(Ev.RUNSTOP_CLEARED)
    assert m.mode == Mode.TELEOP  # back where it was
    m.dispatch(Ev.TELEOP_OFF)
    assert m.mode == Mode.AUTONOMOUS


def _wall(d, half=1.0):
    return np.array([[d, y] for y in np.linspace(-half, half, 41)])


def test_cbf_keeps_barrier_in_closed_loop_and_allows_escape():
    c = CBFSafety({})
    x, hs = 0.0, []
    for _ in range(400):  # full speed at a wall 2 m ahead
        r = c.filter(0.3, 0.0, _wall(2.0) - [x, 0])
        x += r.v * 0.05
        hs.append(r.h_min)
    assert min(hs) >= -1e-3  # h never negative: the margin is never crossed
    assert 2.0 - x == pytest.approx(c.d + c.l, abs=0.02)  # stops right at the barrier
    assert c.filter(-0.3, 0.0, _wall(2.0) - [x, 0]).v == pytest.approx(-0.3)  # backing off is free
    assert c.filter(0.0, 0.6, _wall(2.0) - [x, 0]).w == pytest.approx(0.6)  # so is turning away
    r = c.filter(0.3, 0.0, np.array([[0.6, -0.5], [0.6, 0.5]]))  # gap between two people/legs
    assert r.feasible and 0 <= r.v <= 0.3


def test_cbf_lines_and_safe_follow_steps():
    c = CBFSafety({})
    lines = c.barrier_lines(_wall(1.0), (1.0, 2.0, math.pi / 2))
    assert len(lines) == 1
    p = np.array(lines[0]["pts"])
    # map frame: the robot at (1, 2) facing +y, so the wall 1 m ahead is at y = 3; the keep-out line is d before it
    assert p[:, 1].min() == pytest.approx(3.0 - c.d, abs=0.06)
    robot = SimRobot()

    class W:
        cbf = c

        @staticmethod
        def obstacle_points():
            return _wall(0.55)

    safe = SafeBase(robot, W())
    safe.base_translate(0.25)  # follow wants to approach: the barrier caps the step
    assert robot.true[0] < 0.05
    assert safe.has_base and safe.pan == robot.pan  # everything else passes through

    # closed loop with discrete steps: follow keeps asking for 0.25 m toward a wall 1.2 m away
    robot2 = SimRobot()

    class W2:
        cbf = c

        @staticmethod
        def obstacle_points():
            return _wall(1.2) - [robot2.true[0], 0]

    safe2 = SafeBase(robot2, W2())
    for _ in range(30):
        safe2.base_translate(0.25)
        e = 1.2 - robot2.true[0] - c.l  # front circle to the wall
        assert e >= c.d - 1e-6  # never inside the margin
    assert 1.2 - robot2.true[0] - c.l == pytest.approx(c.d, abs=0.03)  # and it gets close to it


def test_survey_resets_registration_sweeps_and_checks():
    from stretch_posture_coach.survey import Survey

    cfg = load_config(
        overrides={"survey": {"dwell_s": 0.0, "settle_s": 0.0, "rotate_base": "true", "spin_dwell_s": 0.0}}
    )
    robot = SimRobot()
    world, lid = sim_world(cfg, robot)
    try:
        world.map.integrate(np.array([[5.0, 5.0, 1.0]]), np.array([[1.0, 2.0, 3.0]]))  # stale map
        said = []
        sv = Survey(cfg, robot, world, say=said.append)
        looks = []
        robot.look = lambda pan, tilt, now: looks.append((pan, tilt))
        t = 0.0
        sv.start(t)
        assert len(world.map) == 0 and said == ["survey_start"]  # fresh map frame
        world.aligner.n = 2  # (floor fits come from the depth frames in the real pipeline)
        world.map.integrate(np.random.default_rng(0).uniform(-2, 2, (3000, 3)), np.full((3000, 3), 100.0))
        world.map.integrate(np.random.default_rng(0).uniform(-2, 2, (3000, 3)), np.full((3000, 3), 100.0))
        time.sleep(1.0)  # some lidar scans
        r = None
        for _ in range(200):
            t += 0.1
            r = sv.step(t) or r
            if r:
                break
        assert r == "done", sv.result
        pans = [p for p, _ in looks]
        assert min(pans) <= -3.8 and max(pans) >= 1.5  # the head swept its whole range
        assert looks[-1] == (
            cfg["follow"]["forward_pan_rad"],
            cfg["robot"]["front_tilt_rad"],
        )  # back to the front
        assert sv.result["base_spin_deg"] == 360 and robot.true[2] == pytest.approx(0.0, abs=1e-6)
        assert said[-1] == "survey_done"
    finally:
        lid.close()
        world.close()


def test_survey_skips_spin_when_a_barrier_is_too_close():
    from stretch_posture_coach.survey import Survey

    cfg = load_config(overrides={"survey": {"rotate_base": "auto"}})

    class W:
        cbf = CBFSafety({})

        @staticmethod
        def obstacle_points():
            return np.array([[0.3, 0.25], [-0.3, -0.25]])  # someone right next to the base corners

    sv = Survey(cfg, SimRobot(), W())
    assert not sv._may_spin()


class FakeSTT:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def __call__(self, audio, language):
        self.calls += 1
        return self.text


class FakeLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages, max_tokens=None):
        self.messages = messages
        return "Stretch: Por favor, siéntate derecho y apoya la espalda."


def test_voice_relay_stt_llm_tts_and_partials():
    from stretch_posture_coach.voice import SAMPLE_RATE, VoiceRelay

    cfg = load_config(overrides={"coach": {"persona": "norteno"}, "voice": {"partial_every_s": 0.15}})
    spoken, notes, events = [], [], []
    llm = FakeLLM()
    stt = FakeSTT("dile que se siente derecho")
    vr = VoiceRelay(cfg, spoken.append, llm, stt, notes.append, context=lambda: None, log_event=events.append)
    assert vr.language == "es"
    vr.begin()
    pcm = (np.sin(np.arange(SAMPLE_RATE) / 10) * 8000).astype("<i2").tobytes()  # 1 s of "speech"
    for _ in range(4):
        vr.feed(pcm[: len(pcm) // 2])
        vr.feed(pcm[len(pcm) // 2 :])
        time.sleep(0.12)
    vr.end()
    for _ in range(50):
        if spoken:
            break
        time.sleep(0.05)
    assert spoken == ["Por favor, siéntate derecho y apoya la espalda."]  # prefix stripped, robot's voice
    assert any(n.get("partial") for n in notes)  # live partial transcript while talking
    assert notes[-1]["state"] == "spoken" and notes[-1]["heard"] == "dile que se siente derecho"
    assert (
        "norteño" in llm.messages[0]["content"] and "dile que se siente derecho" in llm.messages[-1]["content"]
    )
    assert events[-1]["kind"] == "expert" and events[-1]["source"] == "llm"


def test_voice_relay_verbatim_and_too_short():
    from stretch_posture_coach.voice import VoiceRelay

    cfg = load_config(overrides={"voice": {"mode": "verbatim"}})
    spoken, notes = [], []
    vr = VoiceRelay(cfg, spoken.append, FakeLLM(), FakeSTT("Keep your back straight."), notes.append)
    vr.begin()
    vr.feed(b"\x00\x00" * 1000)  # 60 ms
    vr.end()
    time.sleep(0.2)
    assert not spoken and "too short" in notes[-1]["error"]
    vr.begin()
    vr.feed(b"\x01\x00" * 16000)
    vr.end()
    time.sleep(0.3)
    assert spoken == ["Keep your back straight."]


@pytest.mark.parametrize(
    "heard,said",
    [  # transcripts from a real session (Nov 17)
        ("Ask the patient to stand up.", "Please stand up."),
        ("Hello, Stretch. Can you tell the patient to stand up?", "Please stand up."),
        ("Asked the patient how has he been doing lately?", "How have you been doing lately?"),
        ("Ask her if she feels any pain", "Do you feel any pain?"),
        ("Tell her to relax her shoulders", "Please relax your shoulders."),
        ("Pregúntale cómo se siente", "¿Cómo se siente?"),
    ],
)
def test_relay_rules_for_clear_commands(heard, said):
    from stretch_posture_coach.voice import rule_relay

    assert rule_relay(heard, heard.startswith("Pregúnt")) == said


def test_relay_rejects_llm_replies_that_talk_back():
    from stretch_posture_coach.voice import VoiceRelay

    class TalksBack:
        def chat(self, messages, max_tokens=None):
            return "Thanks for the instruction, specialist. I'll make sure to pass the trap."

    vr = VoiceRelay(load_config(), print, TalksBack(), FakeSTT(""))
    assert vr.compose("Pass the trap.") == ("Pass the trap.", "verbatim")  # the words, not the chatbot reply
    assert vr.compose("Ask the patient to stand up.") == ("Please stand up.", "rule")  # no LLM needed
