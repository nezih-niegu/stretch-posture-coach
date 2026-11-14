import math

import numpy as np
import pytest
from synth import skeleton

from stretch_posture_coach.config import load_config
from stretch_posture_coach.follow import Follower, person_distance


class FakeHead:
    has_base = True
    available = True

    def __init__(self, pan=0.0):
        self.pan = pan
        self.tilt = 0.0
        self.runstop = False
        self.rotations: list[float] = []
        self.translations: list[float] = []

    def base_rotate(self, rad, compensate_head=True):
        self.rotations.append(rad)
        if compensate_head:
            self.pan -= rad

    def base_translate(self, m):
        self.translations.append(m)

    def track(self, target_xy, now):
        pass

    def close(self):
        pass


@pytest.fixture
def cfg():
    return load_config()


def make(cfg, pan=0.0, **follow):
    f_cfg = {**cfg["follow"], "enabled": True, **follow}
    head = FakeHead(pan)
    return Follower(f_cfg, cfg["robot"], head), head


def run(f, dist, n=20, t0=0.0, dt=0.1, target_x=0.5, source="depth", present=True):
    t = t0
    st = None
    for _ in range(n):
        t += dt
        st = f.update(present, dist, source, target_x, t)
    return st, t


def test_disabled_without_base(cfg):
    head = FakeHead()
    head.has_base = False
    f = Follower({**cfg["follow"], "enabled": True}, cfg["robot"], head)
    assert not f.enabled
    st, _ = run(f, 4.0)
    assert st.action == "idle" and not head.translations


def test_holds_inside_band(cfg):
    f, head = make(cfg)
    for d in (2.0, 2.5, 3.0):
        st, _ = run(f, d)
        assert st.action == "hold"
    assert not head.translations and not head.rotations


def test_approaches_when_far_but_never_inside_min(cfg):
    f, head = make(cfg)
    run(f, 3.6, n=40)
    assert head.translations and all(0 < s <= 0.15 for s in head.translations)
    f2, head2 = make(cfg)
    run(f2, 3.05, n=40)  # just outside the band: small step that never ends closer than min_m
    assert all(3.05 - s >= 2.0 for s in head2.translations)


def test_backs_off_when_close_and_respects_limit(cfg):
    f, head = make(cfg, max_backoff_total_m=0.5)
    st, _ = run(f, 1.7, n=200)
    assert head.translations and all(s < 0 for s in head.translations)
    assert -sum(head.translations) <= 0.5 + 1e-9
    assert st.action == "blocked" and "back-up" in st.note


def test_emergency_full_step(cfg):
    f, head = make(cfg)
    run(f, 1.0, n=10)
    assert head.translations[0] == pytest.approx(-0.15)


def test_rotates_toward_person_and_compensates_head(cfg):
    f, head = make(cfg, pan=math.radians(50))
    st, _ = run(f, 2.5, n=200)
    assert head.rotations and all(r > 0 for r in head.rotations)  # CCW toward a person on the left
    assert all(abs(r) <= math.radians(20) + 1e-9 for r in head.rotations)
    assert abs(math.degrees(head.pan)) <= 12  # head back near forward: body faces the person
    assert st.action == "hold"


def test_no_forward_motion_when_not_facing(cfg):
    f, head = make(cfg, pan=math.radians(90))
    f.head.base_rotate = lambda rad, compensate_head=True: head.rotations.append(rad)  # pan never recenters
    run(f, 4.0, n=100)
    assert not head.translations


def test_freezes_when_lost_or_runstop(cfg):
    f, head = make(cfg)
    st, _ = run(f, None, n=30, present=False)
    assert st.action == "lost" and not head.translations
    head.runstop = True
    st, _ = run(f, 4.0, n=30)
    assert st.action == "runstop" and not head.translations


def test_travel_limit(cfg):
    f, head = make(cfg, max_travel_m=0.3)
    st, _ = run(f, 5.0, n=400)
    assert sum(head.translations) <= 0.3 + 1e-9
    assert st.action == "blocked"


def test_needs_stable_detection(cfg):
    f, head = make(cfg)
    run(f, 4.0, n=3)
    assert not head.translations


def test_distance_from_depth_and_fallback():
    p = skeleton()
    p.image = np.column_stack([0.5 + p.world[:, 0] * 0.3, 0.5 + p.world[:, 1] * 0.3])
    depth = np.full((640, 480), 2.37, np.float32)
    d, src = person_distance(p, depth)
    assert src == "depth" and d == pytest.approx(2.37, abs=0.01)
    depth[:] = 0  # no valid depth -> monocular estimate
    d, src = person_distance(p, depth)
    assert src == "pose" and 1.0 < d < 4.0
    assert person_distance(None, depth) == (None, "none")


def test_pipeline_follow_end_to_end(tmp_path):
    """Person stands 3.8 m away (depth) -> Stretch steps closer, logs follow events, says nothing odd."""
    import json

    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline

    cfg = load_config(
        overrides={
            "logging": {"sessions_dir": str(tmp_path / "s")},
            "analysis": {"calibration_file": str(tmp_path / "c.json")},
            "follow": {"enabled": True},
            "coach": {"greet": False, "persona": "norteno"},
        }
    )
    src = FakeSource([(15, {})])
    src.last_depth = np.full((640, 480), 3.8, np.float32)
    head = FakeHead()
    speaker = FakeSpeaker()
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=head,
        speaker=speaker,
        llm=FakeLLM(),
        window=False,
        stream=False,
    )
    session = p.run()
    assert head.translations and all(0 < t <= 0.15 for t in head.translations)
    events = [json.loads(x) for x in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "follow" and e["action"] == "approach" for e in events)
    header = (session / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "distance_m" in header


def test_pipeline_follow_backoff_announces_in_norteno(tmp_path):
    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline

    cfg = load_config(
        overrides={
            "logging": {"enabled": False},
            "analysis": {"calibration_file": str(tmp_path / "c.json")},
            "follow": {"enabled": True},
            "coach": {"greet": False, "persona": "norteno"},
        }
    )
    src = FakeSource([(5, {})])
    src.last_depth = np.full((640, 480), 1.6, np.float32)
    head = FakeHead()
    speaker = FakeSpeaker()
    Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=head,
        speaker=speaker,
        llm=FakeLLM(),
        window=False,
        stream=False,
        log_session=False,
    ).run()
    assert head.translations and head.translations[0] < 0
    assert speaker.said and "pa'trás" in speaker.said[0]
