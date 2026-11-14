from synth import skeleton

from stretch_posture_coach.biomech import compute_metrics, gravity_from_pitch
from stretch_posture_coach.ergonomics import (
    REBA_TABLE_A,
    REBA_TABLE_B,
    REBA_TABLE_C,
    RULA_TABLE_A,
    RULA_TABLE_B,
    RULA_TABLE_C,
    reba,
    rula,
)


def m_of(**kw):
    p = skeleton(**kw)
    return compute_metrics(p.world, p.visibility, gravity_from_pitch(0))


def test_table_shapes():
    assert len(RULA_TABLE_A) == 6 and all(len(r) == 3 and all(len(c) == 8 for c in r) for r in RULA_TABLE_A)
    assert len(RULA_TABLE_B) == 6 and all(len(r) == 12 for r in RULA_TABLE_B)
    assert len(RULA_TABLE_C) == 8 and all(len(r) == 7 for r in RULA_TABLE_C)
    assert len(REBA_TABLE_A) == 3 and all(len(r) == 5 and all(len(c) == 4 for c in r) for r in REBA_TABLE_A)
    assert len(REBA_TABLE_B) == 2 and all(len(r) == 6 and all(len(c) == 3 for c in r) for r in REBA_TABLE_B)
    assert len(REBA_TABLE_C) == 12 and all(len(r) == 12 for r in REBA_TABLE_C)


def test_tables_monotonic():
    # scores never decrease when a body-part score increases
    for t in (RULA_TABLE_C, REBA_TABLE_C):
        for row in t:
            assert row == sorted(row)
        for col in zip(*t, strict=True):
            assert list(col) == sorted(col)


def test_neutral_posture_low_risk():
    m = m_of(elbow_flex=80)
    assert rula(m)["score"] <= 2
    assert reba(m)["score"] <= 3


def test_bad_posture_high_risk():
    m = m_of(trunk_flex=50, neck_flex=35, arm_flex=100, elbow_flex=20)
    r, b = rula(m), reba(m)
    assert r["score"] >= 4, r
    assert b["score"] >= 5 and b["risk"] in ("medium", "high", "very high"), b
    assert rula(m, static=True)["score"] >= 6
    assert rula(m, static=True)["risk"] in ("change soon", "change now")


def test_risk_increases_with_trunk_flexion():
    scores = [rula(m_of(trunk_flex=t))["score"] for t in (0, 15, 40, 70)]
    assert scores == sorted(scores) and scores[-1] > scores[0]


def test_static_adds_muscle_score():
    m = m_of(trunk_flex=30, neck_flex=25)
    assert rula(m, static=True)["score"] >= rula(m)["score"]
    assert reba(m, static=True)["score"] == reba(m)["score"] + 1


def test_none_when_not_visible():
    assert rula({"torso_visible": False}) is None
    assert reba({}) is None
