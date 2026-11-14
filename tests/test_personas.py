from stretch_posture_coach import personas
from stretch_posture_coach.assessment import RULES, Assessment, IssueState
from stretch_posture_coach.coach import build_prompt
from stretch_posture_coach.config import load_config


def _assessment():
    st = IssueState(next(r for r in RULES if r.id == "trunk_flexion"), since=0.0, active=True, detail="x")
    a = Assessment(
        t=15.0,
        present=True,
        metrics={"posture": "sitting"},
        rula={"score": 5, "risk": "change soon"},
        reba={"score": 6, "risk": "medium"},
        active=[st],
    )
    return a, st


def test_resolve():
    assert personas.resolve({"persona": "default", "language": "English"}).key == "default_en"
    assert personas.resolve({"persona": "default", "language": "Spanish"}).key == "default_es"
    assert personas.resolve({"persona": "norteño"}).key == "norteno"


def test_every_persona_has_all_fallbacks():
    ids = {r.id for r in RULES} | {
        "break",
        "praise",
        "greet",
        "follow_back",
        "follow_limit",
        "fall_check",
        "fall_alarm",
        "fall_ok",
    }
    for p in personas.PERSONAS.values():
        assert ids <= set(p.fallback), p.key
    assert set(personas.LABELS_ES) == {r.id for r in RULES}


def test_norteno_prompt_is_spanish():
    cfg = load_config(overrides={"coach": {"persona": "norteno", "user_name": "Chuy"}})
    a, st = _assessment()
    msgs = build_prompt("issue", a, [st], ["¡Ey compa!"], cfg, 15.0)
    assert msgs[0]["role"] == "system" and "sierra" in msgs[0]["content"] and "Chuy" in msgs[0]["content"]
    assert msgs[-1]["content"].startswith("Observación: la persona está sentada")
    assert "encorvado" in msgs[-1]["content"] and "No repitas" in msgs[-1]["content"]
    assert len(msgs) == 1 + 2 * len(personas.EXAMPLES_NORTENO) + 1


def test_english_prompt_unchanged():
    cfg = load_config()
    a, st = _assessment()
    msgs = build_prompt("issue", a, [st], [], cfg, 15.0)
    assert "Observation: the user is sitting" in msgs[-1]["content"]


def test_modelfile():
    mf = personas.modelfile("norteno", "qwen2:7b")
    assert mf.startswith("FROM qwen2:7b\n")
    assert "PARAMETER temperature 0.8" in mf
    assert 'SYSTEM """' in mf and mf.count("MESSAGE user") == len(personas.EXAMPLES_NORTENO)
    assert '"""' not in personas.system_prompt(personas.PERSONAS["norteno"], {})
