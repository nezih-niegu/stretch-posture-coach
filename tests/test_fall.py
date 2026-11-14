import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from synth import skeleton

from stretch_posture_coach.alarm import Alarm, siren_wav
from stretch_posture_coach.biomech import compute_metrics, gravity_from_pitch
from stretch_posture_coach.config import load_config
from stretch_posture_coach.fall import ALARM, CHECK, IMPACT, OK, FallDetector, FallFeatures

CFG = load_config()["fall"]


def play(det, segments, t0=0.0, dt=0.1):
    """segments: list of (seconds, dict of FallFeatures fields). Returns list of (t, state, reason)."""
    t, changes = t0, []
    for secs, kw in segments:
        for _ in range(int(round(secs / dt))):
            t += dt
            ch = det.update(
                FallFeatures(
                    t=t, present=kw.get("present", True), **{k: v for k, v in kw.items() if k != "present"}
                )
            )
            if ch:
                changes.append((round(t, 1), *ch))
    return changes


STAND = {"inclination": 5, "hip_height": 0.95, "hip_y": 0.55, "torso_len": 0.2}
LYING = {"inclination": 85, "hip_height": 0.2, "hip_y": 0.8, "torso_len": 0.15}


def states(changes):
    return [c[1] for c in changes]


def test_fall_then_no_recovery_raises_alarm():
    det = FallDetector(CFG)
    ch = play(
        det,
        [
            (2, STAND),
            (0.3, {"inclination": 45, "hip_height": 0.6, "hip_y": 0.65, "torso_len": 0.2}),
            (15, LYING),
        ],
    )
    assert states(ch) == [IMPACT, CHECK, ALARM]
    assert ch[2][0] - ch[1][0] == pytest.approx(CFG["check_s"], abs=0.2)
    assert det.active


def test_sitting_down_fast_is_not_a_fall():
    det = FallDetector(CFG)
    ch = play(
        det,
        [
            (2, STAND),
            (0.4, {"inclination": 10, "hip_height": 0.5, "hip_y": 0.7, "torso_len": 0.2}),
            (6, {"inclination": 10, "hip_height": 0.48, "hip_y": 0.72, "torso_len": 0.2}),
        ],
    )
    assert CHECK not in states(ch) and ALARM not in states(ch)
    assert det.state == OK


def test_bending_over_is_not_a_fall():
    det = FallDetector(CFG)
    ch = play(
        det,
        [
            (2, STAND),
            (3, {"inclination": 80, "hip_height": 0.85, "hip_y": 0.58, "torso_len": 0.12}),
            (2, STAND),
        ],
    )
    assert ch == []


def test_fall_then_stands_up_cancels():
    det = FallDetector(CFG)
    ch = play(det, [(2, STAND), (0.3, {**LYING, "inclination": 60}), (4, LYING), (4, STAND)])
    assert states(ch) == [IMPACT, CHECK, OK] and "upright" in ch[-1][2]


def test_raised_hand_during_check_means_ok():
    det = FallDetector(CFG)
    ch = play(det, [(2, STAND), (0.3, LYING), (3, LYING), (2, {**LYING, "hand_raised": True})])
    assert states(ch)[-1] == OK and "hand" in ch[-1][2]


def test_fall_detected_from_image_only_without_depth():
    det = FallDetector(CFG)
    nodepth = lambda d: {**d, "hip_height": None}  # noqa: E731
    ch = play(det, [(2, nodepth(STAND)), (0.4, nodepth({**LYING, "hip_y": 0.82})), (14, nodepth(LYING))])
    assert states(ch) == [IMPACT, CHECK, ALARM]
    assert "torso lengths" in ch[0][2]


def test_vanishing_while_falling():
    det = FallDetector(CFG)
    ch = play(
        det,
        [
            (2, STAND),
            (0.3, {**STAND, "inclination": 30, "hip_y": 0.62}),
            (0.3, {**STAND, "inclination": 40, "hip_y": 0.68, "hip_height": None}),
            (15, {"present": False}),
        ],
    )
    assert states(ch) == [IMPACT, CHECK, ALARM]
    assert "out of view" in ch[0][2]


def test_walking_out_of_view_is_not_a_fall():
    det = FallDetector(CFG)
    ch = play(det, [(2, STAND), (15, {"present": False})])
    assert ch == []


def test_long_lying_on_floor_triggers_check():
    det = FallDetector(CFG)
    ch = play(det, [(35, LYING)])
    assert states(ch) == [CHECK] and "floor" in ch[0][2]
    det2 = FallDetector({**CFG, "long_lying_s": 0})
    assert play(det2, [(35, LYING)]) == []


def test_lying_on_a_bed_is_not_floor():
    det = FallDetector(CFG)
    assert play(det, [(40, {**LYING, "hip_height": 0.55})]) == []


def test_acknowledge_clears_alarm():
    det = FallDetector(CFG)
    play(det, [(2, STAND), (0.3, LYING), (14, LYING)])
    assert det.state == ALARM
    st, why = det.acknowledge(20.0, "web")
    assert st == OK and "web" in why and not det.active


def test_features_from_synthetic_skeleton():
    g = gravity_from_pitch(0)
    up = compute_metrics(*(lambda p: (p.world, p.visibility))(skeleton()), g)
    down = compute_metrics(*(lambda p: (p.world, p.visibility))(skeleton(trunk_flex=88, drop=0.6)), g)
    hand = compute_metrics(*(lambda p: (p.world, p.visibility))(skeleton(hand_up=True)), g)
    assert up["trunk_inclination"] < 5 and down["trunk_inclination"] > 80
    assert hand["hand_raised"] and not up["hand_raised"]


def test_siren_wav_and_notifications():
    wav = siren_wav(1.0)
    assert wav[:4] == b"RIFF" and len(wav) > 40000
    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, dict(self.headers), body))
            self.send_response(200)
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    al = Alarm(
        {"ntfy_topic": "stretch-test", "ntfy_server": base, "webhook_url": base + "/hook"},
        None,
        {},
        live_url="http://10.0.0.5:8080/",
    )
    al.notify("alarm", "hips dropped 0.7 m; now lying")
    srv.shutdown()
    ntfy = next(g for g in got if g[0] == "/stretch-test")
    assert ntfy[1]["Priority"] == "urgent" and ntfy[1]["Click"] == "http://10.0.0.5:8080/"
    assert "hips dropped" in ntfy[2].decode()
    hook = json.loads(next(g for g in got if g[0] == "/hook")[2])
    assert hook["event"] == "fall_alarm" and hook["live_url"] == "http://10.0.0.5:8080/"


def test_pipeline_fall_end_to_end(tmp_path):
    from test_follow import FakeHead
    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline

    class Speaker(FakeSpeaker):
        def say_blocking(self, text):
            self.said.append(text)

    cfg = load_config(
        overrides={
            "logging": {"sessions_dir": str(tmp_path / "s")},
            "analysis": {"calibration_file": str(tmp_path / "c")},
            "coach": {"greet": False, "persona": "norteno"},
            "alarm": {"siren": False, "repeat_s": 0.2},
            "follow": {"enabled": True},
        }
    )
    src = FakeSource([(3, {}), (0.3, {"trunk_flex": 50, "drop": 0.3}), (16, {"trunk_flex": 88, "drop": 0.6})])
    src.last_depth = np.full((640, 480), 2.5, np.float32)  # standing inside the 2-3 m band
    head, speaker = FakeHead(), Speaker()
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=head,
        speaker=speaker,
        llm=FakeLLM(),
        window=False,
        stream=False,
        vr=False,
    )
    alarm_seen = []
    orig = p._publish_alarm
    p._publish_alarm = lambda: (orig(), alarm_seen.append(p.fall.state))
    session = p.run()
    assert CHECK in alarm_seen and ALARM in alarm_seen
    assert any("¿Estás bien?" in s for s in speaker.said)  # norteño "are you OK?"
    assert any("Alerta" in s for s in speaker.said)  # spoken alarm
    assert not head.translations  # base paused during the check / alarm
    events = [json.loads(x) for x in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    falls = [e["state"] for e in events if e["kind"] == "fall"]
    assert falls[:3] == [IMPACT, CHECK, ALARM]
    assert list(session.glob("fall_*_alarm.jpg"))  # snapshot saved with the session
    t_check = next(e["wall_time"] for e in events if e["kind"] == "fall" and e["state"] == CHECK)
    assert not [
        e for e in events if e["kind"] == "issue" and e["wall_time"] > t_check
    ]  # no coaching meanwhile
    # acknowledgement path
    p2_src = FakeSource(
        [(3, {}), (0.3, {"trunk_flex": 50, "drop": 0.3}), (16, {"trunk_flex": 88, "drop": 0.6})]
    )
    p2 = Pipeline(
        cfg | {"logging": {**cfg["logging"], "enabled": False}},
        source=p2_src,
        estimator=FakeEstimator(p2_src),
        head=FakeHead(),
        speaker=Speaker(),
        llm=FakeLLM(),
        window=False,
        stream=False,
        vr=False,
        log_session=False,
    )
    for _ in range(int(3.3 / 0.1) + int(13 / 0.1)):  # until the alarm
        frame, ts = p2_src.read()
        p2.step(frame, ts)
    assert p2.fall.state == ALARM and p2.alarm.active
    p2.request_fall_ack("web")
    frame, ts = p2_src.read()
    p2.step(frame, ts)
    assert p2.fall.state == OK and not p2.alarm.active
    p2.close()


def _ntfy_server():
    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((dict(self.headers), body.decode()))
            self.send_response(200)
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, got


def _pipeline(tmp_path, script, depth=2.5, **over):
    from test_follow import FakeHead
    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline

    class Speaker(FakeSpeaker):
        def say_blocking(self, text):
            self.said.append(text)

    base = {
        "logging": {"sessions_dir": str(tmp_path / "s")},
        "analysis": {"calibration_file": str(tmp_path / "c")},
        "coach": {"greet": False, "min_interval_s": 3},
        "alarm": {"siren": False, "repeat_s": 0.2},
        "thresholds": {"hold_s": 2},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v}
    cfg = load_config(overrides=base)
    src = FakeSource(script)
    src.last_depth = np.full((640, 480), depth, np.float32)
    sp = Speaker()
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=FakeHead(),
        speaker=sp,
        llm=FakeLLM(),
        window=False,
        stream=False,
        vr=False,
    )
    return p, sp


def test_no_posture_tips_while_lying_on_the_floor(tmp_path):
    # lying on the floor the whole time: trunk_flexion would normally be coached after 2 s
    p, sp = _pipeline(tmp_path, [(34, {"trunk_flex": 88, "drop": 0.6})])
    session = p.run()
    events = [json.loads(x) for x in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not [e for e in events if e["kind"] in ("issue", "praise", "break")]
    assert [e["state"] for e in events if e["kind"] == "fall"][:1] == [CHECK]  # long-lying check still speaks
    assert any("okay" in s.lower() for s in sp.said)


def test_fall_recovered_sends_check_and_resolved_notifications(tmp_path):
    srv, got = _ntfy_server()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    p, sp = _pipeline(
        tmp_path,
        [(3, {}), (0.3, {"trunk_flex": 50, "drop": 0.3}), (4, {"trunk_flex": 88, "drop": 0.6}), (5, {})],
        alarm={"ntfy_topic": "t", "ntfy_server": base},
    )
    p.run()
    import time as _t

    _t.sleep(0.5)
    srv.shutdown()
    titles = [h["Title"] for h, _ in got]
    assert titles == ["Stretch: possible fall", "Stretch: fall alert cleared"]
    assert got[0][0]["Priority"] == "high"
    assert not any("ALARM" in t for t in titles)


def test_coach_drops_inflight_tip_when_muted():
    from stretch_posture_coach.coach import Coach

    class Sp:
        busy = False

        def __init__(self):
            self.said = []

        def say(self, t):
            self.said.append(t)

    sp = Sp()
    c = Coach(load_config(), sp, None)
    c._t = 10.0
    c.mute(10.0, 30)
    c._speak("Sit up straight!", {"kind": "issue"})
    c._speak("¿Estás bien?", {"kind": "announce"})
    assert sp.said == ["¿Estás bien?"]
    c._t = 41.0
    c._speak("Sit up straight!", {"kind": "issue"})
    assert sp.said[-1] == "Sit up straight!"
