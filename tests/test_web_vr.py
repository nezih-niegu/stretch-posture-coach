"""Web viewer report routes + VR server (HTTPS, WebSocket protocol, look control) end to end."""

import asyncio
import json
import ssl
import struct
import time
import urllib.request

import numpy as np
import pytest
from synth import skeleton

from stretch_posture_coach.camera import Intrinsics
from stretch_posture_coach.config import load_config
from stretch_posture_coach.stream import FrameStreamer
from stretch_posture_coach.vr import T_RGBD, VRServer, landmarks_3d, unpack


def make_session(root, name="20261114_101500"):
    d = root / name
    d.mkdir(parents=True)
    rows = ["t,wall_time,present,posture,rula,reba,neck_flexion,trunk_flexion,distance_m"]
    for i in range(50):
        rows.append(f"{i * 0.2},{1763130000 + i * 0.2},1,sitting,{3 + (i > 25)},{4},{10 + i % 7},{12},{2.4}")
    (d / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (d / "events.jsonl").write_text(
        json.dumps(
            {
                "kind": "issue",
                "issue": "neck_flexion",
                "source": "llm",
                "text": "¡Ey, compa!",
                "wall_time": 1763130005,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return d


def get(url, ctx=None):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_viewer_report_routes(tmp_path):
    sess = make_session(tmp_path)
    s = FrameStreamer(port=0, sessions_dir=tmp_path, vr_port=8443)
    base = f"http://127.0.0.1:{s.port}"
    try:
        code, _, body = get(base + "/")
        assert code == 200 and b"VR view" in body and b"/sessions" in body
        code, _, body = get(base + "/sessions")
        assert code == 200 and sess.name.encode() in body
        code, ctype, body = get(base + f"/sessions/{sess.name}/")
        assert (
            code == 200
            and "html" in ctype
            and b"What Stretch said" in body
            and "¡Ey, compa!".encode() in body
        )
        assert get(base + f"/sessions/{sess.name}/summary.json")[0] == 200
        assert get(base + "/sessions/../etc/")[0] == 404
        assert get(base + "/sessions/nope/")[0] == 404
        assert json.loads(get(base + "/state")[2])["state"] == "running"
        s.set_ended(sess / "report.html")
        st = json.loads(get(base + "/state")[2])
        assert st == {"state": "ended", "report": "/report"}
        code, _, body = get(base + "/")  # redirected to the report
        assert code == 200 and b"Posture session" in body
    finally:
        s.close()


def test_landmarks_3d_backprojection():
    intr = Intrinsics.from_fov(480, 640, 42)
    p = skeleton()
    img = np.tile([[0.5, 0.5]], (33, 1))
    img[11] = [0.75, 0.5]  # right of center
    depth = np.full((640, 480), 2.0, np.float32)
    pts = landmarks_3d(img, p.visibility, depth, intr)
    assert pts[0][2] == pytest.approx(2.0) and abs(pts[0][0]) < 0.01
    assert pts[11][0] > 0.3  # 2 m away, a quarter image to the right


@pytest.fixture
def vr(tmp_path):
    cfg = load_config()["vr"] | {
        "port": 0,
        "host": "127.0.0.1",
        "fps": 30,
        "gripper_camera": False,
        "nav_camera": False,
    }
    srv = VRServer(cfg, tmp_path, tmp_path / "certs", load_config()["thresholds"])
    yield srv
    srv.close()


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # self-signed: the Quest user accepts it once
    return ctx


def test_vr_static_files_over_https(vr):
    base = f"https://127.0.0.1:{vr.port}"
    code, ctype, body = get(base + "/", _ctx())
    assert code == 200 and b"Enter VR" in body and b"importmap" in body
    code, ctype, body = get(base + "/vendor/three.module.js", _ctx())
    assert code == 200 and "javascript" in ctype and len(body) > 100_000
    assert get(base + "/vendor/three.core.js", _ctx())[0] == 200
    assert get(base + "/vr.js", _ctx())[0] == 200
    assert get(base + "/vendor/..%2Fvr.py", _ctx())[0] == 404
    assert get(base + "/report", _ctx())[0] == 404  # no sessions yet


def test_vr_websocket_protocol(vr):
    from websockets.asyncio.client import connect

    intr = Intrinsics.from_fov(480, 640, 42)
    frame = np.zeros((640, 480, 3), np.uint8)
    frame[:, :240] = (0, 0, 255)
    depth = np.full((640, 480), 2.5, np.float32)
    pose = skeleton()
    pose.image = np.column_stack([0.5 + pose.world[:, 0] * 0.3, 0.5 + pose.world[:, 1] * 0.3])
    metrics = {"neck_flexion": 30, "trunk_flexion": 5}

    async def scenario():
        async with connect(f"wss://127.0.0.1:{vr.port}/ws", ssl=_ctx(), max_size=None) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert first == {"type": "state", "state": "running", "report": None}
            for _ in range(20):  # publish until the encoder picks it up
                vr.publish(
                    frame,
                    depth,
                    intr,
                    yaw=0.3,
                    pitch=-0.2,
                    pose=pose,
                    metrics=metrics,
                    hud={"present": True, "posture": "sitting"},
                )
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.5)
                    break
                except TimeoutError:
                    continue
            t, meta, rest = unpack(msg)
            assert t == T_RGBD
            assert (meta["w"], meta["h"], meta["dw"], meta["dh"]) == (480, 640, 120, 160)
            assert meta["yaw"] == pytest.approx(0.3) and meta["pitch"] == pytest.approx(-0.2)
            assert meta["has_depth"] and meta["hud"]["posture"] == "sitting"
            assert meta["levels"]["neck"] == 2 and meta["levels"]["trunk"] == 0
            assert sum(p is not None for p in meta["skeleton"]) >= 10
            (jl,) = struct.unpack_from("<I", rest, 0)
            jpeg, dbytes = rest[4 : 4 + jl], rest[4 + jl :]
            assert jpeg[:2] == b"\xff\xd8"
            d = np.frombuffer(dbytes, "<u2")
            assert d.size == 120 * 160 and np.all(d == 2500)

            await ws.send(json.dumps({"type": "look", "yaw": 0.5, "pitch": -0.1, "active": True}))
            await asyncio.sleep(0.2)
            assert vr.look_target() == (pytest.approx(0.5), pytest.approx(-0.1))
            await ws.send(json.dumps({"type": "look", "yaw": 0.5, "pitch": -0.1, "active": False}))
            await asyncio.sleep(0.2)
            assert vr.look_target() is None

            vr.set_ended(None)
            while True:
                m = await asyncio.wait_for(ws.recv(), 5)
                if isinstance(m, str):
                    assert json.loads(m) == {"type": "state", "state": "ended", "report": "/sessions"}
                    break

    asyncio.run(scenario())
    time.sleep(0.2)
    assert vr.n_clients == 0


def test_pipeline_vr_look_control_pauses_follow(tmp_path):
    from test_follow import FakeHead
    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline

    cfg = load_config(
        overrides={
            "logging": {"enabled": False},
            "analysis": {"calibration_file": str(tmp_path / "c.json")},
            "follow": {"enabled": True},
            "coach": {"greet": False},
            "vr": {
                "port": 0,
                "host": "127.0.0.1",
                "certs_dir": str(tmp_path / "certs"),
                "gripper_camera": False,
                "nav_camera": False,
            },
        }
    )
    src = FakeSource([(3, {})])
    src.last_depth = np.full((640, 480), 4.0, np.float32)  # far away: follow would approach...
    head = FakeHead()
    looks = []
    head.look = lambda pan, tilt, now: looks.append((pan, tilt))
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=head,
        speaker=FakeSpeaker(),
        llm=FakeLLM(),
        window=False,
        stream=False,
        log_session=False,
    )
    assert p.vr is not None
    p.vr._look = {"yaw": 0.4, "pitch": -0.2, "active": True, "t": time.monotonic() + 1e6}  # VR viewer looking
    p.run()
    assert looks and looks[-1] == (pytest.approx(0.4), pytest.approx(-0.2))
    assert not head.translations and not head.rotations  # ...but never drives while someone is in VR
    assert p.follow_state.action == "paused"


def test_vr_flow_control_bounds_latency(vr):
    """A headset that stops acknowledging gets at most MAX_IN_FLIGHT frames; acks resume the stream."""
    from websockets.asyncio.client import connect

    from stretch_posture_coach.vr import MAX_IN_FLIGHT

    intr = Intrinsics.from_fov(48, 64, 42)
    frame = np.zeros((64, 48, 3), np.uint8)
    depth = np.full((64, 48), 2.0, np.float32)

    async def count_frames(ws, seconds):
        n, end = 0, asyncio.get_event_loop().time() + seconds
        while True:
            remaining = end - asyncio.get_event_loop().time()
            if remaining <= 0:
                return n
            try:
                m = await asyncio.wait_for(ws.recv(), remaining)
            except TimeoutError:
                return n
            if isinstance(m, bytes):
                n += 1

    async def scenario():
        stop = asyncio.Event()

        async def publisher():
            while not stop.is_set():
                vr.publish(frame, depth, intr, 0.0, 0.0)
                await asyncio.sleep(1 / 60)

        async with connect(f"wss://127.0.0.1:{vr.port}/ws", ssl=_ctx(), max_size=None) as ws:
            pub = asyncio.ensure_future(publisher())
            unacked = await count_frames(ws, 1.5)
            assert 1 <= unacked <= MAX_IN_FLIGHT[T_RGBD]
            got = 0
            for _ in range(10):  # now behave like the real client: ack every frame
                await ws.send(json.dumps({"type": "ack", "t": T_RGBD}))
                got += await count_frames(ws, 0.3)
            assert got >= 8
            stop.set()
            await pub

    asyncio.run(scenario())


def test_session_end_serves_report_on_both_servers(tmp_path):
    from test_follow import FakeHead
    from test_pipeline import FakeEstimator, FakeLLM, FakeSource, FakeSpeaker

    from stretch_posture_coach.pipeline import Pipeline
    from stretch_posture_coach.report import write_report

    cfg = load_config(
        overrides={
            "logging": {"sessions_dir": str(tmp_path / "sessions")},
            "analysis": {"calibration_file": str(tmp_path / "c.json")},
            "coach": {"greet": False},
            "display": {"stream_port": 0, "stream_host": "127.0.0.1"},
            "vr": {
                "port": 0,
                "host": "127.0.0.1",
                "certs_dir": str(tmp_path / "certs"),
                "gripper_camera": False,
                "nav_camera": False,
            },
        }
    )
    src = FakeSource([(4, {"trunk_flex": 30})])
    p = Pipeline(
        cfg,
        source=src,
        estimator=FakeEstimator(src),
        head=FakeHead(),
        speaker=FakeSpeaker(),
        llm=FakeLLM(),
        window=False,
        stream=True,
        keep_servers=True,
    )
    session = p.run()
    assert p.streamer is not None and p.vr is not None  # still up after the session
    report = write_report(session)
    p.show_report(report)
    try:
        st = json.loads(get(f"http://127.0.0.1:{p.streamer.port}/state")[2])
        assert st == {"state": "ended", "report": "/report"}
        code, _, body = get(f"http://127.0.0.1:{p.streamer.port}/report")
        assert code == 200 and session.name.encode() in body
        code, _, body = get(f"https://127.0.0.1:{p.vr.port}/report", _ctx())
        assert code == 200 and session.name.encode() in body
        code, _, body = get(f"http://127.0.0.1:{p.streamer.port}/sessions")
        assert session.name.encode() in body
    finally:
        p.close_servers()
