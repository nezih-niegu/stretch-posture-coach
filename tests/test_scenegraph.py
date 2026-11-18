"""Scene graph: segmentation, relations, fall support, prompt context, offline build and live world hooks."""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_synth import HARD, furniture, gt_cells, random_room, walls  # noqa: E402

from stretch_posture_coach.fall import CHECK, OK, FallDetector, FallFeatures  # noqa: E402
from stretch_posture_coach.scenegraph import (  # noqa: E402
    SceneGraph,
    build_scene_graph,
    describe_context,
    read_points_bin,
    segment_objects,
)


def _room(*items):
    P = [walls()]
    gts = []
    for kind, x, y, yaw in items:
        p, g = furniture(kind, x, y, yaw)
        P.append(p)
        gts.append(g)
    X = np.concatenate(P)
    k = np.unique(np.floor(X / 0.05).astype(np.int64), axis=0)
    return (k + 0.5) * 0.05, gts


def _cells(node):
    ix, iy = node.cells >> 32, (node.cells & 0xFFFFFFFF).astype(np.int32)
    return set(zip(ix.tolist(), iy.tolist(), strict=True))


def _match(graph, gt):
    G = gt_cells(gt)
    best = max(graph.objects, key=lambda o: len(G & _cells(o)) / max(1, len(G | _cells(o))))
    return best, len(G & _cells(best)) / max(1, len(G | _cells(best)))


def test_segments_each_furniture_class():
    P, gts = _room(
        ("table", 2.0, 2.0, 0.0),
        ("chair", 2.0, 1.2, math.pi / 2),
        ("sofa", 4.5, 3.8, 0.0),
        ("bed", 4.4, 1.5, 0.0),
        ("cabinet", 0.6, 4.3, 0.0),
        ("box", 1.0, 0.8, 0.3),
    )
    g = SceneGraph()
    g.rebuild_static(P)
    for gt in gts:
        o, iou = _match(g, gt)
        assert iou >= 0.4, (gt["kind"], iou)
        assert o.label == gt["label"], (gt["kind"], o.label)
    t, _ = _match(g, gts[0])
    assert 0.70 <= t.top <= 0.80  # table surface height
    assert g.summary().get("wall", 0) >= 1


def test_chair_pushed_against_table_stays_separate():
    # chair 3 cm from the table edge, backrest at table height: must not merge into the table
    P, gts = _room(("table", 3.0, 2.5, 0.0), ("chair", 3.0, 2.5 - 0.35 - 0.03 - 0.225, math.pi / 2))
    g = SceneGraph()
    g.rebuild_static(P)
    labels = sorted(_match(g, gt)[0].label for gt in gts)
    assert labels == ["chair", "table"]
    rels = {(e.src.split("_")[0], e.rel, e.dst.split("_")[0]) for e in g.edges()}
    assert ("chair", "at", "table") in rels


@pytest.mark.parametrize("hard", [False, True])
def test_random_rooms_recall(hard):
    found = total = correct = 0
    for seed in range(6):
        P, gts = random_room(seed, **(HARD if hard else {}))
        g = SceneGraph()
        g.rebuild_static(P)
        for gt in gts:
            o, iou = _match(g, gt)
            total += 1
            if iou >= 0.3:
                found += 1
                correct += o.label == gt["label"]
    assert found / total >= 0.9
    assert correct / max(found, 1) >= 0.9


def test_person_relations_and_description():
    P, gts = _room(("table", 2.0, 2.0, 0.0), ("chair", 2.0, 1.4, math.pi / 2), ("sofa", 4.5, 3.8, 0.0))
    g = SceneGraph()
    g.rebuild_static(P)
    chair = gts[1]
    ctx = g.update_person(np.array([chair["cx"], chair["cy"], 0.9]), hip_height=0.5, posture="sitting")
    assert ctx["support"] == "chair" and ctx["relation"] == "sitting_on"
    assert ctx["at"] and ctx["at"].startswith("table")
    assert "sitting on chair" in ctx["text"] and "at table" in ctx["text"]
    rels = {(e.src, e.rel, e.dst.split("_")[0]) for e in g.edges()}
    assert ("person_1", "sitting_on", "chair") in rels
    assert ("person_1", "at", "table") in rels
    # lying on the sofa vs on the floor
    s = gts[2]
    assert g.context(np.array([s["cx"], s["cy"], 0.6]), 0.5, None, lying=True)["support"] == "sofa"
    floor = g.context(np.array([1.0, 4.0, 0.3]), 0.15, None, lying=True)
    assert floor["support"] == "floor" and floor["relation"] == "lying_on"
    assert "lying on the floor" in describe_context(floor)
    assert "sofa" in g.describe()


def test_occlusion_by_table():
    P, gts = _room(("table", 3.0, 2.5, math.pi / 2))
    g = SceneGraph()
    g.rebuild_static(P)
    cam = np.array([1.0, 2.5, 1.3])
    # someone down on the floor just behind the table, seen from the robot's camera
    ctx = g.context(np.array([3.7, 2.5, 0.4]), 0.3, cam, lying=True)
    assert ctx["occluder"] == "table"
    # standing person in front of the table: not hidden
    assert g.context(np.array([2.0, 2.5, 1.0]), 0.9, cam)["occluder"] is None


def test_detector_votes_override_geometric_label():
    P, gts = _room(("sofa", 3.0, 2.5, 0.0))
    g = SceneGraph({"min_votes": 3})
    g.rebuild_static(P)
    o, _ = _match(g, gts[0])
    assert o.label == "sofa"
    for _ in range(4):
        g.add_detection("bed", [3.0, 2.5, 0.4])
    o, _ = _match(g, gts[0])
    assert o.label == "bed" and o.id.startswith("bed_")
    g.rebuild_static(P)  # votes survive a re-segmentation
    assert _match(g, gts[0])[0].label == "bed"
    g.add_detection("toothbrush", [3.0, 2.5, 0.4])  # unknown classes are ignored


def test_events_attach_to_nearest_object():
    P, gts = _room(("table", 2.0, 2.0, 0.0))
    g = SceneGraph()
    g.rebuild_static(P)
    ev = g.add_event("fall_check", [2.0, 2.6, 0.9], 1.0, "Possible fall")
    assert ev.attrs["at"].startswith("table")
    far = g.add_event("fall_check", [5.0, 4.0, 0.9], 2.0)
    assert far.attrs["at"] == "floor"
    js = g.to_json()
    assert any(e["rel"] == "at" and e["src"] == ev.id for e in js["edges"])
    json.dumps(js)  # serializable


def _ff(t, incl, hip, support=None):
    return FallFeatures(t=t, present=True, inclination=incl, hip_height=hip, support=support)


def test_long_lying_check_ignores_sofa_and_bed():
    det = FallDetector({"long_lying_s": 5})
    for i in range(80):  # lying 8 s on a sofa with low hips: resting, no check
        ch = det.update(_ff(i * 0.1, 80, 0.4, "sofa"))
        assert ch is None or ch[0] != CHECK
    assert det.state == OK
    det2 = FallDetector({"long_lying_s": 5})
    states = [det2.update(_ff(i * 0.1, 80, 0.3, "floor")) for i in range(80)]
    assert any(s and s[0] == CHECK for s in states)
    det3 = FallDetector({"long_lying_s": 5})  # without a scene graph: unchanged behaviour
    states = [det3.update(_ff(i * 0.1, 80, 0.3, None)) for i in range(80)]
    assert any(s and s[0] == CHECK for s in states)


def test_prompt_carries_scene_context():
    from stretch_posture_coach.assessment import Assessment  # noqa: F401
    from stretch_posture_coach.coach import build_prompt
    from stretch_posture_coach.config import load_config

    cfg = load_config()

    class A:
        metrics = {"posture": "sitting"}
        static = False
        static_for_s = 0
        rula = {"score": 4, "risk": "investigate"}
        reba = {"score": 5, "risk": "medium"}

    msgs = build_prompt("break", A(), [], [], cfg, 0.0, scene="The person is sitting on chair 2 at table 1.")
    assert "Scene: The person is sitting on chair 2 at table 1." in msgs[-1]["content"]
    msgs = build_prompt("break", A(), [], [], cfg, 0.0)
    assert "Scene:" not in msgs[-1]["content"]


def _write_session(tmp_path: Path):
    P, gts = _room(("table", 2.0, 2.0, 0.0), ("chair", 2.0, 1.4, math.pi / 2))
    sc = tmp_path / "scene"
    sc.mkdir(parents=True)
    n = len(P)
    cam = np.array([[0.5, 0.5, 1.3]], np.float32)
    trk = np.zeros((0, 3), np.float32)
    buf = struct.pack("<I", n) + P.astype("<f4").tobytes() + np.full((n, 3), 128, np.uint8).tobytes()
    buf += b"\0" * ((-n * 3) % 4)
    buf += struct.pack("<I", 1) + cam.astype("<f4").tobytes() + struct.pack("<I", 0) + trk.tobytes()
    (sc / "points.bin").write_bytes(buf)
    # camera at (0.5, 0.5, 1.3) looking along +x (optical z -> map x)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = [0.5, 0.5, 1.3]
    frames = []
    for i in range(10):
        # person sitting on the chair: chair centre (2.0, 1.4) -> optical coords
        pm = np.array([2.0, 1.4, 0.9])
        pc = np.linalg.inv(T) @ np.array([*pm, 1.0])
        frames.append(
            {
                "i": i,
                "wall": 100.0 + i,
                "person_cam": pc[:3].tolist(),
                "T_map_cam": T.ravel().tolist(),
                "posture": "sitting",
                "intr": [600, 600, 240, 320, 480, 640],
            }
        )
    (sc / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    (sc / "overlay.json").write_text(
        json.dumps(
            {
                "captions": [
                    {
                        "kind": "fall_check",
                        "title": "Possible fall",
                        "text": "x",
                        "t": 104.0,
                        "pos": [2.0, 2.7, 0.9],
                    }
                ]
            }
        )
    )
    rows = ["t,wall_time,present,reba"] + [f"{i},{100.0 + i},1,{4 + (i % 2)}" for i in range(10)]
    (tmp_path / "metrics.csv").write_text("\n".join(rows))
    return P


def test_build_scene_graph_offline(tmp_path):
    P = _write_session(tmp_path)
    Q, C, cam, trk = read_points_bin(tmp_path / "scene" / "points.bin")
    assert np.allclose(Q, P, atol=1e-5) and len(cam) == 1
    g = build_scene_graph(tmp_path)
    assert g is not None and (tmp_path / "scene" / "scene_graph.json").exists()
    assert g["summary"].get("table") == 1 and g["summary"].get("chair") == 1
    top = next(iter(g["posture_by_object"]))
    assert top.startswith("chair") and g["posture_by_object"][top]["samples"] == 10
    assert abs(g["posture_by_object"][top]["reba_mean"] - 4.5) < 1e-6
    assert g["fall_events"][0]["at"].startswith("table")
    assert "sitting on chair" in g["description"] or "chair" in g["description"]


def test_build_scene_graph_needs_points(tmp_path):
    assert build_scene_graph(tmp_path) is None


def test_live_world_builds_graph_and_names_events():
    from stretch_posture_coach.config import load_config
    from stretch_posture_coach.world import LiveWorld

    cfg = load_config()
    cfg["lidar"]["enabled"] = False
    cfg["scene_graph"]["detector"] = False
    w = LiveWorld(cfg)
    try:
        P, gts = _room(("table", 2.0, 2.0, 0.0), ("chair", 2.0, 1.4, math.pi / 2))
        w.map.integrate(P, np.full((len(P), 3), 128, np.float32), None, None, None, 0)
        w.map.integrate(P, np.full((len(P), 3), 128, np.float32), None, None, None, 0)  # min_hits = 2
        w._map_changed = True
        w._maybe_rebuild_graph(force=True)
        assert w.graph.summary().get("table") == 1
        w.last["person"] = np.array([2.0, 2.7, 0.9])
        c = w.event({"kind": "fall", "state": "check", "reason": "test", "text": ""})
        assert c is not None and "near table 1" in c["text"]
        assert "near table" in w.fall_where() or "on table" in w.fall_where()
        msg = w.scene_graph_msg()
        assert msg["type"] == "scene_graph" and any(b["label"] == "table" for b in msg["boxes"])
        assert w.status()["scene_graph"].get("chair") == 1
    finally:
        w.close()


def test_segment_empty_and_tiny_inputs():
    nodes, room = segment_objects(np.zeros((0, 3)))
    assert nodes == []
    nodes, _ = segment_objects(np.array([[1.0, 1.0, 0.5]]))
    assert nodes == []


def test_ceiling_does_not_hide_furniture():
    from room_synth import _box_surface

    P, gts = _room(("table", 2.0, 2.0, 0.0), ("chair", 4.0, 3.0, 0.0))
    ceiling = _box_surface(3.0, 2.5, 0.0, 6.0, 5.0, 2.4, 2.45, top=True, sides=False)
    lamp = _box_surface(2.0, 2.0, 0.0, 0.5, 0.5, 2.0, 2.2, top=True, sides=True)
    X = np.concatenate([P, ceiling, lamp])
    k = np.unique(np.floor(X / 0.05).astype(np.int64), axis=0)
    g = SceneGraph()
    g.rebuild_static((k + 0.5) * 0.05)  # no ceiling hint: estimated from the points
    assert sorted(_match(g, gt)[0].label for gt in gts) == ["chair", "table"]
