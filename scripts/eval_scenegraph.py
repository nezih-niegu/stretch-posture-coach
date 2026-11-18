"""Scene-graph segmentation accuracy on synthetic furnished rooms with ground truth.

    uv run python scripts/eval_scenegraph.py [n_rooms]

Two settings: nominal (1 cm noise, 20 % dropout, chairs 5-10 cm from the desk, near-axis-aligned) and hard
(2 cm noise, 40 % dropout, 3 % stray points, chairs touching the desk, any orientation).

A ground-truth object counts as found when a predicted object's footprint overlaps it with IoU >= 0.3; it is
labelled correctly when that object's label matches. Also checks the person-support relation (sitting on a chair,
lying on a sofa/bed, lying on the floor) with hip heights drawn around each surface."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from room_synth import HARD, gt_cells, random_room  # noqa: E402

from stretch_posture_coach.scenegraph import SceneGraph  # noqa: E402


def cells_of(node):
    ix, iy = node.cells >> 32, (node.cells & 0xFFFFFFFF).astype(np.int32)
    return set(zip(ix.tolist(), iy.tolist(), strict=True))


def evaluate(n_rooms: int = 30, hard: bool = False) -> dict:
    found, correct, total = Counter(), Counter(), Counter()
    conf = Counter()
    pred_total = pred_matched = 0
    sup_ok = sup_n = 0
    for seed in range(n_rooms):
        P, gts = random_room(seed, **(HARD if hard else {}))
        g = SceneGraph()
        g.rebuild_static(P)
        preds = [o for o in g.objects if o.label not in ("wall",)]
        pc = [cells_of(o) for o in preds]
        used = set()
        for gt in gts:
            total[gt["label"]] += 1
            G = gt_cells(gt)
            best, bi = 0.0, None
            for i, C in enumerate(pc):
                iou = len(G & C) / max(1, len(G | C))
                if iou > best:
                    best, bi = iou, i
            if best >= 0.3:
                found[gt["label"]] += 1
                used.add(bi)
                conf[(gt["label"], preds[bi].label)] += 1
                if preds[bi].label == gt["label"]:
                    correct[gt["label"]] += 1
        big = [i for i, o in enumerate(preds) if o.label not in ("object", "tall_object")]
        pred_total += len(big)
        pred_matched += sum(1 for i in big if i in used)
        # support relation: a person placed on each seat/bed/sofa and on open floor
        rng = np.random.default_rng(1000 + seed)
        for gt in gts:
            if gt["label"] in ("chair", "sofa", "bed"):
                lying = gt["label"] != "chair"
                hip = gt["top"] + rng.uniform(0.0, 0.2)
                ctx = g.context(np.array([gt["cx"], gt["cy"], hip + 0.3]), hip, None, lying=lying)
                sup_n += 1
                sup_ok += ctx["support"] == gt["label"]
        for _ in range(3):  # lying on the floor somewhere free
            for _ in range(100):
                x, y = rng.uniform(0.5, 5.5), rng.uniform(0.5, 4.5)
                if g.object_at(x, y, margin=True) is None:
                    break
            ctx = g.context(np.array([x, y, 0.3]), 0.15, None, lying=True)
            sup_n += 1
            sup_ok += ctx["support"] == "floor"
    res = {
        "rooms": n_rooms,
        "setting": "hard" if hard else "nominal",
        "objects": sum(total.values()),
        "recall": sum(found.values()) / max(1, sum(total.values())),
        "label_acc_found": sum(correct.values()) / max(1, sum(found.values())),
        "per_class": {k: {"n": total[k], "found": found[k], "correct": correct[k]} for k in sorted(total)},
        "precision": pred_matched / max(1, pred_total),
        "support_acc": sup_ok / max(1, sup_n),
        "support_n": sup_n,
        "confusions": {f"{a}->{b}": n for (a, b), n in conf.items() if a != b},
    }
    return res


if __name__ == "__main__":
    import json

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for hard in (False, True):
        print(json.dumps(evaluate(n, hard), indent=1))
