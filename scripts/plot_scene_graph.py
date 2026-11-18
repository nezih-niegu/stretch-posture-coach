"""Top-down figure of a session's scene graph: map, segmented objects by class, person track, fall checks.

uv run python scripts/plot_scene_graph.py <session> out.png|out.pdf [--width 3.45]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from stretch_posture_coach.scenegraph import read_points_bin  # noqa: E402

COL = {
    "table": "#2a6fd6",
    "chair": "#d9622b",
    "sofa": "#7a4fd6",
    "bed": "#7a4fd6",
    "cabinet": "#19a979",
    "low_object": "#8a8a8a",
    "wall": "#222222",
}
NAMES = {
    "table": "table",
    "chair": "chair",
    "sofa": "sofa / bed",
    "cabinet": "cabinet",
    "low_object": "low object",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--width", type=float, default=3.45)
    ap.add_argument("--legend", choices=("right", "bottom", "none"), default="right")
    ap.add_argument("--font", type=float, default=7)
    a = ap.parse_args()
    sd = Path(a.session)
    P, C, cam, trk = read_points_bin(sd / "scene" / "points.bin")
    g = json.loads((sd / "scene" / "scene_graph.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.size": a.font})
    fig, ax = plt.subplots(figsize=(a.width, a.width * (1.1 if a.legend == "bottom" else 0.95)))
    m = (P[:, 2] > 0.08) & (P[:, 2] < 2.0)
    ax.scatter(P[m, 0], P[m, 1], s=0.15, c="#cfcfcf", lw=0, rasterized=True)
    seen = set()
    for n in g["nodes"]:
        if n["layer"] != "object" or n["label"] not in COL or n["label"] == "wall":
            continue
        (cx, cy, _), (ln, w, _h), yaw = n["center"], n["size"], n["yaw"]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        cr = np.array([[ln, w], [-ln, w], [-ln, -w], [ln, -w]]) / 2 @ R.T + [cx, cy]
        lab = "sofa" if n["label"] == "bed" else n["label"]
        ax.add_patch(Polygon(cr, fill=True, fc=COL[n["label"]], alpha=0.18, ec=COL[n["label"]], lw=0.6))
        seen.add(lab)
    if len(trk):
        ax.plot(trk[:, 0], trk[:, 1], "-", lw=0.5, color="#444444", alpha=0.6)
    ev = [e for e in g["nodes"] if e["layer"] == "event" and e["label"] in ("fall_check", "fall_alarm")]
    for e in ev:
        x, y = e["center"][:2]
        ax.plot(
            x,
            y,
            marker="^" if e["label"] == "fall_check" else "X",
            ms=4.5,
            mfc="#fab219" if e["label"] == "fall_check" else "#d03b3b",
            mec="black",
            mew=0.4,
            ls="none",
        )
    if len(cam):
        ax.plot(cam[:, 0], cam[:, 1], "-", lw=0.6, color="#3987e5")
    hs = [
        Line2D([], [], marker="s", ls="none", mfc=COL[k], mec=COL[k], alpha=0.6, ms=5, label=NAMES[k])
        for k in ("table", "chair", "sofa", "cabinet", "low_object")
        if k in seen
    ]
    hs += [
        Line2D([], [], color="#444444", lw=0.8, label="person track"),
        Line2D([], [], color="#3987e5", lw=0.8, label="robot camera"),
        Line2D(
            [], [], marker="^", ls="none", mfc="#fab219", mec="black", mew=0.4, ms=4.5, label="fall check"
        ),
        Line2D(
            [], [], marker="X", ls="none", mfc="#d03b3b", mec="black", mew=0.4, ms=4.5, label="fall alarm"
        ),
    ]
    if a.legend == "right":
        ax.legend(
            handles=hs, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=a.font - 0.5
        )
    elif a.legend == "bottom":
        ax.legend(
            handles=hs,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            frameon=False,
            fontsize=a.font - 0.5,
            columnspacing=0.6,
            handlelength=1.2,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
