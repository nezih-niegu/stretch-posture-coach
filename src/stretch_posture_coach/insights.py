"""What to caption in the SLAM view and how to color the posture heat map (shared by the live VR view and the
session report).

Captions: fall checks/alarms, coaching moments, and *measured* improvements (the coached metric before vs
after the tip), each pinned to where the person was in the map at that moment.

Heat map: the floor under the person, in 0.25 m cells, colored by the mean REBA score while they were there -
a diverging blue <-> red ramp with a neutral gray midpoint at REBA 5 (middle of "medium risk"):
blue = better posture there, red = worse.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np

from .assessment import RULES

ISSUE_LABEL = {r.id: r.label for r in RULES}
# coached issue -> (metric column, unit, use absolute value)
ISSUE_METRIC = {
    "neck_flexion": ("neck_flexion", "°", False),
    "neck_extension": ("neck_flexion", "°", True),
    "neck_side_bend": ("neck_side_bend", "°", True),
    "neck_twist": ("neck_twist", "°", True),
    "trunk_flexion": ("trunk_flexion", "°", False),
    "trunk_extension": ("trunk_flexion", "°", True),
    "trunk_side_bend": ("trunk_side_bend", "°", True),
    "trunk_twist": ("trunk_twist", "°", True),
    "shoulder_tilt": ("shoulder_tilt_cm", " cm", True),
    "arms_raised": ("max_upper_arm_elevation", "°", False),
    "lumbar_load": ("l5s1_compression_n", " N", False),
}

# status palette (fixed): an icon + label always accompanies the color
LEVEL_COLORS = {"critical": "#d03b3b", "warning": "#fab219", "serious": "#ec835a", "good": "#0ca30c"}
LEVEL_ICON = {"critical": "!", "warning": "▲", "serious": "▲", "good": "✓"}

# diverging blue <-> red with a gray midpoint (reference palette), per surface
DIVERGING = {
    "dark": ("#3987e5", "#383835", "#e66767"),
    "light": ("#2a78d6", "#f0efec", "#e34948"),
}
REBA_MID, REBA_SPAN = 5.0, 4.0  # REBA 1 -> -1 (blue), 5 -> 0 (gray), 9+ -> +1 (red)


# ------------------------------------------------------------------------------------------ colors
def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def _to_oklab(rgb):
    lms = _M1 @ _srgb_to_lin(rgb)
    return _M2 @ np.cbrt(lms)


def _from_oklab(lab):
    lms = np.linalg.inv(_M2) @ lab
    return _lin_to_srgb(np.linalg.inv(_M1) @ (lms**3))


def diverging_rgb(v: float, mode: str = "dark") -> tuple[int, int, int]:
    """v in [-1, 1] -> RGB (0-255), interpolated in OKLab between the poles and the gray midpoint."""
    lo, mid, hi = (_to_oklab(_hex_to_rgb(h)) for h in DIVERGING[mode])
    v = float(np.clip(v, -1, 1))
    lab = mid + (lo - mid) * (-v) if v < 0 else mid + (hi - mid) * v
    return tuple(int(round(x * 255)) for x in _from_oklab(lab))


def reba_to_v(reba: float) -> float:
    return float(np.clip((reba - REBA_MID) / REBA_SPAN, -1, 1))


def rgb_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# ------------------------------------------------------------------------------------------ heat grid
class HeatGrid:
    """Mean REBA per floor cell where the person was (map frame)."""

    def __init__(self, cell: float = 0.25):
        self.cell = cell
        self.sum: dict[tuple[int, int], float] = {}
        self.n: dict[tuple[int, int], int] = {}
        self.version = 0

    def add(self, x: float, y: float, reba: float | None) -> None:
        if reba is None or not math.isfinite(x) or not math.isfinite(y):
            return
        k = (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))
        self.sum[k] = self.sum.get(k, 0.0) + float(reba)
        self.n[k] = self.n.get(k, 0) + 1
        self.version += 1

    def cells(self, mode: str = "dark", min_n: int = 2) -> list[dict]:
        out = []
        counts, sums = dict(self.n), dict(self.sum)  # snapshot: the pipeline thread keeps adding
        for k, n in counts.items():
            if n < min_n or k not in sums:
                continue
            m = sums[k] / n
            out.append(
                {
                    "x": (k[0] + 0.5) * self.cell,
                    "y": (k[1] + 0.5) * self.cell,
                    "reba": round(m, 2),
                    "n": n,
                    "color": rgb_hex(diverging_rgb(reba_to_v(m), mode)),
                }
            )
        return out

    def to_json(self, mode: str = "dark") -> dict:
        lo, mid, hi = DIVERGING[mode]
        return {
            "cell": self.cell,
            "cells": self.cells(mode),
            "legend": {
                "low": "better posture (REBA 1)",
                "mid": f"REBA {REBA_MID:.0f}",
                "high": f"worse posture (REBA {REBA_MID + REBA_SPAN:.0f}+)",
                "colors": [lo, mid, hi],
                "ramp": [rgb_hex(diverging_rgb(v, mode)) for v in np.linspace(-1, 1, 9)],
            },
        }


# ------------------------------------------------------------------------------------------ captions
def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "-"
    return f"{v:.0f}{unit}"


def caption_from_event(
    ev: dict, pos, metrics: dict | None = None, coached: dict | None = None
) -> dict | None:
    """Live caption for an event (pos = map-frame [x, y, z] of the person, or None).

    ``coached`` maps issue id -> metric value at the time it was coached (for "improved" captions)."""
    kind = ev.get("kind")
    t = ev.get("wall_time")
    if kind == "fall":
        st = ev.get("state")
        if st == "check":
            c = ("warning", "Possible fall · checking", ev.get("reason", ""))
        elif st == "alarm":
            c = ("critical", "FALL ALARM", ev.get("reason", ""))
        elif st == "ok":
            c = ("good", "Recovered", ev.get("reason", ""))
        else:
            return None
    elif kind == "issue":
        iss = ev.get("issue", "")
        c = ("serious", f"Coached: {ISSUE_LABEL.get(iss, iss)}", ev.get("detail") or ev.get("text", ""))
    elif kind == "praise":
        iss = ev.get("issue", "")
        sub = "after the tip"
        mk = ISSUE_METRIC.get(iss)
        if mk and metrics is not None and coached and coached.get(iss) is not None:
            now = metrics.get(mk[0])
            if now is not None:
                now = abs(now) if mk[2] else now
                sub = f"{_fmt(coached[iss], mk[1])} → {_fmt(now, mk[1])} after the tip"
        c = ("good", f"Improved: {ISSUE_LABEL.get(iss, iss)}", sub)
    else:
        return None
    level, title, text = c
    return {
        "kind": kind if kind != "fall" else f"fall_{ev.get('state')}",
        "level": level,
        "icon": LEVEL_ICON[level],
        "color": LEVEL_COLORS[level],
        "title": title,
        "text": text,
        "t": t,
        "pos": None if pos is None else [round(float(v), 3) for v in pos],
    }


def load_track(session_dir: Path) -> list[dict]:
    p = Path(session_dir) / "track.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return sorted(out, key=lambda r: r.get("wall", 0))


def _nearest(track_walls: list[float], track: list[dict], wall: float, max_dt: float = 10.0) -> dict | None:
    if not track:
        return None
    i = bisect.bisect_left(track_walls, wall)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(track) and (best is None or abs(track_walls[j] - wall) < abs(best[0])):
            best = (track_walls[j] - wall, track[j])
    return best[1] if best and abs(best[0]) <= max_dt else None


def _metric_series(session_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p = Path(session_dir) / "metrics.csv"
    if not p.exists():
        return np.zeros(0), {}
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    wall = np.array([num(r.get("wall_time")) for r in rows])
    cols = {k: np.array([num(r.get(k)) for r in rows]) for k in (rows[0].keys() if rows else [])}
    return wall, cols


def _window_mean(wall, series, t0, t1, absval=False) -> float | None:
    m = (wall >= t0) & (wall <= t1) & np.isfinite(series)
    if m.sum() < 3:
        return None
    v = np.abs(series[m]) if absval else series[m]
    return float(np.mean(v))


def session_captions(
    session_dir: str | Path, track: list[dict] | None = None, max_items: int = 40
) -> list[dict]:
    """Captions for the report's SLAM view, placed with the registered person track."""
    session_dir = Path(session_dir)
    track = load_track(session_dir) if track is None else track
    walls = [r.get("wall", 0) for r in track]
    evp = session_dir / "events.jsonl"
    events = []
    if evp.exists():
        for line in evp.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    wall, cols = _metric_series(session_dir)
    th = {}
    cfgp = session_dir / "config.yaml"
    if cfgp.exists():
        import yaml

        th = (yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}).get("thresholds", {})
    out = []
    praised = set()
    for ev in events:
        if ev.get("kind") == "praise":
            praised.add(ev.get("issue"))
    for ev in events:
        kind, t = ev.get("kind"), ev.get("wall_time")
        if t is None:
            continue
        here = _nearest(walls, track, float(t))
        pos = None if here is None else [here["x"], here["y"], here.get("z", 1.0)]
        if kind == "fall" and ev.get("state") in ("check", "alarm", "ok"):
            c = caption_from_event(ev, pos)
        elif kind == "issue" and ev.get("issue") in ISSUE_METRIC and len(wall):
            iss = ev["issue"]
            col, unit, absval = ISSUE_METRIC[iss]
            series = cols.get(col)
            before = after = None
            if series is not None:
                before = _window_mean(wall, series, t - 12, t, absval)
                after = _window_mean(wall, series, t + 5, t + 40, absval)
            c = caption_from_event(ev, pos)
            thr = th.get(col) or th.get(iss)
            if before is not None and after is not None and before > 0:
                drop = (before - after) / before
                c["before"], c["after"] = round(before, 1), round(after, 1)
                if drop >= 0.25 or (thr and before > thr >= after):
                    # measured improvement after the tip: caption it where it happened (a little later)
                    later = _nearest(walls, track, float(t) + 20) or here
                    lp = None if later is None else [later["x"], later["y"], later.get("z", 1.0)]
                    imp = caption_from_event({"kind": "praise", "issue": iss, "wall_time": t + 20}, lp)
                    imp["text"] = f"{_fmt(before, unit)} → {_fmt(after, unit)} in the 30 s after the tip"
                    imp["before"], imp["after"] = c["before"], c["after"]
                    out.append(imp)
                    praised.discard(iss)
                else:
                    c["text"] = (
                        f"{c['text']} · no lasting change ({_fmt(before, unit)} → {_fmt(after, unit)})"
                    )
        elif kind == "praise" and ev.get("issue") in praised:
            c = caption_from_event(ev, pos)  # praised but no numeric window (short session)
            praised.discard(ev.get("issue"))
        else:
            continue
        if c is not None:
            out.append(c)
    out.sort(key=lambda c: c.get("t") or 0)
    order = {"critical": 0, "warning": 1, "good": 2, "serious": 3}
    if len(out) > max_items:  # keep alarms and improvements first
        out = sorted(out, key=lambda c: (order.get(c["level"], 9), c.get("t") or 0))[:max_items]
        out.sort(key=lambda c: c.get("t") or 0)
    for i, c in enumerate(out):
        c["id"] = i + 1
    return stack_captions(out)


def stack_captions(captions: list[dict], radius: float = 0.45, step: float = 0.32) -> list[dict]:
    """Give captions pinned at (nearly) the same spot increasing label heights ("lift") so they don't overlap."""
    placed: list[tuple[float, float]] = []
    for c in captions:
        if not c.get("pos"):
            continue
        x, y = c["pos"][0], c["pos"][1]
        n = sum(1 for (px, py) in placed if math.hypot(px - x, py - y) < radius)
        c["lift"] = round(n * step, 2)
        placed.append((x, y))
    return captions


def session_heat(track: list[dict], cell: float = 0.25, mode: str = "dark") -> dict:
    h = HeatGrid(cell)
    for r in track:
        h.add(r.get("x", math.nan), r.get("y", math.nan), r.get("reba"))
    return h.to_json(mode)
