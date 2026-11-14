"""Session report: summary statistics + charts (HTML with embedded PNGs) from a sessions/<stamp>/ folder."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
RULA_BANDS = [
    ("acceptable (1-2)", 1, 2, "good"),
    ("investigate (3-4)", 3, 4, "warning"),
    ("change soon (5-6)", 5, 6, "serious"),
    ("change now (7)", 7, 7, "critical"),
]
REBA_BANDS = [
    ("negligible/low (1-3)", 1, 3, "good"),
    ("medium (4-7)", 4, 7, "warning"),
    ("high (8-10)", 8, 10, "serious"),
    ("very high (11+)", 11, 15, "critical"),
]
TRACKED = [
    ("neck_flexion", "Neck flexion", "deg", "neck_flexion"),
    ("trunk_flexion", "Trunk flexion", "deg", "trunk_flexion"),
    ("trunk_side_bend", "Trunk side bend", "deg", "trunk_side_bend"),
    ("trunk_twist", "Trunk twist", "deg", "trunk_twist"),
    ("shoulder_tilt_cm", "Shoulder tilt", "cm", "shoulder_tilt_cm"),
    ("l5s1_compression_n", "L5/S1 compression", "N", "l5s1_compression_n"),
    ("neck_load_kg", "Neck equivalent load", "kg", None),
    ("distance_m", "Distance to robot", "m", None),
]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_session(session_dir: str | Path) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    d = Path(session_dir)
    with open(d / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols: dict[str, np.ndarray] = {}
    if rows:
        for k in rows[0]:
            if k in (
                "posture",
                "rula_risk",
                "reba_risk",
                "active_issues",
                "distance_source",
                "follow_action",
            ):
                cols[k] = np.array([r[k] for r in rows], dtype=object)
            else:
                cols[k] = np.array([_f(r[k]) for r in rows])
    events = []
    ev_path = d / "events.jsonl"
    if ev_path.exists():
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    cfg = {}
    if (d / "config.yaml").exists():
        import yaml

        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
    return cols, events, cfg


def summarize(session_dir: str | Path) -> dict:
    cols, events, cfg = load_session(session_dir)
    th = cfg.get("thresholds", {})
    out: dict = {"session": str(session_dir), "samples": 0}
    if not cols or len(cols.get("t", [])) == 0:
        return out
    t = cols["t"]
    present = cols["present"] == 1
    dt = np.diff(t, append=t[-1])
    dt = np.clip(dt, 0, 2.0)  # gaps (e.g. paused) do not count as time
    out["samples"] = int(len(t))
    out["duration_min"] = round(float(t[-1] - t[0]) / 60, 2)
    out["in_view_min"] = round(float(dt[present].sum()) / 60, 2)

    def bands(key, spec):
        s = cols.get(key)
        res = {}
        if s is None:
            return res
        tot = dt[present & ~np.isnan(s)].sum()
        for label, lo, hi, _ in spec:
            sel = present & (s >= lo) & (s <= hi)
            res[label] = round(100 * float(dt[sel].sum()) / tot, 1) if tot else 0.0
        return res

    out["rula_time_pct"] = bands("rula", RULA_BANDS)
    out["reba_time_pct"] = bands("reba", REBA_BANDS)
    for key, name in (("rula", "rula"), ("reba", "reba")):
        s = cols.get(key)
        if s is not None and np.any(~np.isnan(s)):
            out[f"{name}_mean"] = round(float(np.nanmean(s)), 2)
            out[f"{name}_max"] = int(np.nanmax(s))

    stats = {}
    for key, label, unit, thkey in TRACKED:
        s = cols.get(key)
        if s is None or not np.any(~np.isnan(s[present])):
            continue
        v = s[present]
        row = {
            "label": label,
            "unit": unit,
            "mean": round(float(np.nanmean(v)), 1),
            "p95": round(float(np.nanpercentile(np.abs(v), 95)), 1),
            "max_abs": round(float(np.nanmax(np.abs(v))), 1),
        }
        if thkey and thkey in th:
            over = present & (np.abs(np.nan_to_num(s)) > float(th[thkey]))
            row["threshold"] = th[thkey]
            row["pct_time_over"] = round(100 * float(dt[over].sum()) / max(float(dt[present].sum()), 1e-9), 1)
        stats[key] = row
    out["metrics"] = stats

    postures = Counter(p for p, pr in zip(cols.get("posture", []), present, strict=False) if pr and p)
    out["posture_samples"] = dict(postures)

    coaching = [e for e in events if e.get("kind") in ("issue", "break")]
    out["coaching_messages"] = len(coaching)
    out["coaching_by_issue"] = dict(Counter(e.get("issue", e.get("kind")) for e in coaching))
    llm = [e for e in coaching if e.get("source") == "llm"]
    out["llm_messages"] = len(llm)
    if llm:
        out["llm_latency_s_mean"] = round(float(np.mean([e.get("llm_latency_s", 0) for e in llm])), 2)
    out["praise"] = sum(1 for e in events if e.get("kind") == "praise")
    return out


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def make_charts(cols: dict, cfg: dict) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = cfg.get("thresholds", {})
    t = (cols["t"] - cols["t"][0]) / 60.0
    charts = []

    def series_chart(key, title, unit, thval, color):
        s = cols.get(key)
        if s is None or np.all(np.isnan(s)):
            return
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.plot(t, s, color=color, linewidth=2)
        if thval is not None:
            ax.axhline(thval, color=INK2, linestyle="--", linewidth=1)
            ax.text(t[-1], thval, f" limit {thval:g} {unit}", color=INK2, fontsize=8, va="bottom", ha="right")
        _style(ax, title, unit)
        ax.set_xlabel("minutes", color=INK2, fontsize=9)
        charts.append((title, _fig_to_b64(fig)))
        plt.close(fig)

    series_chart(
        "neck_flexion", "Neck flexion (head forward/down vs trunk)", "deg", th.get("neck_flexion"), BLUE
    )
    series_chart("trunk_flexion", "Trunk flexion (forward lean)", "deg", th.get("trunk_flexion"), BLUE)
    series_chart(
        "l5s1_compression_n", "Estimated L5/S1 compression", "N", th.get("l5s1_compression_n"), ORANGE
    )
    d = cols.get("distance_m")
    if d is not None and np.any(~np.isnan(d)):
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.axhspan(2.0, 3.0, color=STATUS["good"], alpha=0.10, linewidth=0)
        ax.text(t[0], 2.95, "safe band 2-3 m", color=INK2, fontsize=7, va="top")
        ax.plot(t, d, color=BLUE, linewidth=2)
        _style(ax, "Distance between you and Stretch", "m")
        ax.set_xlabel("minutes", color=INK2, fontsize=9)
        charts.append(("Distance between you and Stretch", _fig_to_b64(fig)))
        plt.close(fig)

    for key, spec, title in (
        ("rula", RULA_BANDS, "RULA score over time"),
        ("reba", REBA_BANDS, "REBA score over time"),
    ):
        s = cols.get(key)
        if s is None or np.all(np.isnan(s)):
            continue
        fig, ax = plt.subplots(figsize=(9, 2.6))
        for label, lo, hi, status in spec:
            ax.axhspan(lo - 0.5, hi + 0.5, color=STATUS[status], alpha=0.10, linewidth=0)
            ax.text(t[0], hi + 0.35, label, color=INK2, fontsize=7, va="top")
        ax.step(t, s, where="post", color=INK, linewidth=1.6)
        _style(ax, title, "score")
        ax.set_xlabel("minutes", color=INK2, fontsize=9)
        charts.append((title, _fig_to_b64(fig)))
        plt.close(fig)
    return charts


def scene_section(session_dir: Path, cfg: dict) -> str:
    """'Scene understanding' block: static map (people removed), where the person was, free space, OneSLAM."""
    from .scene import ONESLAM_REPO, build_scene, load_frames

    if not load_frames(session_dir):
        return ""
    try:
        sc = build_scene(session_dir, cfg)
    except Exception as e:  # never lose the whole report over the map
        return f"<h2>Scene understanding</h2><p class=muted>Scene map failed: {html.escape(str(e))}</p>"
    if not sc or sc.get("empty"):
        return (
            "<h2>Scene understanding</h2><p class=muted>Keyframes were recorded but without depth, so no "
            "map could be built.</p>"
        )
    img = ""
    mp = session_dir / "scene" / "map.png"
    if mp.exists():
        img = (
            f'<p><img alt="Top-down map" src="data:image/png;base64,{base64.b64encode(mp.read_bytes()).decode()}">'
            "</p>"
        )
    src = {
        "kinematic": "head joint states + wheel odometry (reference, not camera-only)",
        "oneslam+kinematic": "OneSLAM camera-only trajectory (metric scale from depth)",
    }.get(sc["source"], sc["source"])
    rows = [
        ("Pose source", src),
        ("Keyframes (with depth)", f"{sc['frames']} ({sc['frames_with_depth']})"),
        ("Observed floor area", f"{sc.get('observed_area_m2', '-')} m²"),
        ("Free floor", f"{sc.get('free_floor_m2', '-')} m²"),
        ("Obstacles (footprint)", f"{sc.get('obstacle_area_m2', '-')} m²"),
        ("Mapped extent", " × ".join(f"{v} m" for v in sc.get("extent_m", []))),
        ("Camera travel", f"{sc.get('camera_travel_m', 0)} m"),
    ]
    p = sc.get("person")
    if p:
        rows += [
            ("Person: distance walked", f"{p['walked_m']} m"),
            ("Person: floor area used", f"{p['area_used_m2']} m²"),
            (
                "Person: free space around them (mean / min)",
                f"{p['clearance_mean_m']} / {p['clearance_min_m']} m"
                if p["clearance_mean_m"] is not None
                else "-",
            ),
            ("Person: mean distance to Stretch", f"{p['distance_to_robot_mean_m']} m"),
            ("Person: posture samples", ", ".join(f"{k} {v}" for k, v in p["posture_samples"].items())),
        ]
    table = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    os = sc.get("oneslam")
    if os:
        if os.get("metric"):
            sc_ = os["scale"]
            orow = [
                (
                    "Metric scale (from depth)",
                    f"{sc_['scale']:.4f} ({sc_['pairs']} frame pairs, ±{sc_['spread_pct']}%)",
                ),
                (
                    "Camera path vs. kinematic reference (RMSE / final)",
                    f"{os['cam_vs_kinematic_rmse_m']} / {os['cam_vs_kinematic_final_m']} m",
                ),
                (
                    "Head-only motion check (base must stay still)",
                    f"{os.get('head_only_check') or 'no head-only segments'}"
                    + (
                        f" - max drift {os['head_only_base_drift_max_m']} m"
                        if os.get("head_only_base_drift_max_m") is not None
                        else ""
                    ),
                ),
            ]
        else:
            orow = [("Status", os.get("note", "imported without metric scale"))]
        os_html = (
            "<h3>OneSLAM (camera-only localization)</h3><table>"
            + "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in orow)
            + "</table>"
        )
    else:
        os_html = (
            f"<p class=muted>Camera-only localization: export this session with <code>posture-coach scene "
            f'export-oneslam {html.escape(session_dir.name)}</code>, run <a href="{ONESLAM_REPO}">OneSLAM</a> on a '
            f"GPU machine (<code>scripts/run_oneslam.sh</code>), then <code>posture-coach scene import-oneslam</code>. "
            f"See docs/ONESLAM.md.</p>"
        )
    return (
        f"<h2>Scene understanding</h2>{img}"
        "<p class=muted>Top-down view, Stretch's start at the black dot looking along the arrow. Light = free floor, "
        "blue = obstacles (darker = taller), dark = not seen, orange = where the person spent time. People are removed "
        "from the static map. "
        f'<a href="/sessions/{session_dir.name}/scene3d">Open the 3-D view</a> · '
        f'<a href="/sessions/{session_dir.name}/scene/scene.ply" download>download the point cloud (PLY)</a></p>'
        f"<table>{table}</table>{os_html}"
    )


def write_report(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    cols, events, cfg = load_session(session_dir)
    summary = summarize(session_dir)
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    charts = make_charts(cols, cfg) if summary.get("samples") else []
    scene_html = scene_section(session_dir, cfg)

    def table(d: dict) -> str:
        return (
            "<table>"
            + "".join(f"<tr><td>{html.escape(str(k))}</td><td>{v}%</td></tr>" for k, v in d.items())
            + "</table>"
        )

    rows = "".join(
        f"<tr><td>{html.escape(r['label'])}</td><td>{r['mean']}</td><td>{r['p95']}</td><td>{r['max_abs']}</td>"
        f"<td>{r.get('threshold', '-')}</td><td>{r.get('pct_time_over', '-')}"
        f"{'%' if 'pct_time_over' in r else ''}</td><td>{r['unit']}</td></tr>"
        for r in summary.get("metrics", {}).values()
    )
    coach = "".join(
        f"<li>{html.escape(str(k))}: {v}</li>" for k, v in summary.get("coaching_by_issue", {}).items()
    )
    said = [e for e in events if e.get("text")]
    transcript = "".join(
        f"<tr><td>{datetime.fromtimestamp(float(e.get('wall_time', 0))).strftime('%H:%M:%S')}</td>"
        f"<td>{html.escape(str(e.get('issue') or e.get('key') or e.get('kind', '')))}</td>"
        f"<td>{html.escape(e['text'])}</td><td class=muted>{html.escape(str(e.get('source', '')))}</td></tr>"
        for e in said
    )
    imgs = "".join(
        f'<p><img alt="{html.escape(title)}" src="data:image/png;base64,{b64}"></p>' for title, b64 in charts
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Posture session {session_dir.name}</title>
<style>body{{font-family:system-ui,sans-serif;background:#f9f9f7;color:{INK};max-width:980px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;margin:8px 0 16px}}td,th{{border-bottom:1px solid {GRID};padding:4px 10px;text-align:left;font-size:14px}}
th{{color:{INK2};font-weight:600}}img{{max-width:100%;background:{SURFACE};border-radius:6px}}.muted{{color:{INK2};font-size:13px}}</style>
</head><body><p class="muted"><a href="/sessions">All sessions</a> · <a href="/">Live view</a></p>
<h1>Posture session {html.escape(session_dir.name)}</h1>
<p>Duration {summary.get("duration_min", 0)} min, in view {summary.get("in_view_min", 0)} min.
RULA mean {summary.get("rula_mean", "-")} (max {summary.get("rula_max", "-")}), REBA mean {summary.get("reba_mean", "-")}
(max {summary.get("reba_max", "-")}). Coaching messages: {summary.get("coaching_messages", 0)}
({summary.get("llm_messages", 0)} from the LLM, mean latency {summary.get("llm_latency_s_mean", "-")} s); praise: {summary.get("praise", 0)}.</p>
<h2>Time in risk bands</h2><div style="display:flex;gap:40px"><div><b>RULA</b>{table(summary.get("rula_time_pct", {}))}</div>
<div><b>REBA</b>{table(summary.get("reba_time_pct", {}))}</div></div>
<h2>Metrics (while in view)</h2><table><tr><th>Metric</th><th>Mean</th><th>P95 |x|</th><th>Max |x|</th><th>Limit</th>
<th>Time over limit</th><th>Unit</th></tr>{rows}</table>
<h2>Coaching by issue</h2><ul>{coach or "<li>none</li>"}</ul>{imgs}
{scene_html}
<h2>What Stretch said</h2><table><tr><th>Time</th><th>Why</th><th>Message</th><th>Source</th></tr>
{transcript or "<tr><td colspan=4>Nothing.</td></tr>"}</table>
<p class="muted">Estimates from monocular pose (MediaPipe) and a static segment model; intended for ergonomic coaching,
not clinical diagnosis.</p></body></html>"""
    out = session_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out


def latest_session(sessions_dir: str | Path = "sessions") -> Path | None:
    d = Path(sessions_dir)
    if not d.exists():
        return None
    subs = sorted([p for p in d.iterdir() if (p / "metrics.csv").exists()])
    return subs[-1] if subs else None
