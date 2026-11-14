"""Shared web routes: session list, per-session reports and raw files (used by the live viewer and VR server)."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
RAW_FILES = {
    "summary.json": "application/json",
    "metrics.csv": "text/csv",
    "events.jsonl": "application/x-ndjson",
    "config.yaml": "text/plain; charset=utf-8",
}

Route = tuple[int, str, bytes]


def list_sessions(sessions_dir: str | Path) -> list[Path]:
    d = Path(sessions_dir)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if (p / "metrics.csv").exists()), reverse=True)


def ensure_report(session: Path) -> Path:
    """Return session/report.html, (re)building it if missing or older than the data."""
    from .report import write_report

    out = session / "report.html"
    csv_path = session / "metrics.csv"
    if not out.exists() or (csv_path.exists() and csv_path.stat().st_mtime > out.stat().st_mtime + 1):
        out = write_report(session)
    return out


def _summary(session: Path) -> dict:
    p = session / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


def _label(name: str) -> str:
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").strftime("%a %d %b %Y, %H:%M")
    except ValueError:
        return name


def sessions_index(sessions_dir: str | Path, extra_links: str = "") -> bytes:
    rows = []
    for s in list_sessions(sessions_dir):
        sm = _summary(s)
        rows.append(
            f"<tr><td><a href='/sessions/{html.escape(s.name)}/'>{html.escape(_label(s.name))}</a></td>"
            f"<td>{sm.get('in_view_min', '-')}</td><td>{sm.get('rula_mean', '-')}</td>"
            f"<td>{sm.get('reba_mean', '-')}</td><td>{sm.get('coaching_messages', '-')}</td></tr>"
        )
    body = "".join(rows) or "<tr><td colspan=5>No sessions yet.</td></tr>"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Posture sessions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;background:#f9f9f7;color:#0b0b0b;max-width:900px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #e4e3df;padding:8px;text-align:left}}
th{{color:#52514e}}a{{color:#2a78d6}}</style></head><body><h1>Posture sessions</h1>
<p>{extra_links}</p><table><tr><th>Session</th><th>In view (min)</th><th>RULA mean</th><th>REBA mean</th>
<th>Coaching messages</th></tr>{body}</table></body></html>"""
    return page.encode("utf-8")


def route(
    path: str, sessions_dir: str | Path, current_report: Path | None = None, extra_links: str = ""
) -> Route | None:
    """Handle report/session URLs. Returns (status, content_type, body) or None if the path isn't ours."""
    path = path.split("?", 1)[0]
    if path == "/report":
        report = current_report
        if report is None:
            sessions = list_sessions(sessions_dir)
            report = ensure_report(sessions[0]) if sessions else None
        if report is None or not report.exists():
            return 404, "text/plain", b"No report yet."
        return 200, "text/html; charset=utf-8", report.read_bytes()
    if path in ("/sessions", "/sessions/"):
        return 200, "text/html; charset=utf-8", sessions_index(sessions_dir, extra_links)
    m = re.match(r"^/sessions/([^/]+)/(.*)$", path)
    if not m:
        return None
    name, rest = m.group(1), m.group(2)
    if not SESSION_RE.match(name) or name in (".", ".."):
        return 404, "text/plain", b"not found"
    session = Path(sessions_dir) / name
    if not (session / "metrics.csv").exists():
        return 404, "text/plain", b"not found"
    if rest in ("", "report.html"):
        return 200, "text/html; charset=utf-8", ensure_report(session).read_bytes()
    if rest in RAW_FILES and (session / rest).exists():
        return 200, RAW_FILES[rest], (session / rest).read_bytes()
    return 404, "text/plain", b"not found"
