"""Live web viewer (MJPEG + JSON) on http://<robot>:8080, plus session reports.

When the session ends, open pages switch to the report automatically, and the server keeps serving it.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from . import web

log = logging.getLogger(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Stretch Posture Coach</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{background:#151515;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:16px}
img{max-width:100%;max-height:85vh;border-radius:8px}pre{font-size:12px;white-space:pre-wrap}
.row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1;min-width:280px}a{color:#7fb2f0}
.bar{display:flex;gap:18px;align-items:baseline}
#alarm{display:none;position:sticky;top:0;z-index:5;padding:16px 20px;border-radius:10px;margin-bottom:12px;
 background:#d03b3b;color:#fff;font-size:20px;animation:blink 1s steps(2,start) infinite}
#alarm.check{background:#c77700;animation:none}#alarm button{font-size:20px;padding:10px 18px;margin-left:16px;
 border:0;border-radius:8px;background:#fff;color:#900;font-weight:700}#alarm img{max-height:160px;float:right;border-radius:6px}
@keyframes blink{to{background:#7a1010}}
#endbtn{font-size:15px;padding:7px 14px;border-radius:8px;border:0;background:#2a78d6;color:#fff;cursor:pointer}
#endbtn:disabled{background:#555}#saved{display:none;padding:10px 14px;border-radius:8px;background:#1f3b26;
 color:#cfe9d4;margin-bottom:12px}#saved a{color:#9fe0b0;font-weight:600}</style></head>
<body><div id="alarm"><img id="snap" alt=""><b id="atitle"></b> <span id="areason"></span>
<button onclick="fetch('/alarm/ack',{method:'POST'})">I'm OK / acknowledge</button></div><div class="bar"><h3>Stretch Posture Coach</h3><a href="/sessions">Past sessions</a>__VR__
<button id="endbtn" onclick="endSession()">End session &amp; save report</button></div>
<div id="saved"></div>
<div class="row"><div><img id="live" src="/stream"></div>
<div class="col"><h4>Live metrics</h4><pre id="m">waiting...</pre></div></div>
<script>
setInterval(async()=>{try{const r=await fetch('/metrics');document.getElementById('m').textContent=
JSON.stringify(await r.json(),null,1)}catch(e){}},1000);
let lastShown='';
async function endSession(){
 if(!confirm('End this session and save its report? Stretch keeps running and starts a new session.'))return;
 const b=document.getElementById('endbtn');b.disabled=true;b.textContent='Saving…';
 try{await fetch('/session/end',{method:'POST'})}catch(e){}
 setTimeout(()=>{b.disabled=false;b.textContent='End session & save report'},4000);
}
function showSaved(r){
 if(!r||!r.name)return;const box=document.getElementById('saved');const key=r.name+r.status;
 if(key===lastShown)return;lastShown=key;box.style.display='block';
 box.innerHTML=r.status==='ready'?`Session <b>${r.name}</b> saved (${r.in_view_min??'-'} min in view, RULA mean ${r.rula_mean??'-'}). `+
  `<a href="${r.url}" target="_blank">Open report</a> · a new session is recording.`:
  r.status==='building'?`Saving session <b>${r.name}</b>… building the report.`:`Report for ${r.name} failed - see <a href="/sessions">sessions</a>.`;
}
setInterval(async()=>{try{const s=await (await fetch('/state')).json();showSaved(s.last_report);
 const a=s.alarm||{}, box=document.getElementById('alarm');
 if(a.active){box.style.display='block';box.className=a.state==='alarm'?'':'check';
  document.getElementById('atitle').textContent=a.state==='alarm'?'FALL ALARM':'Possible fall - checking…';
  document.getElementById('areason').textContent=a.reason||'';
  document.getElementById('snap').src='/alarm/snapshot.jpg?'+Date.now();}
 else{box.style.display='none'}
 if(s.state==='ended'&&s.report){location.href=s.report}}catch(e){}},1000);
</script></body></html>"""


class FrameStreamer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        quality: int = 75,
        sessions_dir: str | Path = "sessions",
        vr_port: int | None = None,
    ):
        self.quality = int(quality)
        self.sessions_dir = Path(sessions_dir)
        self._jpeg: bytes | None = None
        self._metrics: dict = {}
        self._cond = threading.Condition()
        self._ended = False
        self.report_path: Path | None = None
        self.alarm: dict = {}  # fall-detector state, set by the pipeline
        self.on_ack = None  # callable(who) - "I'm OK" / acknowledge button
        self.snapshot_provider = None  # callable() -> jpeg bytes | None
        self.on_end_session = None  # callable(who) -> dict - "End session & save report" button
        self.last_report: dict | None = None
        vr_link = (
            (
                f' <a id="vr" href="#">VR view (Quest)</a><script>document.getElementById("vr").href='
                f'"https://"+location.hostname+":{vr_port}/"</script>'
            )
            if vr_port
            else ""
        )
        page = PAGE.replace("__VR__", vr_link).encode("utf-8")
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    if streamer._ended:
                        self.send_response(302)
                        self.send_header("Location", "/report" if streamer.report_path else "/sessions")
                        self.end_headers()
                    else:
                        self._send(200, "text/html; charset=utf-8", page)
                elif path == "/metrics":
                    self._send(200, "application/json", json.dumps(streamer._metrics, default=str).encode())
                elif path == "/state":
                    st = {
                        "state": "ended" if streamer._ended else "running",
                        "report": "/report" if streamer.report_path else None,
                        "alarm": streamer.alarm,
                        "last_report": streamer.last_report,
                    }
                    self._send(200, "application/json", json.dumps(st).encode())
                elif path.startswith("/stream"):
                    self._mjpeg()
                elif path == "/alarm/snapshot.jpg":
                    snap = streamer.snapshot_provider() if streamer.snapshot_provider else None
                    if snap:
                        self._send(200, "image/jpeg", snap)
                    else:
                        self._send(404, "text/plain", b"no snapshot")
                else:
                    res = web.route(
                        path, streamer.sessions_dir, streamer.report_path, '<a href="/">Live view</a>'
                    )
                    if res is None:
                        self._send(404, "text/plain", b"not found")
                    else:
                        self._send(*res)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                if path == "/alarm/ack" and streamer.on_ack:
                    streamer.on_ack("web")
                    self._send(200, "application/json", b'{"ok":true}')
                elif path == "/session/end" and streamer.on_end_session and not streamer._ended:
                    self._send(
                        200, "application/json", json.dumps(streamer.on_end_session("web button")).encode()
                    )
                else:
                    self._send(404, "text/plain", b"not found")

            def _mjpeg(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while not streamer._ended:
                        with streamer._cond:
                            streamer._cond.wait(timeout=2.0)
                            jpg = streamer._jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpg)).encode()
                            + b"\r\n\r\n"
                            + jpg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="mjpeg")
        self._thread.start()
        self.port = self._server.server_address[1]

    def publish(self, frame: np.ndarray, metrics: dict | None = None) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            if metrics is not None:
                self._metrics = metrics
            self._cond.notify_all()

    def set_ended(self, report_path: Path | None) -> None:
        """Session over: live pages redirect to the report."""
        self.report_path = report_path
        self._ended = True
        with self._cond:
            self._cond.notify_all()

    def close(self):
        self._ended = True
        with self._cond:
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()


def serve_reports(sessions_dir: str | Path, host: str = "0.0.0.0", port: int = 8080) -> FrameStreamer:
    """Reports-only server (no live session): every page shows the sessions list / latest report."""
    s = FrameStreamer(host, port, sessions_dir=sessions_dir)
    s.set_ended(None)
    return s
