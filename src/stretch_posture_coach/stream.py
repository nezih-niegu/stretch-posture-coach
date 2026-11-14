"""Tiny MJPEG + JSON web viewer so you can watch the overlay from your laptop's browser over Wi-Fi."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

log = logging.getLogger(__name__)

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Stretch Posture Coach</title>
<style>body{background:#151515;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:16px}
img{max-width:100%;max-height:85vh;border-radius:8px}pre{font-size:12px;white-space:pre-wrap}
.row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1;min-width:280px}</style></head>
<body><h3>Stretch Posture Coach</h3><div class="row"><div><img src="/stream"></div>
<div class="col"><h4>Live metrics</h4><pre id="m">waiting...</pre></div></div>
<script>setInterval(async()=>{try{const r=await fetch('/metrics');document.getElementById('m').textContent=
JSON.stringify(await r.json(),null,1)}catch(e){}},1000)</script></body></html>"""


class FrameStreamer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, quality: int = 75):
        self.quality = int(quality)
        self._jpeg: bytes | None = None
        self._metrics: dict = {}
        self._cond = threading.Condition()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html", PAGE)
                elif self.path == "/metrics":
                    self._send(200, "application/json", json.dumps(streamer._metrics, default=str).encode())
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
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
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
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

    def close(self):
        self._server.shutdown()
        self._server.server_close()
