"""Command-line interface: ``posture-coach run | follow | calibrate | check | report | say | norteno-setup | ...``."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from .config import home, load_config, parse_override, set_dotted

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Stretch 3 posture coach: MediaPipe full-body biomechanics + spoken coaching via Ollama (qwen2).",
)

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: ./config.yaml if present)")
SetOpt = typer.Option(
    None, "--set", "-s", help="Override any setting, e.g. -s llm.model=qwen2:7b (repeatable)"
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cfg(config: Path | None, sets: list[str] | None, **direct) -> dict:
    overrides: dict = {}
    for item in sets or []:
        k, v = parse_override(item)
        set_dotted(overrides, k, v)
    for dotted, value in direct.items():
        if value is not None:
            set_dotted(overrides, dotted.replace("__", "."), value)
    return load_config(config, overrides)


NORTENO_MODEL = "stretch-norteno"
SPANISH_VOICE = "es_MX-claude-high"


def _localize(cfg: dict, norteno: bool = False) -> None:
    """Switch persona/voices for Spanish. With --norteno, also use the custom Ollama model if it exists."""
    c = cfg["coach"]
    if norteno:
        c["persona"], c["language"] = "norteno", "Spanish"
    spanish = str(c.get("language", "")).lower().startswith(("span", "espa")) or c.get("persona") == "norteno"
    if spanish:
        if str(cfg["speech"].get("piper_voice", "")).startswith("en_"):
            cfg["speech"]["piper_voice"] = SPANISH_VOICE
        if str(cfg["speech"].get("espeak_voice", "")).startswith("en"):
            cfg["speech"]["espeak_voice"] = "es-419"
    if c.get("persona") == "norteno" and str(cfg["llm"].get("model", "")).startswith("qwen2"):
        from .llm import OllamaClient, OllamaError

        try:
            names = OllamaClient.from_config(cfg["llm"]).list_models()
            if any(n.split(":")[0] == NORTENO_MODEL for n in names):
                cfg["llm"]["model"] = NORTENO_MODEL
            else:
                logging.getLogger(__name__).info(
                    "Custom model %s not found (run `posture-coach norteno-setup`); using %s with the norteño prompt",
                    NORTENO_MODEL,
                    cfg["llm"]["model"],
                )
        except OllamaError:
            pass


def _run_pipeline(cfg: dict, duration: float | None, report: bool) -> None:
    import time

    from .pipeline import Pipeline, _host_ip

    serve = bool(cfg["display"].get("serve_report", True)) and report
    pipe = Pipeline(cfg, keep_servers=serve)
    session = pipe.run(duration_s=duration)
    out = None
    if session and report:
        from .report import summarize, write_report

        out = write_report(session)
        s = summarize(session)
        typer.echo(f"\nSession saved to {session}")
        typer.echo(f"Report: {out}")
        typer.echo(
            f"In view {s.get('in_view_min', 0)} min | RULA mean {s.get('rula_mean', '-')} | "
            f"REBA mean {s.get('reba_mean', '-')} | coaching messages {s.get('coaching_messages', 0)}"
        )
    if not serve or (pipe.streamer is None and pipe.vr is None):
        pipe.close_servers()
        return
    pipe.show_report(out)
    ip = _host_ip()
    if pipe.streamer:
        typer.echo(
            f"\nReport on the web:  http://{ip}:{pipe.streamer.port}/report   (all sessions: /sessions)"
        )
    if pipe.vr:
        typer.echo(f"Report in the Quest: {pipe.vr.scheme}://{ip}:{pipe.vr.port}/report")
    typer.echo("Open pages switch to the report automatically. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close_servers()


@app.command()
def run(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    source: str = typer.Option(None, help="realsense | webcam | webcam:N | path/to/video.mp4"),
    model: str = typer.Option(None, help="Ollama model, e.g. qwen2:1.5b or qwen2:7b"),
    host: str = typer.Option(None, help="Ollama URL, e.g. http://192.168.1.20:11434 to use your laptop"),
    language: str = typer.Option(None, help="Language Stretch speaks (English, Spanish, ...)"),
    norteno: bool = typer.Option(
        False, "--norteno", help="Spanish norteño de la sierra persona (custom model)"
    ),
    follow: bool = typer.Option(False, "--follow", help="Also turn/drive the base to face you at 2-3 m"),
    mass: float = typer.Option(None, "--mass", help="Your body mass in kg (load model)"),
    no_voice: bool = typer.Option(False, "--no-voice", help="Print messages instead of speaking"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use rule-based messages only"),
    no_robot: bool = typer.Option(False, "--no-robot", help="Do not move the head"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable the web viewer"),
    no_vr: bool = typer.Option(False, "--no-vr", help="Disable the VR telepresence server"),
    window: bool = typer.Option(None, "--window/--no-window", help="Force the OpenCV window on/off"),
    calibrate_first: bool = typer.Option(False, "--calibrate", help="Run neutral-posture calibration first"),
    duration: float = typer.Option(None, help="Stop after N seconds"),
    report: bool = typer.Option(True, "--report/--no-report", help="Write an HTML report when stopping"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch your posture and coach you out loud."""
    _setup_logging(verbose)
    cfg = _cfg(
        config,
        sets,
        camera__source=source,
        llm__model=model,
        llm__host=host,
        coach__language=language,
        subject__body_mass_kg=mass,
    )
    if no_voice:
        cfg["speech"]["engine"] = "print"
    if no_llm:
        cfg["llm"]["enabled"] = False
    if no_robot:
        cfg["robot"]["enabled"] = False
    if no_stream:
        cfg["display"]["stream"] = False
    if no_vr:
        cfg["vr"]["enabled"] = False
    if window is not None:
        cfg["display"]["window"] = window
    if follow:
        cfg["follow"]["enabled"] = True
        cfg["robot"]["enabled"] = True
    _localize(cfg, norteno)
    if calibrate_first:
        from .pipeline import calibrate as do_calibrate

        do_calibrate(cfg)
    _run_pipeline(cfg, duration, report)


@app.command()
def follow(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    target: float = typer.Option(None, help="Preferred distance in m (default 2.5)"),
    coach: bool = typer.Option(False, "--coach/--no-coach", help="Also give posture coaching"),
    norteno: bool = typer.Option(
        False, "--norteno", help="Spanish norteño persona for anything Stretch says"
    ),
    language: str = typer.Option(None, help="Language for spoken notices (English, Spanish)"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    no_vr: bool = typer.Option(False, "--no-vr", help="Disable the VR telepresence server"),
    duration: float = typer.Option(None, help="Stop after N seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Face the person and keep a safe 2-3 m distance with the mobile base (head tracks, base turns/drives).

    SAFETY: clear the floor around the robot, keep the runstop (head button) in reach. Ctrl+C stops."""
    _setup_logging(verbose)
    cfg = _cfg(config, sets, follow__target_m=target, coach__language=language)
    cfg["follow"]["enabled"] = True
    cfg["robot"]["enabled"] = True
    cfg["coach"]["enabled"] = coach
    if no_stream:
        cfg["display"]["stream"] = False
    if no_vr:
        cfg["vr"]["enabled"] = False
    _localize(cfg, norteno)
    _run_pipeline(cfg, duration, report=coach)


@app.command("alarm-test")
def alarm_test(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    kind: str = typer.Option("test", help="test | check | alarm | resolved (which notification to send)"),
    siren: bool = typer.Option(False, "--siren", help="Also play the siren once through the speaker"),
) -> None:
    """Send a fall-alarm phone notification (ntfy / webhook) exactly like a real alarm would."""
    _setup_logging(False)
    cfg = _cfg(config, sets)
    _localize(cfg)
    from .alarm import Alarm, siren_wav
    from .personas import resolve
    from .pipeline import _host_ip

    a = cfg.get("alarm", {})
    topic = str(a.get("ntfy_topic") or "").strip()
    if not topic and not str(a.get("webhook_url") or "").strip():
        typer.echo(
            "No alarm.ntfy_topic (or webhook_url) in config.yaml - nothing to send.\n"
            "Add to config.yaml:\n\nalarm:\n  ntfy_topic: <your-topic>\n"
        )
        raise typer.Exit(1)
    spanish = resolve(cfg["coach"]).spanish
    al = Alarm(
        a,
        None,
        {},
        live_url=f"http://{_host_ip()}:{cfg['display'].get('stream_port', 8080)}/",
        spanish=spanish,
    )
    reason = (
        ("Prueba de notificación de caída" if spanish else "Fall alarm notification test")
        if kind == "test"
        else ("Prueba: " if spanish else "Test: ") + kind
    )
    al.notify(kind, reason)
    ok = [n for n in al.sent_notifications if n.get("status", 999) < 400]
    for n in al.sent_notifications:
        typer.echo(f"  {n['to']}: " + (f"HTTP {n['status']}" if "status" in n else f"ERROR {n['error']}"))
    if siren:
        import tempfile

        from .speech import _play_wav

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(siren_wav(2.0))
            f.flush()
            _play_wav(f.name, float(a.get("volume", 1.5)))
    if ok:
        typer.echo(f"Sent to ntfy topic '{topic}'. Check your phone." if topic else "Sent.")
    else:
        typer.echo("Sending failed (see errors above). Test the network: curl -4 -d test ntfy.sh/<topic>")
        raise typer.Exit(1)


@app.command("follow-test")
def follow_test(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    move: bool = typer.Option(True, "--move/--no-move", help="Do the small base test moves (asks first)"),
    seconds: float = typer.Option(20, help="Length of the camera dry run"),
) -> None:
    """Diagnose follow mode: robot base access, a tiny confirmed test move, then a camera dry run that shows
    what follow WOULD do (distance, bearing, decision) without moving the base."""
    import math
    import time

    _setup_logging(False)
    cfg = _cfg(config, sets)
    from .robot import StretchHead, wait_for_tilt

    typer.echo("1) Opening the robot in base mode (stretch_body Robot)...")
    try:
        head = StretchHead(dict(cfg["robot"], enabled=True), base=True)
    except Exception as e:
        typer.echo(f"   FAIL: {e}")
        typer.echo("   -> Is ROS/another program using the robot? Run: stretch_free_robot_process.py")
        raise typer.Exit(1) from None
    try:
        wait_for_tilt(head, 3)
        time.sleep(0.5)
        typer.echo(
            f"   base access: {'YES' if head.has_base else 'NO'} | runstop: {head.runstop} | "
            f"head pan {head.pan} tilt {head.tilt} | odometry x={head.base_x} theta={head.base_theta}"
        )
        if head.runstop:
            typer.echo(
                "   The RUNSTOP is active (the glowing button on the head): press it to release, then retry."
            )
            raise typer.Exit(1)
        if (
            move
            and head.has_base
            and typer.confirm(
                "2) Test the base: turn ~10 deg left and back, then 5 cm forward and back. Area clear?",
                default=False,
            )
        ):

            def odo():
                time.sleep(0.3)
                return head.base_x, head.base_theta

            x0, t0 = odo()
            head.base_rotate(math.radians(10), compensate_head=False)
            time.sleep(2.5)
            x1, t1 = odo()
            head.base_rotate(math.radians(-10), compensate_head=False)
            time.sleep(2.5)
            head.base_translate(0.05)
            time.sleep(2.0)
            x2, _ = odo()
            head.base_translate(-0.05)
            time.sleep(2.0)
            if None not in (t0, t1, x1, x2):
                typer.echo(
                    f"   rotation measured {math.degrees(t1 - t0):+.1f} deg (expected +10), "
                    f"translation {100 * (x2 - x1):+.1f} cm (expected +5)"
                )
                ok = abs(math.degrees(t1 - t0)) > 3 and abs(x2 - x1) > 0.01
                typer.echo(
                    "   base moves: OK" if ok else "   base did NOT move: check runstop / battery / motors"
                )
            else:
                typer.echo("   (no odometry reported; did you see it move?)")

        typer.echo(
            f"3) Camera dry run for {seconds:.0f} s: stand 1-5 m away and walk around (base will NOT move)."
        )
        from .camera import open_source
        from .follow import Follower, person_distance
        from .pipeline import _torso_target, make_estimator

        class DryHead:  # real head pan/runstop, recorded (not executed) base commands
            has_base = True

            def __init__(self, real):
                self.real, self.last = real, ""

            pan = property(lambda self: self.real.pan)
            runstop = property(lambda self: self.real.runstop)

            def base_rotate(self, rad, compensate_head=True):
                self.last = f"WOULD TURN {math.degrees(rad):+.0f} deg"

            def base_translate(self, m):
                self.last = f"WOULD DRIVE {100 * m:+.0f} cm"

        dry = DryHead(head)
        fol = Follower(dict(cfg["follow"], enabled=True), cfg["robot"], dry)
        try:
            src = open_source(cfg["camera"])
            est = make_estimator(cfg)
        except Exception as e:
            typer.echo(f"   camera/pose unavailable: {e}")
            raise typer.Exit(1) from None
        min_vis = float(cfg["pose"].get("min_visibility", 0.5))
        end, last_print = time.monotonic() + seconds, 0.0
        try:
            while time.monotonic() < end:
                frame, ts = src.read()
                if frame is None:
                    continue
                pose = est.process(frame, ts)
                target = _torso_target(pose, min_vis)
                head.track(target, ts)
                dist, dsrc = person_distance(
                    pose, getattr(src, "last_depth", None), min_vis, float(cfg["robot"].get("vfov_deg", 69))
                )
                st = fol.update(target is not None, dist, dsrc, target[0] if target else None, ts)
                if ts - last_print > 0.5:
                    last_print = ts
                    d = "-" if st.distance is None else f"{st.distance:.2f} m ({st.source})"
                    b = "-" if st.bearing_deg is None else f"{st.bearing_deg:+.0f} deg"
                    typer.echo(f"   distance {d:<16} bearing {b:<9} -> {st.action:<9} {st.note} {dry.last}")
                    dry.last = ""
        finally:
            src.close()
            est.close()
        typer.echo("Done. If it says WOULD TURN/DRIVE here, `posture-coach follow` will move the base.")
    finally:
        head.close()


@app.command("lidar-test")
def lidar_test(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    seconds: float = typer.Option(6, help="How long to listen"),
) -> None:
    """Check the base lidar (RPLidar A1) and its mounting: stand ~1 m in FRONT of the robot while it runs."""
    import math
    import time

    import numpy as np

    _setup_logging(False)
    cfg = _cfg(config, sets)
    lc = cfg["lidar"]
    from .lidar import RPLidar, to_base

    typer.echo(f"Opening {lc['port']}...")
    try:
        lid = RPLidar(str(lc["port"]), int(lc.get("baud", 115200)))
    except Exception as e:
        typer.echo(
            f"FAIL: {e}\n-> Is the lidar plugged in (ls -l /dev/hello-lrf)? Is ROS (stretch_driver) using it?"
        )
        raise typer.Exit(1) from None
    lid.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds and lid.error is None:
        time.sleep(0.2)
    n, scan = lid.scans, lid.latest
    lid.close()
    if lid.error or scan is None:
        typer.echo(f"FAIL: no scans ({lid.error or 'timeout'}). Health: {lid.health}")
        raise typer.Exit(1)
    pts = to_base(
        scan,
        float(lc.get("yaw_offset_deg", 180)),
        bool(lc.get("clockwise", True)),
        float(lc.get("x", 0)),
        float(lc.get("y", 0)),
        float(lc.get("min_range_m", 0.25)),
    )
    typer.echo(f"OK: {n / seconds:.1f} scans/s, {len(pts)} valid points per scan, health {lid.health}")
    if len(pts):
        d = np.hypot(pts[:, 0], pts[:, 1])
        k = int(np.argmin(d))
        bearing = math.degrees(math.atan2(pts[k, 1], pts[k, 0]))
        typer.echo(
            f"Nearest object: {d[k]:.2f} m at bearing {bearing:+.0f} deg (0 = straight ahead, + = robot's left)"
        )
        typer.echo(
            "If you stood in front and it says ~180 deg: set lidar.yaw_offset_deg: 0.  If left and right are "
            "swapped (step to the robot's left: bearing should be positive): set lidar.clockwise: false."
        )
        for name, (lo, hi) in {
            "front": (-20, 20),
            "left": (70, 110),
            "back": (160, 200),
            "right": (-110, -70),
        }.items():
            a = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360
            lo_, hi_ = lo % 360, hi % 360
            sel = (a >= lo_) & (a <= hi_) if lo_ < hi_ else (a >= lo_) | (a <= hi_)
            typer.echo(
                f"  {name:5s}: nearest {d[sel].min():.2f} m" if sel.any() else f"  {name:5s}: no returns"
            )


@app.command("norteno-setup")
def norteno_setup(
    base: str = typer.Option(
        "qwen2:1.5b", help="Base model, e.g. qwen2:1.5b (fast) or qwen2:7b (better Spanish)"
    ),
    name: str = typer.Option(NORTENO_MODEL, help="Name of the custom Ollama model"),
    skip_voice: bool = typer.Option(False, "--skip-voice"),
    test: bool = typer.Option(True, "--test/--no-test", help="Ask the new model for a sample line"),
) -> None:
    """Create the custom 'norteño de la sierra' Ollama model + download the Mexican Spanish voice."""
    import shutil
    import subprocess

    from .personas import modelfile

    if not shutil.which("ollama"):
        typer.echo("ollama not found on PATH")
        raise typer.Exit(1)
    pdir = home() / "personas"
    pdir.mkdir(parents=True, exist_ok=True)
    mf = pdir / f"{name}.Modelfile"
    mf.write_text(modelfile("norteno", base), encoding="utf-8")
    typer.echo(f"Modelfile: {mf}")
    if subprocess.run(["ollama", "pull", base]).returncode != 0:
        raise typer.Exit(1)
    if subprocess.run(["ollama", "create", name, "-f", str(mf)]).returncode != 0:
        typer.echo("ollama create failed")
        raise typer.Exit(1)
    if not skip_voice:
        vdir = home() / "models" / "piper"
        vdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", "--download-dir", str(vdir), SPANISH_VOICE]
        )
    if test:
        from .llm import OllamaClient, OllamaError, clean_spoken

        try:
            cli = OllamaClient(model=name, timeout_s=120)
            txt = cli.chat(
                [
                    {
                        "role": "user",
                        "content": "Observación: la persona está sentada. Problema principal: "
                        "encorvado / inclinado hacia adelante durante 15 segundos. ¿Qué le dices?",
                    }
                ]
            )
            typer.echo(f"\n{name} dice: {clean_spoken(txt)}")
        except OllamaError as e:
            typer.echo(f"test failed: {e}")
    typer.echo(
        f"\nListo. Run:  posture-coach run --norteno     (or set coach.persona: norteno and llm.model: {name})"
    )


@app.command()
def calibrate(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    seconds: float = typer.Option(8.0, help="How long to record the neutral posture"),
    source: str = typer.Option(None),
    no_robot: bool = typer.Option(False, "--no-robot"),
    norteno: bool = typer.Option(False, "--norteno", help="Spanish norteño voice/persona for instructions"),
    language: str = typer.Option(None, help="Language for spoken instructions (English, Spanish)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record your neutral posture (sit/stand tall) so angles are measured relative to YOU."""
    _setup_logging(verbose)
    cfg = _cfg(config, sets, camera__source=source, coach__language=language)
    if no_robot:
        cfg["robot"]["enabled"] = False
    _localize(cfg, norteno)
    from .pipeline import calibrate as do_calibrate

    cal = do_calibrate(cfg, seconds)
    typer.echo(json.dumps(cal.to_dict(), indent=2))


@app.command()
def check(
    config: Path = ConfigOpt, sets: list[str] = SetOpt, speak: bool = typer.Option(True, "--speak/--quiet")
) -> None:
    """Check camera, pose model, Ollama, speaker and robot head. Saves check_snapshot.jpg."""
    _setup_logging(False)
    cfg = _cfg(config, sets)
    ok_all = True

    def report(name: str, ok: bool, msg: str):
        nonlocal ok_all
        ok_all &= ok or name in ("robot head", "piper voice")
        typer.echo(f"[{'OK ' if ok else 'FAIL'}] {name:<12} {msg}")

    import time

    import cv2

    from .camera import open_source
    from .pose import model_path

    frame = None
    try:
        with open_source(cfg["camera"]) as src:
            for _ in range(15):  # let auto-exposure settle
                frame, _ = src.read()
            if frame is None:
                raise RuntimeError("no frames")
            snap = home() / "check_snapshot.jpg"
            cv2.imwrite(str(snap), frame)
            report(
                "camera",
                True,
                f"{src.device_name} -> {frame.shape[1]}x{frame.shape[0]} ({snap}: is it upright?)",
            )
    except Exception as e:
        report("camera", False, str(e))

    lport = Path(str(cfg["lidar"].get("port", "/dev/hello-lrf")))
    if cfg["lidar"].get("enabled", "auto") in (False, "false", "off"):
        report("lidar", True, "disabled in config (maps use wheel odometry)")
    elif lport.exists():
        report("lidar", True, f"{lport} present (run `posture-coach lidar-test` to check the mounting)")
    else:
        report("lidar", True, f"{lport} not found: VR scan + scene map use wheel odometry only")

    mp_path = model_path(cfg["pose"]["models_dir"], cfg["pose"]["model"])
    if not mp_path.exists():
        report("pose model", False, f"{mp_path} missing - run `posture-coach download-models`")
    else:
        try:
            from .pipeline import make_estimator

            est = make_estimator(cfg)
            if frame is not None:
                t0 = time.monotonic()
                res = est.process(frame, 0.0)
                ms = (time.monotonic() - t0) * 1000
                report(
                    "pose model",
                    True,
                    f"{mp_path.name} loaded, {ms:.0f} ms/frame, person {'FOUND' if res else 'not found'}",
                )
            else:
                report("pose model", True, f"{mp_path.name} loaded")
            est.close()
        except Exception as e:
            report("pose model", False, str(e))

    from .llm import OllamaClient, OllamaError, clean_spoken

    client = OllamaClient.from_config(cfg["llm"])
    try:
        if not client.has_model():
            report(
                "ollama",
                False,
                f"up at {client.host} but '{client.model}' not pulled: ollama pull {client.model}",
            )
        else:
            t0 = time.monotonic()
            txt = clean_spoken(
                client.chat(
                    [{"role": "user", "content": "In one short sentence, remind me to sit up straight."}]
                )
            )
            report("ollama", True, f"{client.model} answered in {time.monotonic() - t0:.1f}s: {txt!r}")
    except OllamaError as e:
        report("ollama", False, str(e))

    from .speech import PiperEngine, make_engine

    try:
        PiperEngine(cfg["speech"]["piper_voice"], cfg["speech"]["voices_dir"])
        report("piper voice", True, cfg["speech"]["piper_voice"])
    except Exception as e:
        report("piper voice", False, f"{e} (espeak-ng will be used)")
    try:
        eng = make_engine(cfg["speech"])
        if speak:
            eng.say("Hello, I am Stretch. Speaker test.")
        report("speaker", eng.name != "print", f"engine={eng.name}")
    except Exception as e:
        report("speaker", False, str(e))

    from .robot import make_head, wait_for_tilt

    try:
        rcfg = dict(cfg["robot"], enabled=True)
        head = make_head(rcfg)
        tilt = wait_for_tilt(head)
        report("robot head", True, f"pan={head.pan} tilt={tilt}")
        head.close()
    except Exception as e:
        report("robot head", False, f"{e} (optional; run with --no-robot)")

    typer.echo("\nAll required checks passed." if ok_all else "\nSome checks failed - see above.")
    raise typer.Exit(0 if ok_all else 1)


@app.command()
def serve(
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
    port: int = typer.Option(None, help="Port (default: display.stream_port, 8080)"),
) -> None:
    """Serve past session reports at http://<robot>:8080/sessions (no camera, no robot)."""
    import time

    from .pipeline import _host_ip
    from .stream import serve_reports

    _setup_logging(False)
    cfg = _cfg(config, sets)
    srv = serve_reports(
        cfg["logging"]["sessions_dir"],
        cfg["display"].get("stream_host", "0.0.0.0"),
        port or int(cfg["display"].get("stream_port", 8080)),
    )
    typer.echo(f"Reports: http://{_host_ip()}:{srv.port}/sessions   (latest: /report)  - Ctrl+C to quit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


scene_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Scene understanding: static 3-D map of the room + OneSLAM bridge (camera-only SLAM).",
)
app.add_typer(scene_app, name="scene")


def _session_path(session: str | None, cfg: dict) -> Path:
    from .report import latest_session

    root = Path(cfg["logging"]["sessions_dir"])
    if session is None:
        found = latest_session(root)
        if found is None:
            typer.echo("No sessions found.")
            raise typer.Exit(1)
        return found
    p = Path(session)
    return p if p.exists() else root / session


@scene_app.command("build")
def scene_build(
    session: str = typer.Argument(None, help="Session folder or name (default: latest)"),
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
) -> None:
    """(Re)build the scene map and the report for a session."""
    from .report import write_report

    cfg = _cfg(config, sets)
    sess = _session_path(session, cfg)
    out = write_report(sess)
    sc = (
        json.loads((sess / "scene" / "scene.json").read_text())
        if (sess / "scene" / "scene.json").exists()
        else {}
    )
    typer.echo(json.dumps({k: v for k, v in sc.items() if k != "oneslam"}, indent=2))
    typer.echo(f"Report: {out}")


@scene_app.command("export-oneslam")
def scene_export(
    session: str = typer.Argument(None, help="Session folder or name (default: latest)"),
    out: Path = typer.Option(None, help="Output folder (default: <session>/scene/oneslam_input)"),
    with_depths: bool = typer.Option(
        False, "--with-depths", help="Also write depths/ (for an RGB-D OneSLAM fork)"
    ),
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
) -> None:
    """Write the session's keyframes in OneSLAM's input format (images/, calibration.json, ...)."""
    from .pipeline import _host_ip
    from .scene import export_oneslam

    cfg = _cfg(config, sets)
    sess = _session_path(session, cfg)
    d = export_oneslam(sess, out, with_depths, cfg)
    notes = json.loads((d / "stretch_export.json").read_text())
    typer.echo(
        f"OneSLAM input: {d}  ({notes['frames']} frames @ {notes['fps']} Hz, camera travel "
        f"{notes['camera_travel_m']} m)"
    )
    if notes.get("warning"):
        typer.echo(f"WARNING: {notes['warning']}")
    typer.echo("\nOn a machine with an NVIDIA GPU (needs conda + cmake):")
    typer.echo(f"  scp -r hello-robot@{_host_ip()}:{d.resolve()} ./stretch_{sess.name}")
    typer.echo(f"  ./scripts/run_oneslam.sh ./stretch_{sess.name}")
    typer.echo("Then copy poses_pred.txt back and run:")
    typer.echo(f"  posture-coach scene import-oneslam {sess.name} poses_pred.txt")


@scene_app.command("import-oneslam")
def scene_import(
    session: str = typer.Argument(..., help="Session folder or name"),
    poses: Path = typer.Argument(..., help="OneSLAM poses_pred.txt (or its experiment folder)"),
    config: Path = ConfigOpt,
    sets: list[str] = SetOpt,
) -> None:
    """Import a OneSLAM trajectory: metric scale from depth, base poses, validation, rebuilt map + report."""
    from .report import write_report
    from .scene import import_oneslam

    cfg = _cfg(config, sets)
    sess = _session_path(session, cfg)
    m = import_oneslam(sess, poses, cfg)
    typer.echo(json.dumps(m, indent=2))
    typer.echo(f"Report: {write_report(sess)}")


@app.command("report")
def report_cmd(
    session: Path = typer.Argument(None, help="Session folder (default: latest in ./sessions)"),
) -> None:
    """Build report.html + summary.json for a recorded session."""
    from .report import latest_session, summarize, write_report

    session = session or latest_session(load_config()["logging"]["sessions_dir"])
    if session is None:
        typer.echo("No sessions found.")
        raise typer.Exit(1)
    out = write_report(session)
    typer.echo(json.dumps(summarize(session), indent=2))
    typer.echo(f"\nReport written to {out}")


@app.command()
def say(
    text: str = typer.Argument(..., help="Text to speak"), config: Path = ConfigOpt, sets: list[str] = SetOpt
) -> None:
    """Speak a sentence through the robot's speaker (audio test)."""
    from .speech import make_engine

    cfg = _cfg(config, sets)
    eng = make_engine(cfg["speech"])
    typer.echo(f"engine: {eng.name}")
    eng.say(text)


@app.command("download-models")
def download_models_cmd(
    models_dir: Path = typer.Option(None, help="Where to store models (default: $POSTURE_COACH_HOME/models)"),
    voices: list[str] = typer.Option(["en_US-lessac-medium"], "--voice", help="Piper voices (repeatable)"),
    skip_voices: bool = typer.Option(False, "--skip-voices"),
    stt_model: str = typer.Option("base", "--stt-model", help="faster-whisper model for VR push-to-talk"),
    skip_stt: bool = typer.Option(False, "--skip-stt"),
) -> None:
    """Download MediaPipe pose models (lite/full/heavy), Piper voices and the speech-to-text model."""
    from .pose import download_models

    models_dir = models_dir or home() / "models"
    for p in download_models(models_dir):
        typer.echo(f"pose model: {p}")
    if not skip_stt:
        try:
            from .voice import WhisperTranscriber

            WhisperTranscriber(stt_model, models_dir).load()
            typer.echo(f"speech-to-text: whisper {stt_model} in {models_dir / 'whisper'}")
        except Exception as e:
            typer.echo(
                f"warning: speech-to-text model download failed ({e}); VR push-to-talk won't work", err=True
            )
    if not skip_voices:
        import subprocess

        vdir = models_dir / "piper"
        vdir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "-m", "piper.download_voices", "--download-dir", str(vdir), *voices]
        )
        if r.returncode != 0:
            typer.echo("warning: piper voice download failed; espeak-ng will be used", err=True)
        else:
            typer.echo(f"piper voices in {vdir}: {', '.join(voices)}")


@app.command("init-config")
def init_config(
    path: Path = typer.Argument(None, help="default: $POSTURE_COACH_HOME/config.yaml"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Write a config.yaml you can edit (copy of the defaults)."""
    import yaml

    path = path or home() / "config.yaml"
    if path.exists() and not force:
        typer.echo(f"{path} exists (use --force to overwrite)")
        raise typer.Exit(1)
    from importlib import resources

    text = (
        resources.files("stretch_posture_coach").joinpath("default_config.yaml").read_text(encoding="utf-8")
    )
    yaml.safe_load(text)  # sanity
    path.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
