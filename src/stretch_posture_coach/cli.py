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
    from .pipeline import Pipeline

    session = Pipeline(cfg).run(duration_s=duration)
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
    _localize(cfg, norteno)
    _run_pipeline(cfg, duration, report=coach)


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
) -> None:
    """Download MediaPipe pose models (lite/full/heavy) and Piper voices."""
    from .pose import download_models

    models_dir = models_dir or home() / "models"
    for p in download_models(models_dir):
        typer.echo(f"pose model: {p}")
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
