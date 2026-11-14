# stretch-posture-coach

Full-body biomechanical posture analysis with **MediaPipe Pose** on a **Hello Robot Stretch 3**. Stretch
watches you with its head camera, estimates joint angles, **RULA / REBA** ergonomic risk and spinal loads
(L5/S1 compression, neck load). When a bad posture lasts too long, it **tells you out loud** how to fix it,
using a local LLM (**Ollama + qwen2**) and an offline neural voice (Piper, with espeak-ng as fallback).

Everything runs on the robot (NUC12 i5-1240P). No cloud.

```
 D435i head camera ──► MediaPipe Pose Landmarker (33 3-D landmarks, ~25 FPS)
                              │
             gravity from head tilt (stretch_body)
                              ▼
        biomech.py   joint angles · L5/S1 & neck load model
        ergonomics.py RULA + REBA (worse side)
        assessment.py smoothing · static-posture timer · sustained-issue rules
                              │
          ┌───────────────────┼──────────────────────────┐
          ▼                   ▼                          ▼
   coach.py ──► Ollama qwen2 ──► Piper TTS ──► Stretch speaker
   head tracking (pan/tilt keeps you centered)
   web viewer :8080 · CSV/JSONL session log · HTML report
```

## Install (one command)

On the robot (`hello-robot@stretch-se3-XXXX`):

```bash
cd ~/Projects && git clone <YOUR-REPO-URL> stretch-posture-coach && cd stretch-posture-coach && ./install.sh
```

`install.sh` is idempotent and does everything:

1. apt: `espeak-ng pulseaudio-utils alsa-utils libportaudio2` (asks for sudo once)
2. installs **uv** if missing, then `uv sync` (Python 3.11 in an isolated `.venv`; the robot's own Python is untouched)
3. puts a `posture-coach` command in `~/.local/bin` that works from any folder
4. downloads MediaPipe pose models (lite/full/heavy) and the Piper voice
5. creates `config.yaml`
6. installs **Ollama**, starts it, and pulls **qwen2:1.5b**
7. runs `posture-coach check` (camera, model, LLM, speaker, head) and Stretch says hello

Options: `./install.sh --model qwen2:7b`, `--spanish` (Spanish voice and coaching), `--no-ollama`, `--no-apt`.

> **Before running**, free the hardware: if the ROS 2 driver or a RealSense node is running, it holds the
> camera and the head motors. Stop it (Ctrl+C in that terminal) or run `stretch_free_robot_process.py`.

## Use

```bash
posture-coach calibrate        # once per person: sit/stand tall for 8 s -> calibration.json
posture-coach run              # coach me! Ctrl+C to stop -> session report
```

Open **http://stretch-se3-XXXX.local:8080/** (or `http://<robot-ip>:8080/`) on your laptop to see the live
skeleton, angles, RULA/REBA and what Stretch said. Over SSH no window opens; the web viewer replaces it.

Useful flags:

| Flag | Meaning |
|---|---|
| `--mass 80` | your body mass in kg (load model) |
| `--language Spanish` | Stretch coaches in Spanish (use `./install.sh --spanish` for the Spanish voice) |
| `--model qwen2:7b --host http://<laptop-ip>:11434` | use a bigger model on your laptop's Ollama |
| `--no-llm` | rule-based messages only (no Ollama) |
| `--no-voice` | print messages instead of speaking |
| `--no-robot` | don't move the head |
| `--source webcam:6` / `--source video.mp4` | other camera, or analyze a recorded video |
| `--calibrate` | calibrate first, then run |
| `-s section.key=value` | override any setting in `config.yaml`, e.g. `-s thresholds.neck_flexion=15` |

Other commands: `posture-coach check`, `posture-coach say "hello"`, `posture-coach report [sessions/<stamp>]`,
`posture-coach download-models --voice es_MX-claude-high`, `posture-coach init-config`.

### Placement

* Stand or sit **2-3 m** from Stretch with your **whole body** in the frame. Head tracking keeps you centered.
* A **side (profile) view** gives the most accurate neck/trunk flexion; a front view is best for side bend,
  shoulder tilt and twist. A 45° view is a good compromise.
* Run `posture-coach calibrate` in your best neutral posture. Angles are then measured relative to *your*
  neutral, which removes landmark bias (ears vs. shoulders differ between people).

## What it measures

| Metric | How |
|---|---|
| Neck flexion / extension, side bend, twist | ear-midpoint vs shoulder-midpoint, relative to the trunk |
| Trunk flexion, side bend, twist | shoulder-mid vs hip-mid against gravity; shoulder line vs hip line |
| Shoulder tilt / shrug | shoulder height difference; ear-to-shoulder distance vs calibration |
| Upper arm elevation, abduction; elbow, wrist, hip, knee flexion | 3-D segment angles (per side) |
| Posture state | sitting / standing from hip and knee angles |
| **RULA** (1-7) and **REBA** (1-15) | standard tables, worse side, +1 when a posture is held > 60 s |
| **L5/S1** moment, compression, shear | static segment model (Winter/Dempster mass fractions, 5 cm erector arm) + `hand_load_kg` |
| Neck moment / equivalent load | head mass (8.1 % BW) × forward offset of the head |

Gravity is taken from the **head tilt joint**, so angles stay correct when Stretch looks down at you.

Coaching rules (thresholds in `config.yaml → thresholds`) fire only after a violation lasts `hold_s`
(default 6 s), with hysteresis. Stretch waits `min_interval_s` between messages, does not repeat the same
issue within `repeat_interval_s`, praises you when you fix it, and reminds you to take a movement break
every `break_reminder_min`. If Ollama is down or slow, it falls back to built-in messages.

### Outputs

Each run creates `sessions/<timestamp>/`:
`metrics.csv` (5 Hz), `events.jsonl` (everything Stretch said and why), `config.yaml`, `summary.json`, and
`report.html` (time in RULA/REBA risk bands, % time over each limit, charts).

## Configuration

Edit `config.yaml` (only the keys you change matter; the defaults are documented in
`src/stretch_posture_coach/default_config.yaml`). Common tweaks:

```yaml
subject: {body_mass_kg: 78, hand_load_kg: 0, arms_supported: true}
coach:   {language: Spanish, user_name: Ana, min_interval_s: 40}
llm:     {model: qwen2:1.5b, host: http://localhost:11434}
speech:  {piper_voice: es_MX-claude-high, espeak_voice: es-419}
robot:   {head_tracking: true, pan_sign: -1, tilt_sign: -1}
camera:  {rotate: cw90}
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No RealSense camera found` / `Could not start` | Something else holds the D435i (ROS 2 driver, realsense node). Stop it or run `stretch_free_robot_process.py`. |
| Snapshot `check_snapshot.jpg` is upside-down/sideways | `camera.rotate: ccw90` (or `none`, `180`). |
| Head turns **away** from you | flip `robot.pan_sign` (or `tilt_sign`) to `+1`. |
| `head.startup() failed` | the ROS driver is using the motors; free the robot (see above), or run with `--no-robot`. |
| No sound over SSH | `posture-coach say "test"`. The app sets `XDG_RUNTIME_DIR` for you; also check the volume with `pactl list sinks short` / `alsamixer`, and that a desktop session is logged in. |
| Messages are slow | Normal first time (model load). Use `qwen2:0.5b`, or run Ollama on your laptop: `--host http://<laptop-ip>:11434`. On the laptop, start it with `OLLAMA_HOST=0.0.0.0 ollama serve`. |
| Angles look off | calibrate; prefer a side view; make sure your whole body is visible. |

## Limitations

This is an **ergonomic coaching tool, not a clinical instrument**. Monocular pose is noisy in depth
(especially wrist and twist angles), RULA/REBA force, coupling and wrist-twist inputs are assumed from
config, and the load model is a static 3-D approximation (no dynamics, no muscle co-contraction).

## Development

```bash
uv sync
uv run pytest            # 37 tests: geometry, RULA/REBA tables, load model, end-to-end pipeline with fake camera/LLM
uv run ruff check .
```

Layout: `src/stretch_posture_coach/` contains `camera.py`, `pose.py`, `biomech.py`, `ergonomics.py`,
`assessment.py`, `coach.py`, `llm.py`, `speech.py`, `robot.py` + `robot_helper.py`, `visualize.py`,
`stream.py`, `logger.py`, `report.py`, `pipeline.py` and `cli.py`.

`robot_helper.py` runs on the robot's **system** Python (`/usr/bin/python3`, which has Hello Robot's
`stretch_body`) as a small subprocess, so the uv environment never conflicts with the robot's packages.
