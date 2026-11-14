"""Configuration loading: packaged defaults deep-merged with a user YAML file and CLI overrides."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

# Relative data paths (models, sessions, calibration) resolve against POSTURE_COACH_HOME (set by the
# installed `posture-coach` wrapper to the repo folder) or, if unset, the current directory.
PATH_KEYS = [
    ("pose", "models_dir"),
    ("speech", "voices_dir"),
    ("logging", "sessions_dir"),
    ("analysis", "calibration_file"),
    ("vr", "certs_dir"),
]


def home() -> Path:
    return Path(os.environ.get("POSTURE_COACH_HOME", ".")).expanduser()


def default_user_config() -> Path | None:
    for cand in (Path("config.yaml"), home() / "config.yaml"):
        if cand.exists():
            return cand
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_defaults() -> dict[str, Any]:
    text = (
        resources.files("stretch_posture_coach").joinpath("default_config.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load defaults, then ``path`` (or ./config.yaml if it exists), then ``overrides``."""
    cfg = load_defaults()
    user_path = path if path is not None else default_user_config()
    if user_path is not None:
        user_path = Path(user_path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {user_path}")
        cfg = _deep_merge(cfg, yaml.safe_load(user_path.read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    for section, key in PATH_KEYS:
        val = cfg.get(section, {}).get(key)
        if val and not Path(str(val)).expanduser().is_absolute():
            cfg[section][key] = str(home() / str(val))
    return cfg


def set_dotted(overrides: dict[str, Any], dotted: str, value: Any) -> None:
    """``set_dotted(d, "llm.model", "qwen2:7b")`` -> ``d["llm"]["model"] = "qwen2:7b"``."""
    node = overrides
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_override(item: str) -> tuple[str, Any]:
    """Parse ``key.sub=value`` with YAML typing for the value (numbers, bools, null)."""
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got: {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)
