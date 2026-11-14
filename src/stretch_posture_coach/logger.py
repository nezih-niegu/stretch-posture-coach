"""Session logging: decimated per-frame metrics (CSV) + coaching events (JSONL)."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from .assessment import Assessment

COLUMNS = [
    "t",
    "wall_time",
    "present",
    "posture",
    "rula",
    "rula_risk",
    "reba",
    "reba_risk",
    "static_s",
    "active_issues",
    "neck_flexion",
    "neck_side_bend",
    "neck_twist",
    "head_forward_cm",
    "trunk_flexion",
    "trunk_side_bend",
    "trunk_twist",
    "shoulder_tilt_cm",
    "l_upper_arm_elevation",
    "r_upper_arm_elevation",
    "l_elbow_flexion",
    "r_elbow_flexion",
    "l_wrist_deviation",
    "r_wrist_deviation",
    "l_hip_flexion",
    "r_hip_flexion",
    "l_knee_flexion",
    "r_knee_flexion",
    "l5s1_moment_nm",
    "l5s1_compression_n",
    "l5s1_shear_n",
    "neck_moment_nm",
    "neck_load_kg",
    "max_shoulder_moment_nm",
    "head_tilt_rad",
    "distance_m",
    "distance_source",
    "follow_action",
]


class SessionLogger:
    def __init__(self, sessions_dir: str | Path, cfg: dict, log_hz: float = 5.0, name: str | None = None):
        stamp = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(sessions_dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self._csv_f = open(self.dir / "metrics.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=COLUMNS, extrasaction="ignore")
        self._csv.writeheader()
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._period = 1.0 / max(log_hz, 0.1)
        self._last = -1e9
        self._rows = 0

    def log(self, a: Assessment, head_tilt: float | None = None, follow=None) -> None:
        if a.t - self._last < self._period:
            return
        self._last = a.t
        row = {
            "t": round(a.t, 3),
            "wall_time": round(time.time(), 3),
            "present": int(a.present),
            "head_tilt_rad": None if head_tilt is None else round(head_tilt, 3),
        }
        if follow is not None:
            row.update(
                {
                    "distance_m": follow.distance,
                    "distance_source": follow.source,
                    "follow_action": follow.action,
                }
            )
        if a.present:
            row.update({k: a.metrics.get(k) for k in COLUMNS if k in a.metrics})
            row["posture"] = a.metrics.get("posture")
            if a.rula:
                row["rula"], row["rula_risk"] = a.rula["score"], a.rula["risk"]
            if a.reba:
                row["reba"], row["reba_risk"] = a.reba["score"], a.reba["risk"]
            row["static_s"] = round(a.static_for_s, 1)
            row["active_issues"] = "|".join(s.rule.id for s in a.active)
        self._csv.writerow(row)
        self._rows += 1
        if self._rows % 25 == 0:
            self._csv_f.flush()

    def event(self, ev: dict) -> None:
        self._events.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
        self._events.flush()

    def close(self) -> None:
        self._csv_f.close()
        self._events.close()
