"""360° mapping survey at the start of every session (mode SURVEY).

Why: every session starts from a freshly calibrated map instead of trusting an old one - the lidar SLAM map is
rebuilt (the map frame = where the robot stands now), the floor-plane correction of the head camera is
re-measured, and the room is scanned all around before anything moves on its own.

Phases (non-blocking: ``step()`` is called every frame by the pipeline):
  reset   new map frame (``LiveWorld.reset_registration``), head to the first sweep angle, announce it
  settle  ~1.5 s of lidar scans at rest (the lidar itself sees 360° every scan)
  sweep   head pans across its whole range (~315°) at a floor-seeing tilt, pausing at each step so RGB-D
          keyframes land in the map and the floor plane is measured
  spin    optional: the base turns 360° in 30° steps to see behind the mast - only if the CBF filter
          (lidar barriers) allows every step; otherwise skipped
  front   head back to the front (pan = follow.forward_pan_rad, tilt = robot.front_tilt_rad)
  check   pass/fail: enough matched lidar scans, at least one floor-plane fit, enough map voxels

The result is written to <session>/survey.json and the report.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Survey:
    def __init__(self, cfg: dict, head, world, say=None):
        sc = cfg.get("survey", {})
        self.c = sc
        self.head, self.world, self.say = head, world, say
        self.front_pan = float(cfg.get("follow", {}).get("forward_pan_rad", 0.0))
        self.front_tilt = float(cfg.get("robot", {}).get("front_tilt_rad", -0.15))
        self.tilt = float(sc.get("tilt_rad", -0.55))
        lo, hi = float(sc.get("pan_min_rad", -3.9)), float(sc.get("pan_max_rad", 1.6))
        self.pans = list(np.linspace(lo, hi, int(sc.get("steps", 10))))
        self.dwell = float(sc.get("dwell_s", 1.2))
        self.settle = float(sc.get("settle_s", 1.5))
        self.spin = str(sc.get("rotate_base", "auto")).lower()
        self.spin_step = math.radians(
            float(sc.get("spin_step_deg", 30))
        )  # robot_helper caps a step at 0.6 rad
        self.spin_dwell = float(sc.get("spin_dwell_s", 2.8))
        self.timeout = float(sc.get("timeout_s", 90))
        self.phase = "idle"
        self.t0 = self.t_phase = 0.0
        self.i = 0
        self.spun = 0.0
        self.result: dict | None = None
        self.session_dir: Path | None = None

    @property
    def running(self) -> bool:
        return self.phase not in ("idle", "done")

    def start(self, now: float, session_dir: Path | None = None) -> None:
        self.session_dir = session_dir
        self.world.reset_registration()
        self.phase, self.t0, self.t_phase, self.i, self.spun = "settle", now, now, 0, 0.0
        self.result = None
        self.head.look(self.pans[0], self.tilt, now)
        if self.say:
            self.say("survey_start")
        log.info("SURVEY: 360° mapping started (%d head steps, base spin: %s)", len(self.pans), self.spin)

    def _next(self, phase: str, now: float):
        self.phase, self.t_phase, self.i = phase, now, 0

    def step(self, now: float) -> str | None:
        """Advance; returns "done" / "failed" once, else None."""
        if not self.running:
            return None
        if now - self.t0 > self.timeout:
            return self._finish(now, "timeout")
        dt = now - self.t_phase
        if self.phase == "settle":
            if dt >= self.settle:
                self._next("sweep", now)
                self.head.look(self.pans[0], self.tilt, now)
        elif self.phase == "sweep":
            if dt >= self.dwell * (self.i + 1):
                self.i += 1
                if self.i >= len(self.pans):
                    self._next("spin" if self._may_spin() else "front", now)
                    if self.phase == "front":
                        self.head.look(self.front_pan, self.front_tilt, now)
                else:
                    self.head.look(self.pans[self.i], self.tilt, now)
        elif self.phase == "spin":
            if dt >= self.spin_dwell * self.i:
                if self.spun >= 2 * math.pi - 1e-3:
                    self._next("front", now)
                    self.head.look(self.front_pan, self.front_tilt, now)
                else:
                    pts = self.world.obstacle_points()
                    _, step, res = self.world.cbf.limit_step(0.0, self.spin_step, pts, dt=self.spin_dwell)
                    if step < 0.9 * self.spin_step:
                        log.warning(
                            "SURVEY: base spin stopped by a barrier (h_min %s) after %.0f°",
                            None if res.h_min is None else round(res.h_min, 2),
                            math.degrees(self.spun),
                        )
                        self._next("front", now)
                        self.head.look(self.front_pan, self.front_tilt, now)
                    else:
                        self.head.base_rotate(self.spin_step, compensate_head=False)
                        self.spun += self.spin_step
                        self.i += 1
        elif self.phase == "front":
            if dt >= 1.0:
                return self._finish(now, None)
        return None

    def _may_spin(self) -> bool:
        if self.spin in ("false", "off", "no", "0"):
            return False
        if not getattr(self.head, "has_base", False):
            return False
        if self.spin == "true":
            return True
        pts = self.world.obstacle_points()  # auto: only with a lidar, and only if the first step is clear
        if pts is None:
            return False
        _, step, _ = self.world.cbf.limit_step(0.0, self.spin_step, pts, dt=self.spin_dwell)
        return step >= 0.9 * self.spin_step

    def _finish(self, now: float, error: str | None) -> str:
        w = self.world
        st = w.status()
        lid = st["lidar"]
        checks = {
            "lidar": (not lid["enabled"]) or lid["matched"] >= 3,
            "floor": st["floor"]["floor_fits"] >= 1,
            "map": st["map_voxels"] >= int(self.c.get("min_voxels", 1500)),
        }
        ok = error is None and all(checks.values())
        self.result = {
            "ok": ok,
            "error": error or (None if ok else "failed: " + ", ".join(k for k, v in checks.items() if not v)),
            "duration_s": round(now - self.t0, 1),
            "head_steps": len(self.pans),
            "base_spin_deg": round(math.degrees(self.spun)),
            "checks": checks,
            **st,
        }
        self.phase = "done"
        log.info("SURVEY %s in %.0f s: %s", "OK" if ok else "FAILED", now - self.t0, self.result)
        if self.session_dir is not None:
            try:
                (Path(self.session_dir) / "survey.json").write_text(
                    json.dumps(self.result, indent=1), encoding="utf-8"
                )
            except OSError:
                pass
        if ok and self.say:
            self.say("survey_done")
        return "done" if ok else "failed"

    def abort(self) -> None:
        if self.running:
            log.info("SURVEY interrupted in phase %s", self.phase)
        self.phase = "idle"

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "step": self.i,
            "steps": len(self.pans),
            "result": self.result,
        }
