"""The live, registered world model shared by the pipeline, the VR view and the scene recorder.

base pose   lidar SLAM (``slam2d``) + wheel odometry; odometry / follow dead-reckoning without a lidar
camera pose base pose ⊕ head joints ⊕ floor-plane correction (``registration``)
LiveMap     colored voxel map in the map frame, fused from the head D435i at ``vr.map_hz``:
            people are masked out, and voxels we can now see *through* are carved away, so furniture
            stays and people/moving things don't leave ghosts. Streams incremental deltas to the headset.
heat        mean REBA per 25 cm floor cell where the person was
captions    fall checks/alarms, coaching moments and improvements, pinned where they happened
track       <session>/track.jsonl: registered person + robot positions (for the report's SLAM view)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .insights import HeatGrid, caption_from_event
from .registration import FloorAligner, map_T_cam
from .scene import HeadKinematics, _pack, _unpack, backproject, person_mask, torso_point_cam

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------ voxel map
class LiveMap:
    """Sparse colored voxel map with hit counts, see-through carving and versioned deltas."""

    def __init__(self, voxel: float = 0.05, min_hits: int = 2, max_voxels: int = 400_000, carve: bool = True):
        self.voxel = voxel
        self.min_hits = min_hits
        self.max_voxels = max_voxels
        self.carve = carve
        self.keys = np.zeros(0, np.int64)
        self.csum = np.zeros((0, 3), np.float32)
        self.ccount = np.zeros(0, np.float32)
        self.hits = np.zeros(0, np.float32)
        self.ver = np.zeros(0, np.int64)  # version at which the voxel's shown state/color last changed
        self.shown = np.zeros(0, bool)
        self.version = 0
        self._deleted: list[tuple[int, np.ndarray]] = []  # (version, keys) of shown voxels removed
        self.oldest_delta = 0
        self.lock = threading.Lock()

    def __len__(self):
        return int(self.shown.sum())

    def integrate(
        self,
        pts: np.ndarray,
        cols: np.ndarray,
        T_map_cam: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        intr4=None,
        zmax: float = 4.5,
    ) -> None:
        """Add map-frame points (N,3) with colors (N,3 uint8). With the frame's depth + pose, also carve
        voxels in front of the observed surface (seen through = no longer there)."""
        with self.lock:
            self.version += 1
            v = self.version
            if self.carve and T_map_cam is not None and depth is not None and len(self.keys):
                self._carve(T_map_cam, depth, intr4, zmax, v)
            if len(pts):
                k = _pack(np.floor(pts / self.voxel).astype(np.int64))
                uk, inv = np.unique(k, return_inverse=True)
                inv = inv.reshape(-1)
                cnt = np.bincount(inv, minlength=len(uk)).astype(np.float32)
                cs = np.stack([np.bincount(inv, weights=cols[:, j], minlength=len(uk)) for j in range(3)], 1)
                pos = np.searchsorted(self.keys, uk)
                ex = pos < len(self.keys)
                ex[ex] = self.keys[pos[ex]] == uk[ex]
                # existing voxels
                pe = pos[ex]
                self.csum[pe] = self.csum[pe] * 0.9 + cs[ex].astype(np.float32)  # slow color update
                self.ccount[pe] = self.ccount[pe] * 0.9 + cnt[ex]
                self.hits[pe] = np.minimum(self.hits[pe] + 1, 30)
                # new voxels
                nk = uk[~ex]
                if len(nk):
                    ins = pos[~ex]
                    self.keys = np.insert(self.keys, ins, nk)
                    self.csum = np.insert(self.csum, ins, cs[~ex].astype(np.float32), axis=0)
                    self.ccount = np.insert(self.ccount, ins, cnt[~ex])
                    self.hits = np.insert(self.hits, ins, np.ones(len(nk), np.float32))
                    self.ver = np.insert(self.ver, ins, np.full(len(nk), v))
                    self.shown = np.insert(self.shown, ins, np.zeros(len(nk), bool))
                # touched voxels that cross the visibility threshold (or change color while shown)
                touched = np.searchsorted(self.keys, uk)
                newly = (~self.shown[touched]) & (self.hits[touched] >= self.min_hits)
                self.shown[touched[newly]] = True
                self.ver[touched[newly]] = v
                recolor = self.shown[touched] & ~newly & (self.hits[touched] < 6)  # still settling
                self.ver[touched[recolor]] = v
            if len(self.keys) > self.max_voxels:
                self._evict(v)

    def _carve(self, T, depth, intr4, zmax, v):
        fx, fy, cx, cy = intr4
        h, w = depth.shape
        c = (_unpack(self.keys) + 0.5) * self.voxel
        Tinv_R = T[:3, :3].T
        p = (c - T[:3, 3]) @ Tinv_R.T  # optical frame
        z = p[:, 2]
        front = (z > 0.3) & (z < zmax)
        if not front.any():
            return
        idx = np.nonzero(front)[0]
        u = np.round(p[idx, 0] / z[idx] * fx + cx).astype(np.int64)
        r = np.round(p[idx, 1] / z[idx] * fy + cy).astype(np.int64)
        ok = (u >= 0) & (u < w) & (r >= 0) & (r < h)
        idx, u, r = idx[ok], u[ok], r[ok]
        d = depth[r, u]
        margin = 0.10 + 0.015 * z[idx] ** 2  # D435 depth noise grows ~ z^2
        seen_through = (d > 0.3) & (d > z[idx] + margin)
        k = idx[seen_through]
        if not len(k):
            return
        self.hits[k] -= 1.5
        gone = k[self.hits[k] <= 0]
        if len(gone):
            self._remove(gone, v)

    def _remove(self, idx: np.ndarray, v: int):
        was = idx[self.shown[idx]]
        if len(was):
            self._deleted.append((v, self.keys[was].copy()))
        keep = np.ones(len(self.keys), bool)
        keep[idx] = False
        for name in ("keys", "csum", "ccount", "hits", "ver", "shown"):
            setattr(self, name, getattr(self, name)[keep])

    def _evict(self, v: int):
        """Drop the least-confirmed, least-recently-seen voxels."""
        n_drop = len(self.keys) - int(self.max_voxels * 0.9)
        score = self.hits * 1e6 + self.ver
        drop = np.argpartition(score, n_drop)[:n_drop]
        self._remove(drop, v)

    def remove_column(self, x: float, y: float, radius: float, zlo: float = 0.1, zhi: float = 2.1):
        """Carve a vertical column (e.g. where a person stands)."""
        with self.lock:
            if not len(self.keys):
                return
            c = (_unpack(self.keys) + 0.5) * self.voxel
            m = (np.hypot(c[:, 0] - x, c[:, 1] - y) < radius) & (c[:, 2] > zlo) & (c[:, 2] < zhi)
            if m.any():
                self.version += 1
                self._remove(np.nonzero(m)[0], self.version)

    def delta(self, since: int, max_items: int = 120_000) -> dict:
        """Changes after version ``since`` (or a full snapshot if ``since`` is 0 / too old).

        Returns {version, full, idx (N,3 int16 voxel indices), rgb (N,3 u8), del (M,3 int16)} plus
        ``complete``: False if more remain (the client keeps asking with the returned version)."""
        with self.lock:
            full = since <= 0 or since < self.oldest_delta
            sel = self.shown & (self.ver > (0 if full else since))
            idx = np.nonzero(sel)[0]
            complete = True
            if len(idx) > max_items:
                order = np.argsort(self.ver[idx], kind="stable")[:max_items]
                idx = idx[order]
                version = int(self.ver[idx].max())
                # all voxels at that version must be included for the cursor to be exact
                last = np.nonzero(sel & (self.ver == version))[0]
                idx = np.union1d(idx, last)
                complete = False
            else:
                version = self.version
            keys = _unpack(self.keys[idx])
            rgb = np.clip(self.csum[idx] / np.maximum(self.ccount[idx], 1e-6)[:, None], 0, 255).astype(
                np.uint8
            )
            dels = []
            if not full:
                dels = [k for (vv, k) in self._deleted if since < vv <= version]
            dk = _unpack(np.concatenate(dels)) if dels else np.zeros((0, 3), np.int64)
            # trim the deletion log (clients older than this get a full snapshot)
            if len(self._deleted) > 400:
                self.oldest_delta = self._deleted[-300][0]
                self._deleted = self._deleted[-300:]
        return {
            "version": version,
            "full": full,
            "complete": complete,
            "idx": np.clip(keys, -32768, 32767).astype("<i2"),
            "rgb": rgb,
            "del": np.clip(dk, -32768, 32767).astype("<i2"),
        }

    def clear(self) -> None:
        """Forget everything (new survey). Connected headsets get a full (empty) snapshot next."""
        with self.lock:
            self.keys = np.zeros(0, np.int64)
            self.csum = np.zeros((0, 3), np.float32)
            self.ccount = np.zeros(0, np.float32)
            self.hits = np.zeros(0, np.float32)
            self.ver = np.zeros(0, np.int64)
            self.shown = np.zeros(0, bool)
            self.version += 1
            self.oldest_delta = self.version
            self._deleted = []

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            c = (_unpack(self.keys[self.shown]) + 0.5) * self.voxel
            rgb = self.csum[self.shown] / np.maximum(self.ccount[self.shown], 1e-6)[:, None]
        return c, np.clip(rgb, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------ world
class LiveWorld:
    """Owns registration (lidar SLAM + floor) and the live map/heat/captions. Thread-safe getters."""

    def __init__(self, cfg: dict, head=None, follower=None, lidar=None):
        self.cfg = cfg
        self.head, self.follower = head, follower
        sc = cfg.get("scene", {})
        vc = cfg.get("vr", {})
        lc = cfg.get("lidar", {})
        self.kin = HeadKinematics.from_cfg(cfg)
        self.aligner = FloorAligner(self.kin, sc)
        from .slam2d import Localizer

        self.loc = Localizer(lc)
        self.lidar = lidar
        if self.lidar is None and lc.get("enabled", "auto") not in (False, "false", "off"):
            from .lidar import open_lidar

            self.lidar = open_lidar(lc, on_scan=self._on_scan)
        elif self.lidar is not None:
            self.lidar.on_scan = self._on_scan
        self.map = LiveMap(
            float(vc.get("map_voxel_m", 0.05)),
            int(vc.get("map_min_hits", 2)),
            int(vc.get("map_max_voxels", 400_000)),
            bool(vc.get("map_carve", True)),
        )
        self.map_period = 1.0 / max(float(vc.get("map_hz", 2.0)), 0.1)
        self.map_stride = max(1, int(vc.get("map_stride", 4)))
        self.max_range = float(vc.get("max_range_m", 4.5))
        self.person_radius = float(vc.get("map_person_radius_m", 0.3))
        self.heat = HeatGrid(float(vc.get("heat_cell_m", 0.25)))
        from .cbf import CBFSafety

        self.cbf = CBFSafety(cfg.get("safety", {}))
        from .scenegraph import SceneGraph, SemanticDetector

        gc = cfg.get("scene_graph", {})
        self.graph_enabled = bool(gc.get("enabled", True))
        self.graph = SceneGraph(gc)
        self.graph_period = float(gc.get("rebuild_s", 5.0))
        self._last_graph = -1e9
        self._map_changed = False
        self.detector = None
        if self.graph_enabled and gc.get("detector", "auto") not in (False, "false", "off"):
            from .scenegraph import default_detector_path

            self.detector = SemanticDetector(gc.get("detector_model") or default_detector_path(),
                                             float(gc.get("detector_score", 0.4)))
            if not self.detector.ok:
                self.detector = None
        self.detector_stride = max(1, int(gc.get("detector_every_keyframes", 3)))
        self._kf = 0
        self.captions: list[dict] = []
        self.captions_version = 0
        self._caption_id = 0
        self._coached: dict[str, float] = {}
        self._last_map = -1e9
        self._last_track = -1e9
        self.track_period = 1.0 / max(float(cfg.get("logging", {}).get("log_hz", 5)), 0.5)
        self._track_f = None
        self.last = {"T": np.eye(4), "base": (0.0, 0.0, 0.0), "person": None, "t": 0.0}
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._running = True
        self._thread = threading.Thread(target=self._map_loop, daemon=True, name="live-map")
        self._thread.start()
        if self.graph_enabled:
            threading.Thread(target=self._graph_loop, daemon=True, name="scene-graph").start()

    # ---------------------------------------------------------------- poses
    def odom(self):
        h = self.head
        if h is not None and getattr(h, "base_x", None) is not None:
            return (h.base_x, h.base_y or 0.0, h.base_theta or 0.0)
        if self.follower is not None:
            return (self.follower.x, self.follower.y, self.follower.theta)
        return (0.0, 0.0, 0.0)

    def _on_scan(self, scan):
        try:
            self.loc.on_scan(scan, self.odom())
        except Exception as e:
            log.debug("slam: %s", e)

    def base_pose(self):
        odom = self.odom()
        return self.loc.pose(odom) if self.loc.enabled else odom

    @property
    def lidar_ok(self) -> bool:
        return self.loc.enabled

    # ---------------------------------------------------------------- per frame (pipeline thread)
    def update(self, ts, frame, depth, intr, pan, tilt, pose=None, min_vis=0.5, assessment=None) -> dict:
        base = self.base_pose()
        T = map_T_cam(base, pan or 0.0, tilt or 0.0, self.aligner)
        person = None
        pc = torso_point_cam(pose, depth, intr, min_vis) if pose is not None else None
        if pc is not None:
            person = (T @ np.array([*pc, 1.0]))[:3]
            self._last_person = person
        self.last = {"T": T, "base": base, "person": person, "t": ts, "pan": pan or 0.0, "tilt": tilt or 0.0}
        reba = None
        if assessment is not None and assessment.present and assessment.reba:
            reba = assessment.reba.get("score")
        if self.graph_enabled and ts - getattr(self, "_last_graph_person", -1e9) >= self.track_period:
            self._last_graph_person = ts
            m = assessment.metrics if assessment is not None and assessment.present else {}
            incl = m.get("trunk_inclination") if m else None
            self.last["scene"] = self.graph.update_person(
                person, None, m.get("posture") if m else None, cam_xyz=T[:3, 3],
                lying=incl is not None and incl >= 55, metrics=m,
            )
            self.graph.update_robot(base, T[:3, 3])
        elif "scene" in getattr(self, "_prev_last", {}):
            self.last["scene"] = self._prev_last["scene"]
        self._prev_last = self.last
        if person is not None and ts - self._last_track >= self.track_period:
            self._last_track = ts
            if reba is not None:
                self.heat.add(person[0], person[1], reba)
            self._write_track(ts, person, base, assessment)
        if depth is not None and ts - self._last_map >= self.map_period:
            self._last_map = ts
            mask = person_mask(pose, depth.shape[:2], min_vis) if pose is not None else None
            try:
                self._q.put_nowait((frame, depth, intr, base, mask, pan or 0.0, tilt or 0.0, person))
            except queue.Full:
                pass  # map thread busy: skip this keyframe
        return self.last

    def _map_loop(self):
        while self._running:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._integrate(*item)
                self._map_changed = True
            except Exception as e:  # never kill the map thread
                log.debug("live map: %s", e)

    def _graph_loop(self):
        """Own thread, so re-segmenting never delays keyframe integration."""
        import time as _time

        while self._running:
            _time.sleep(min(1.0, self.graph_period))
            try:
                self._maybe_rebuild_graph()
            except Exception as e:
                log.debug("scene graph: %s", e)

    def _maybe_rebuild_graph(self, force: bool = False) -> None:
        """Re-segment the object layer from the live map (scene-graph thread, every `rebuild_s`)."""
        import time as _time

        now = _time.monotonic()
        if not self.graph_enabled or not (force or (self._map_changed and now - self._last_graph >= self.graph_period)):
            return
        self._last_graph, self._map_changed = now, False
        centers, _ = self.map.snapshot()
        if len(centers) > 200:
            self.graph.rebuild_static(centers)

    def _integrate(self, frame, depth, intr, base, mask, pan, tilt, person):
        h, w = depth.shape[:2]
        s = self.map_stride
        d = cv2.resize(depth, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
        sx, sy = intr.w / w * s, intr.h / h * s
        intr4 = (intr.fx / sx, intr.fy / sy, (intr.cx + 0.5) / sx - 0.5, (intr.cy + 0.5) / sy - 0.5)
        keep = None
        if mask is not None and mask.any():
            keep = cv2.resize(mask, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST) == 0
        pts, rc = backproject(d, intr4, 1, keep, 0.3, self.max_range)
        # floor-plane registration from this frame (corrects the tilt/roll/height used for later frames)
        self.aligner.observe(pts, pan, tilt)
        if not len(pts):
            return
        T = map_T_cam(base, pan, tilt, self.aligner)  # with the floor correction just updated
        pw = pts @ T[:3, :3].T + T[:3, 3]
        small = cv2.resize(frame, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
        cols = small[rc[:, 0], rc[:, 1], ::-1].astype(np.float32)
        ok = pw[:, 2] > -0.3  # below-floor outliers
        dcarve = d.copy()
        if keep is not None:
            dcarve[~keep] = 0  # don't carve with person pixels (they're in front of everything)
        self.map.integrate(pw[ok], cols[ok], T, dcarve, intr4, self.max_range)
        if person is not None and self.person_radius > 0:
            self.map.remove_column(person[0], person[1], self.person_radius)
        self._kf += 1
        if self.detector is not None and self._kf % self.detector_stride == 0:
            from .scenegraph import backproject_box

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for name, score, box in self.detector.detect(rgb):
                xyz = backproject_box(box, depth, [intr.fx, intr.fy, intr.cx, intr.cy, intr.w, intr.h], T, frame.shape)
                if xyz is not None:
                    self.graph.add_detection(name, xyz, score)

    def reset_registration(self) -> None:
        """Start a fresh map frame: new lidar SLAM map (map = base frame now), empty voxel map, and the floor
        correction re-measured from scratch. Called by the 360° survey at the start of every session."""
        from .slam2d import Localizer

        old = self.loc
        self.loc = Localizer(self.cfg.get("lidar", {}))
        self.loc.enabled = old.enabled
        with self.aligner.lock:
            self.aligner.n = 0
        self.map.clear()
        from .scenegraph import SceneGraph

        self.graph = SceneGraph(self.cfg.get("scene_graph", {}))  # objects are re-segmented from the new map
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        log.info("registration reset: new map frame, floor correction will be re-measured")

    # ---------------------------------------------------------------- events -> captions
    def event(self, ev: dict, metrics: dict | None = None) -> dict | None:
        kind = ev.get("kind")
        if kind == "issue" and ev.get("issue"):
            from .insights import ISSUE_METRIC

            mk = ISSUE_METRIC.get(ev["issue"])
            if mk and metrics is not None and metrics.get(mk[0]) is not None:
                v = metrics[mk[0]]
                self._coached[ev["issue"]] = abs(v) if mk[2] else v
        pos = self.last.get("person")
        if pos is None and kind == "fall":  # not seen now: put it where they were last seen
            pos = getattr(self, "_last_person", None)
        c = caption_from_event(
            ev, None if pos is None else [pos[0], pos[1], max(pos[2], 0.3)], metrics, self._coached
        )
        if c is None:
            return None
        if self.graph_enabled and c.get("pos") is not None:
            gev = self.graph.add_event(c.get("kind", kind or "event"), c["pos"], c.get("t", 0.0), c.get("title", ""))
            if gev is not None and gev.attrs.get("at") not in (None, "floor"):
                from .scenegraph import pretty

                lab, num = gev.attrs["at"].rsplit("_", 1)
                c["at"] = gev.attrs["at"]
                c["text"] = (c.get("text") or "") + f" · near {pretty(lab).replace('_', ' ')} {num}"
        self._caption_id += 1
        c["id"] = self._caption_id
        if kind == "fall" and ev.get("state") == "ok":  # replace the open fall caption
            self.captions = [x for x in self.captions if not x["kind"].startswith("fall_")]
        if kind == "fall" and ev.get("state") == "alarm":
            self.captions = [x for x in self.captions if x["kind"] != "fall_check"]
        self.captions.append(c)
        self.captions = self.captions[-30:]
        from .insights import stack_captions

        for x in self.captions:
            x.pop("lift", None)
        stack_captions(self.captions)
        self.captions_version += 1
        return c

    # ---------------------------------------------------------------- sessions
    def new_session(self, session_dir: Path | None, old_dir: Path | None = None) -> None:
        if old_dir is not None:
            self.save_occupancy(old_dir)
            self.save_scene_graph(old_dir)
        with self.graph.lock:
            self.graph.events = []
        if self._track_f:
            self._track_f.close()
            self._track_f = None
        self.heat = HeatGrid(self.heat.cell)
        self.captions = []
        self.captions_version += 1
        self._coached = {}
        if session_dir is not None:
            self._track_f = open(Path(session_dir) / "track.jsonl", "a", encoding="utf-8")

    def _write_track(self, ts, person, base, a):
        if self._track_f is None:
            return
        rec = {
            "t": round(ts, 3),
            "wall": round(time.time(), 3),
            "x": round(float(person[0]), 3),
            "y": round(float(person[1]), 3),
            "z": round(float(person[2]), 3),
            "robot": [round(float(v), 3) for v in base],
        }
        if a is not None and a.present:
            rec["reba"] = a.reba and a.reba.get("score")
            rec["rula"] = a.rula and a.rula.get("score")
            rec["posture"] = a.metrics.get("posture")
        try:
            self._track_f.write(json.dumps(rec) + "\n")
            self._track_f.flush()
        except ValueError:
            pass

    def save_occupancy(self, session_dir: Path) -> None:
        """Lidar occupancy grid of the session (report background)."""
        if not self.loc.enabled or self.loc.grid.scans == 0:
            return
        try:
            img, meta = self.loc.grid.image()
            d = Path(session_dir) / "scene"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "occupancy.png"), img)
            (d / "occupancy.json").write_text(json.dumps({**meta, **self.loc.status()}), encoding="utf-8")
        except Exception as e:
            log.debug("save occupancy: %s", e)

    def obstacle_points(self):
        """Latest lidar scan in the base frame (None without a lidar)."""
        return self.loc.latest_points if self.loc.enabled else None

    def barrier_lines(self) -> list[dict]:
        if not self.loc.enabled:
            return []
        return self.cbf.barrier_lines(self.loc.latest_points, self.loc.pose(self.odom()))

    def fall_where(self) -> str:
        """Where a fall check happens, from the scene graph: the furniture under / next to the person and what
        hides them from the camera. When they are out of view, the hips are assumed near the floor (0.4 m)."""
        if not self.graph_enabled:
            return ""
        pos = self.last.get("person")
        if pos is None:
            pos = getattr(self, "_last_person", None)
        if pos is None:
            return ""
        cam = self.last["T"][:3, 3]
        ctx = self.graph.context(np.array(pos), 0.4, cam, lying=True)
        bits = []
        if ctx.get("support_id"):
            bits.append(f"on {ctx['support_id'].replace('_', ' ')}")
        elif ctx.get("near"):
            bits.append(f"near {ctx['near'][0]['id'].replace('_', ' ')}")
        if ctx.get("occluder_id"):
            bits.append(f"hidden behind {ctx['occluder_id'].replace('_', ' ')}")
        return ", ".join(bits)

    def scene_text(self) -> str:
        return self.graph.describe() if self.graph_enabled else ""

    def scene_graph_msg(self) -> dict:
        ctx = self.last.get("scene") or {}
        return {"type": "scene_graph", "version": self.graph.version, "boxes": self.graph.boxes(),
                "person": ctx.get("text", ""), "support": ctx.get("support_id"), "at": ctx.get("at"),
                "occluder": ctx.get("occluder_id"), "summary": self.graph.summary()}

    def save_scene_graph(self, session_dir: Path) -> None:
        if not self.graph_enabled:
            return
        try:
            self._maybe_rebuild_graph(force=True)
            d = Path(session_dir) / "scene"
            d.mkdir(parents=True, exist_ok=True)
            (d / "scene_graph_live.json").write_text(json.dumps(self.graph.to_json(), indent=1), encoding="utf-8")
        except Exception as e:
            log.debug("save scene graph: %s", e)

    def status(self) -> dict:
        return {"lidar": self.loc.status(), "floor": self.aligner.status(), "map_voxels": len(self.map),
                "scene_graph": self.graph.summary() if self.graph_enabled else None,
                "scene": (self.last.get("scene") or {}).get("text")}

    def close(self):
        self._running = False
        if self.lidar is not None:
            try:
                self.lidar.close()
            except Exception:
                pass
        if self._track_f:
            self._track_f.close()
            self._track_f = None
