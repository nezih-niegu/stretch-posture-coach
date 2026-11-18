"""A layered 3-D scene graph of the room: room -> objects -> agents (person, robot) -> events.

Why: the voxel map says *where* things are but not *what* they are. The graph turns it into nodes and relations
the rest of the system can reason with:

  coaching   "sitting on chair 3 at table 1, neck flexed 28 deg" -> a desk-specific tip (raise the screen)
             instead of a generic one
  falls      lying on a sofa or bed is not "lying on the floor" (no 30 s floor check); a check names the
             furniture under the person and the furniture hiding them from the camera
  captions   events are attached to objects ("near table 1"), and the report aggregates posture per object
  VR         the clinician sees labelled boxes and the person's relation to them

Layers (after Armeni et al. 2019 and Rosinol et al. 2020, reduced to one room and a CPU budget):
  room     extent, floor area, ceiling
  object   segmented from the registered voxel map: 2-D footprint cells, height band, oriented box, top surface,
           geometric label; optional semantic votes from an object detector (MediaPipe EfficientDet-Lite, COCO)
  agent    person (torso point, hip height, posture) and robot (base pose, camera)
  event    fall checks/alarms/recoveries, coaching tips, improvements - each attached to an object or the floor

Segmentation (no learning needed): voxels between the floor and the ceiling are binned into 5 cm columns. Each
column's top height puts it in a band (low < 0.35 m, seat < 0.66, table < 1.0, tall < 1.6, wall above). Connected
components per band become objects; a small component in the table band touching a seat component is the
chair's backrest and is merged into it. Tall, thin, long components are walls.

Relations: in (object -> room), next_to (footprints within 0.25 m), at (chair -> table), on / sitting_on /
lying_on (person -> supporting object: footprint under the hips and hip height near its top surface), near
(person -> objects within 0.8 m), occluded_by (an object between the camera and the person), at (event ->
object), sees / faces (robot -> person).
"""

from __future__ import annotations

import json
import logging
import math
import struct
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# height bands of a column's top (m above the floor)
BANDS = (
    ("low", 0.08, 0.35),
    ("seat", 0.35, 0.66),
    ("table", 0.66, 1.0),
    ("tall", 1.0, 1.6),
    ("wall", 1.6, 9.0),
)
SUPPORT_LABELS = {"chair", "sofa", "bed", "table", "bench", "low_object"}
SEAT_LABELS = {"chair", "sofa", "bench"}
RECLINE_LABELS = {"sofa", "bed"}  # lying on these is not a fall onto the floor
# detector (COCO) class -> graph label
COCO_TO_LABEL = {
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",
    "bench": "bench",
    "tv": "screen",
    "laptop": "laptop",
    "potted plant": "plant",
    "refrigerator": "cabinet",
    "toilet": "toilet",
    "suitcase": "low_object",
    "backpack": "low_object",
}
PRETTY = {"low_object": "low object", "screen": "screen"}


def pretty(label: str) -> str:
    return PRETTY.get(label, label)


@dataclass
class Node:
    id: str
    layer: str  # room | object | agent | event
    label: str
    center: np.ndarray  # map frame, m
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))  # oriented box extents (l, w, h)
    yaw: float = 0.0
    top: float = 0.0  # top surface height (m) - what a person sits / leans on
    attrs: dict = field(default_factory=dict)
    cells: np.ndarray | None = None  # footprint cell keys (objects only)
    votes: Counter = field(default_factory=Counter)

    def to_json(self) -> dict:
        d = {
            "id": self.id,
            "layer": self.layer,
            "label": self.label,
            "center": [round(float(x), 3) for x in self.center],
        }
        if self.layer in ("object", "room"):
            d.update(
                size=[round(float(x), 3) for x in self.size],
                yaw=round(float(self.yaw), 4),
                top=round(self.top, 3),
            )
        if self.votes:
            d["votes"] = dict(self.votes)
        if self.attrs:
            d["attrs"] = self.attrs
        return d


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    attrs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"src": self.src, "rel": self.rel, "dst": self.dst}
        if self.attrs:
            d["attrs"] = self.attrs
        return d


# ------------------------------------------------------------------------------------------ segmentation
def _cell_keys(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xFFFFFFFF)


def segment_objects(
    points: np.ndarray,
    cell: float = 0.05,
    floor_z: float = 0.08,
    ceiling_z: float | None = None,
    min_area_m2: float = 0.04,
    max_z: float = 2.0,
) -> tuple[list[Node], dict]:
    """Segment registered map points (N x 3, z up, floor at 0) into object nodes with geometric labels."""
    P = np.asarray(points, float).reshape(-1, 3)
    if ceiling_z is None and len(P):
        ceiling_z = float(np.percentile(P[:, 2], 99.5))
    # furniture lives below 2 m; capping the columns there keeps ceilings, lamps and beams from becoming the "top"
    # of the columns under them (which would hide the furniture) whatever the ceiling estimate is
    zhi = min((ceiling_z - 0.15) if ceiling_z else 2.4, max_z)
    P = P[(P[:, 2] > floor_z) & (P[:, 2] < zhi)]
    room = {"ceiling_m": ceiling_z, "cell": cell}
    if len(P) < 20:
        return [], room
    ix = np.floor(P[:, 0] / cell).astype(np.int64)
    iy = np.floor(P[:, 1] / cell).astype(np.int64)
    x0, y0 = int(ix.min()) - 2, int(iy.min()) - 2
    W, H = int(ix.max()) - x0 + 3, int(iy.max()) - y0 + 3
    gx, gy = ix - x0, iy - y0
    flat = gy * W + gx
    count = np.bincount(flat, minlength=W * H)
    # column top; a column survives if it has >= 2 voxels or >= 3 occupied neighbours (a table top seen from
    # above is one voxel thick, a stray voxel has no neighbours)
    top = np.full(W * H, -1.0)
    np.maximum.at(top, flat, P[:, 2])
    top = top.reshape(H, W)
    count = count.reshape(H, W)
    neigh = cv2.filter2D(
        (count > 0).astype(np.float32), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT
    )
    top[~((count >= 2) | (neigh - 1 >= 3))] = -1.0
    occ = top > 0
    room.update(origin=[x0 * cell, y0 * cell], shape=[H, W])

    nodes: list[Node] = []
    band_lab = np.zeros((H, W), np.int32)
    comps: list[dict] = []
    k = 0
    kernel = np.ones((3, 3), np.uint8)
    for name, lo, hi in BANDS:
        base = occ & (top >= lo) & (top < hi)
        m = base.astype(np.uint8)
        if name != "wall":
            # bridge dropout holes in a surface - only through empty cells, never into another band
            closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel).astype(bool)
            m = (base | (closed & ~occ)).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        for i in range(1, n):
            ys, xs = np.nonzero(lab == i)
            parts = [(ys, xs)] if name == "wall" else _split_component(ys, xs, cell)
            for pys, pxs in parts:
                k += 1
                band_lab[pys, pxs] = k
                comps.append({"k": k, "band": name, "ys": pys, "xs": pxs})

    # merge chair backrests (small table/tall-band parts touching a seat part) into the seat
    by_k = {c["k"]: c for c in comps}
    for c in comps:
        if c["band"] not in ("table", "tall") or len(c["xs"]) * cell * cell > 0.08:
            continue
        mask = np.zeros((H, W), np.uint8)
        mask[c["ys"], c["xs"]] = 1
        ring = cv2.dilate(mask, kernel) & (mask == 0)
        neigh = Counter(band_lab[ring.astype(bool)].tolist())
        neigh.pop(0, None)
        seat = [kk for kk, _ in neigh.most_common() if by_k.get(kk, {}).get("band") == "seat"]
        if seat:
            s = by_k[seat[0]]
            s["ys"] = np.concatenate([s["ys"], c["ys"]])
            s["xs"] = np.concatenate([s["xs"], c["xs"]])
            s["backrest"] = True
            c["merged"] = True

    counts: Counter = Counter()
    for c in comps:
        if c.get("merged"):
            continue
        ys, xs = c["ys"], c["xs"]
        area = len(xs) * cell * cell
        if area < min_area_m2:
            continue
        pts2 = np.stack([(xs + x0 + 0.5) * cell, (ys + y0 + 0.5) * cell], 1).astype(np.float32)
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(pts2)
        bw, bh = bw + cell, bh + cell
        length, width = max(bw, bh), min(bw, bh)
        yaw = math.radians(ang) if bw >= bh else math.radians(ang + 90.0)
        tops = top[ys, xs]
        tops = tops[tops > 0]  # gap-filled cells have no height
        if tops.size == 0:
            continue
        surface = float(np.median(tops))
        height = float(np.percentile(tops, 95))
        label = _classify(c["band"], area, length, width, height, surface, bool(c.get("backrest")))
        if label is None:
            continue
        counts[label] += 1
        nodes.append(
            Node(
                id=f"{label}_{counts[label]}",
                layer="object",
                label=label,
                center=np.array([cx, cy, height / 2.0]),
                size=np.array([length, width, height]),
                yaw=yaw,
                top=surface,
                attrs={"area_m2": round(area, 3), "band": c["band"], "geometric": label},
                cells=_cell_keys(xs + x0, ys + y0),
            )
        )
    return nodes, room


def _fill_ratio(ys: np.ndarray, xs: np.ndarray, cell: float) -> tuple[float, float]:
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    box = (w + 1) * (h + 1)
    return len(xs) / max(box, 1.0), len(xs) * cell * cell


def _split_component(
    ys: np.ndarray,
    xs: np.ndarray,
    cell: float,
    max_area_m2: float = 3.5,
    min_fill: float = 0.45,
    depth: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a footprint that is not one compact piece of furniture (an L-shape, two desks joined by a thin
    bridge, a whole row): erode until it falls apart, then regrow each seed geodesically inside the mask."""
    fill, area = _fill_ratio(ys, xs, cell)
    if (fill >= min_fill and area <= max_area_m2) or len(xs) < 12 or depth > 3:
        return [(ys, xs)]
    y0, x0 = ys.min() - 1, xs.min() - 1
    H, W = ys.max() - y0 + 2, xs.max() - x0 + 2
    mask = np.zeros((H, W), np.uint8)
    mask[ys - y0, xs - x0] = 1
    for r in range(1, 7):
        er = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, seeds = cv2.connectedComponents(er, connectivity=4)
        sizes = np.bincount(seeds.ravel())[1:] if n > 1 else []
        if n > 2 and sum(1 for z in sizes if z >= 4) >= 2:
            break
    else:
        return [(ys, xs)]
    lab = seeds.astype(np.int32)
    lab[(lab > 0) & (np.isin(lab, [i + 1 for i, z in enumerate(sizes) if z < 4]))] = 0
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(H + W):  # geodesic regrowth: each unlabelled mask cell takes a neighbour's label
        grown = cv2.dilate(lab.astype(np.float32), k3).astype(np.int32)
        upd = (lab == 0) & (mask == 1) & (grown > 0)
        if not upd.any():
            break
        lab[upd] = grown[upd]
    out = []
    for i in np.unique(lab[lab > 0]):
        py, px = np.nonzero(lab == i)
        out += _split_component(py + y0, px + x0, cell, max_area_m2, min_fill, depth + 1)
    left = (mask == 1) & (lab == 0)
    if left.any():
        py, px = np.nonzero(left)
        out.append((py + y0, px + x0))
    return out


def _classify(band, area, length, width, height, surface, backrest) -> str | None:
    if band == "wall":
        if (length >= 0.9 and width <= 0.45) or length >= 2.0:
            return "wall"
        return "cabinet" if area >= 0.12 else "tall_object"
    if band == "tall":
        if length >= 2.0 and width <= 0.8:
            return "wall"  # long and shoulder-high: a partition, or a wall seen at a grazing angle
        return "cabinet" if 0.2 <= area <= 2.5 else ("wall" if area > 2.5 else "tall_object")
    if band == "table":
        return "table" if area >= 0.3 else "object"
    if band == "seat":
        if area >= 2.2 and width >= 0.85:
            return "bed"
        if area >= 0.6 and length >= 1.1:
            return "sofa"
        if area >= 0.06 or backrest:
            return "chair"
        return "object"
    return "low_object" if area >= 0.05 else None


# ------------------------------------------------------------------------------------------ the graph
class SceneGraph:
    """Thread-safe scene graph. `rebuild_static` replaces the object layer; agents and events are updated
    incrementally. All positions are in the registered map frame (z up, floor at 0)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.cell = float(self.cfg.get("cell_m", 0.05))
        self.lock = threading.RLock()
        self.room = Node("room_1", "room", "room", np.zeros(3))
        self.objects: list[Node] = []
        self.person: Node | None = None
        self.robot: Node | None = None
        self.events: list[Node] = []
        self._cell_owner: dict[int, int] = {}  # footprint cell -> index into self.objects
        self._dilated: dict[int, int] = {}  # footprint dilated by `support_margin_m`
        self._detections: list[tuple[str, np.ndarray, float]] = []
        self.version = 0
        self.person_ctx: dict = {}

    # ----------------------------------------------------------- static layer
    def rebuild_static(self, points: np.ndarray, ceiling_z: float | None = None) -> None:
        nodes, room = segment_objects(
            points,
            cell=self.cell,
            floor_z=float(self.cfg.get("floor_z_m", 0.08)),
            ceiling_z=ceiling_z,
            min_area_m2=float(self.cfg.get("min_area_m2", 0.04)),
            max_z=float(self.cfg.get("max_object_z_m", 2.0)),
        )
        P = np.asarray(points, float).reshape(-1, 3)
        with self.lock:
            if len(P):
                lo, hi = P[:, :2].min(0), P[:, :2].max(0)
                self.room = Node(
                    "room_1",
                    "room",
                    "room",
                    np.array([*(lo + hi) / 2, (room.get("ceiling_m") or 2.5) / 2]),
                    size=np.array([*(hi - lo), room.get("ceiling_m") or 2.5]),
                    attrs={"ceiling_m": round(room.get("ceiling_m") or 0.0, 2)},
                )
            self.objects = nodes
            self._index()
            for lab, xyz, sc in self._detections:  # re-apply semantic votes to the new segmentation
                self._vote(lab, xyz, sc)
            self._relabel()
            self.version += 1

    def _index(self):
        self._cell_owner, self._dilated = {}, {}
        r = max(1, int(round(float(self.cfg.get("support_margin_m", 0.15)) / self.cell)))
        for i, o in enumerate(self.objects):
            for key in o.cells.tolist():
                self._cell_owner[key] = i
            ix, iy = o.cells >> 32, (o.cells & 0xFFFFFFFF).astype(np.int32).astype(np.int64)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for key in _cell_keys(ix + dx, iy + dy).tolist():
                        self._dilated.setdefault(key, i)

    def _key(self, x: float, y: float) -> int:
        return int(
            _cell_keys(np.array([math.floor(x / self.cell)]), np.array([math.floor(y / self.cell)]))[0]
        )

    def object_at(self, x: float, y: float, margin: bool = True) -> Node | None:
        with self.lock:
            i = (self._dilated if margin else self._cell_owner).get(self._key(x, y))
            return None if i is None else self.objects[i]

    def nearest_object(self, xyz, max_d: float = 1.0, labels: set | None = None) -> tuple[Node | None, float]:
        best, bd = None, max_d
        with self.lock:
            for o in self.objects:
                if labels is not None and o.label not in labels:
                    continue
                if o.label == "wall" and labels is None:
                    continue
                d = _footprint_distance(o, xyz[0], xyz[1], self.cell)
                if d < bd:
                    best, bd = o, d
        return best, bd

    # ----------------------------------------------------------- semantics (optional detector)
    def add_detection(self, coco_label: str, xyz, score: float = 1.0) -> None:
        lab = COCO_TO_LABEL.get(coco_label)
        if lab is None:
            return
        xyz = np.asarray(xyz, float)
        with self.lock:
            self._detections.append((lab, xyz, score))
            self._detections = self._detections[-2000:]
            self._vote(lab, xyz, score)
            self._relabel()

    def _vote(self, lab: str, xyz: np.ndarray, score: float) -> None:
        o, d = self.nearest_object(xyz, max_d=0.35)
        if o is not None:
            o.votes[lab] += 1

    def _relabel(self) -> None:
        """A detector label overrides the geometric one after `min_votes` consistent votes."""
        need = int(self.cfg.get("min_votes", 3))
        counts: Counter = Counter()
        for o in self.objects:
            lab = o.attrs.get("geometric", o.label)
            if o.votes:
                top, n = o.votes.most_common(1)[0]
                if n >= need and n >= 0.6 * sum(o.votes.values()):
                    lab = top
            o.label = lab
            counts[lab] += 1
            o.id = f"{lab}_{counts[lab]}"

    # ----------------------------------------------------------- agents
    def update_person(
        self,
        xyz,
        hip_height: float | None = None,
        posture: str | None = None,
        cam_xyz=None,
        lying: bool = False,
        metrics: dict | None = None,
    ) -> dict:
        """Place the person and compute their relations. Returns the person context (also kept in
        `person_ctx`): support, sitting/lying, occluder, near objects and a one-line description."""
        with self.lock:
            if xyz is None:
                self.person = None
                self.person_ctx = {}
                return {}
            xyz = np.asarray(xyz, float)
            self.person = Node(
                "person_1",
                "agent",
                "person",
                xyz,
                attrs={k: v for k, v in (("posture", posture), ("hip_m", hip_height)) if v is not None},
            )
            ctx = self.context(xyz, hip_height, cam_xyz, lying=lying, posture=posture)
            ctx["metrics"] = {
                k: metrics[k]
                for k in ("neck_flexion", "trunk_flexion")
                if metrics and metrics.get(k) is not None
            }
            self.person_ctx = ctx
            return ctx

    def update_robot(self, base, cam_xyz=None) -> None:
        if base is None:
            return
        x, y, th = base
        with self.lock:
            self.robot = Node(
                "robot_1",
                "agent",
                "robot",
                np.array([x, y, 0.0]),
                yaw=float(th),
                attrs={"camera": None if cam_xyz is None else [round(float(v), 3) for v in cam_xyz]},
            )

    def context(
        self, xyz, hip_height=None, cam_xyz=None, lying: bool = False, posture: str | None = None
    ) -> dict:
        """Relations of a person at `xyz` (torso point) with hip height `hip_height` (m, may be None)."""
        x, y = float(xyz[0]), float(xyz[1])
        ctx: dict = {
            "support": "floor",
            "support_id": None,
            "relation": "on",
            "occluder": None,
            "occluder_id": None,
            "near": [],
            "at": None,
        }
        sup = self.object_at(x, y, margin=True)
        if sup is not None and sup.label in SUPPORT_LABELS | {"low_object"}:
            hh = hip_height
            if sup.label in SEAT_LABELS | RECLINE_LABELS:
                ok = hh is None or (sup.top - 0.20 <= hh <= sup.top + 0.40)
                if ok:
                    ctx["support"], ctx["support_id"] = sup.label, sup.id
                    ctx["relation"] = (
                        "lying_on" if lying else ("sitting_on" if sup.label in SEAT_LABELS else "on")
                    )
            elif sup.label == "table" and hh is not None and abs(hh - sup.top) < 0.25:
                ctx["support"], ctx["support_id"], ctx["relation"] = "table", sup.id, "on"
        if lying and ctx["support"] == "floor":
            ctx["relation"] = "lying_on"
        elif ctx["support"] == "floor" and posture == "sitting":
            seat, d = self.nearest_object((x, y), 0.5, SEAT_LABELS)
            if seat is not None:
                ctx["support"], ctx["support_id"], ctx["relation"] = seat.label, seat.id, "sitting_on"
        near = []
        with self.lock:
            for o in self.objects:
                if o.label == "wall":
                    continue
                d = _footprint_distance(o, x, y, self.cell)
                if d <= float(self.cfg.get("near_m", 0.8)):
                    near.append((d, o))
        near.sort(key=lambda t: t[0])
        ctx["near"] = [{"id": o.id, "label": o.label, "d": round(d, 2)} for d, o in near[:4]]
        tables = [(d, o) for d, o in near if o.label == "table" and d <= 0.6]
        if tables:
            ctx["at"] = tables[0][1].id
        if cam_xyz is not None:
            occ = self.occluder(
                cam_xyz, (x, y, float(xyz[2]) if hip_height is None else max(hip_height, 0.2))
            )
            if occ is not None:
                ctx["occluder"], ctx["occluder_id"] = occ.label, occ.id
        ctx["text"] = describe_context(ctx)
        return ctx

    def occluder(self, cam_xyz, target_xyz) -> Node | None:
        """First object whose top is above the camera->target ray (sampled every cell). The object directly under
        the target is skipped only when the target is above it (someone sitting on a chair is not hidden by it)."""
        c, t = np.asarray(cam_xyz, float), np.asarray(target_xyz, float)
        L = float(np.linalg.norm(t[:2] - c[:2]))
        if L < 0.3:
            return None
        n = max(2, int(L / self.cell))
        with self.lock:
            own = self._cell_owner.get(self._key(t[0], t[1]))
            if own is not None and t[2] < self.objects[own].size[2] - 0.05:
                own = None  # below the top of the object it is over (under / behind a desk): that object can hide it
            for s in np.linspace(0.05, 1.0 - 0.10 / L, n):
                p = c + s * (t - c)
                i = self._cell_owner.get(self._key(p[0], p[1]))
                if i is None or i == own:
                    continue
                o = self.objects[i]
                if o.label in ("wall", "tall_object"):
                    continue
                if o.size[2] >= p[2] - 0.02:
                    return o
        return None

    # ----------------------------------------------------------- events
    def add_event(self, kind: str, xyz, t: float, title: str = "", **attrs) -> Node | None:
        if xyz is None:
            return None
        xyz = np.asarray(xyz, float)
        with self.lock:
            o, d = self.nearest_object(xyz, max_d=float(self.cfg.get("event_object_m", 1.0)))
            ev = Node(
                f"event_{len(self.events) + 1}",
                "event",
                kind,
                xyz,
                attrs={
                    "t": t,
                    "title": title,
                    "at": o.id if o else "floor",
                    "d": round(d, 2) if o else None,
                    **attrs,
                },
            )
            self.events.append(ev)
            self.events = self.events[-500:]
            self.version += 1
            return ev

    # ----------------------------------------------------------- export
    def edges(self) -> list[Edge]:
        E: list[Edge] = []
        with self.lock:
            for o in self.objects:
                E.append(Edge(o.id, "in", self.room.id))
            for i, a in enumerate(self.objects):
                for b in self.objects[i + 1 :]:
                    if "wall" in (a.label, b.label):
                        continue
                    d = _footprint_gap(a, b, self.cell)
                    if d <= 0.25:
                        if a.label in SEAT_LABELS and b.label == "table":
                            E.append(Edge(a.id, "at", b.id))
                        elif b.label in SEAT_LABELS and a.label == "table":
                            E.append(Edge(b.id, "at", a.id))
                        else:
                            E.append(Edge(a.id, "next_to", b.id, {"gap_m": round(d, 2)}))
            if self.person is not None:
                E.append(Edge("person_1", "in", self.room.id))
                ctx = self.person_ctx
                if ctx.get("support_id"):
                    E.append(Edge("person_1", ctx["relation"], ctx["support_id"]))
                elif ctx.get("relation") == "lying_on":
                    E.append(Edge("person_1", "lying_on", "floor"))
                if ctx.get("at"):
                    E.append(Edge("person_1", "at", ctx["at"]))
                for n in ctx.get("near", []):
                    if n["id"] not in (ctx.get("support_id"), ctx.get("at")):
                        E.append(Edge("person_1", "near", n["id"], {"d": n["d"]}))
                if ctx.get("occluder_id"):
                    E.append(Edge("person_1", "occluded_by", ctx["occluder_id"]))
            if self.robot is not None:
                E.append(Edge("robot_1", "in", self.room.id))
                if self.person is not None:
                    d = float(np.linalg.norm(self.person.center[:2] - self.robot.center[:2]))
                    E.append(Edge("robot_1", "sees", "person_1", {"d": round(d, 2)}))
            for ev in self.events:
                E.append(Edge(ev.id, "at", ev.attrs.get("at", "floor")))
        return E

    def to_json(self) -> dict:
        with self.lock:
            nodes = [self.room.to_json()] + [o.to_json() for o in self.objects]
            nodes += [n.to_json() for n in (self.person, self.robot) if n is not None]
            nodes += [e.to_json() for e in self.events]
            return {
                "version": self.version,
                "nodes": nodes,
                "edges": [e.to_json() for e in self.edges()],
                "summary": self.summary(),
            }

    def summary(self) -> dict:
        with self.lock:
            c = Counter(o.label for o in self.objects)
        return dict(sorted(c.items()))

    def describe(self) -> str:
        """Short room + person description for the language model and the VR HUD."""
        s = self.summary()
        parts = [
            f"{n} {pretty(k)}{'s' if n > 1 else ''}"
            for k, n in s.items()
            if k not in ("wall", "object", "tall_object")
        ]
        room = f"Room with {', '.join(parts)}." if parts else "Room."
        p = self.person_ctx.get("text") if self.person_ctx else None
        return f"{room} {p}" if p else room

    def boxes(self) -> list[dict]:
        """Compact list for the VR / report viewer."""
        with self.lock:
            return [
                {
                    "id": o.id,
                    "label": o.label,
                    "c": [round(float(v), 3) for v in o.center],
                    "s": [round(float(v), 3) for v in o.size],
                    "yaw": round(float(o.yaw), 4),
                }
                for o in self.objects
                if o.label != "wall"
            ]


def describe_context(ctx: dict) -> str:
    rel = ctx.get("relation")
    if ctx.get("support_id"):
        what = {"sitting_on": "sitting on", "lying_on": "lying on", "on": "on"}.get(rel, "on")
        s = f"The person is {what} {pretty(ctx['support']).replace('_', ' ')} {ctx['support_id'].split('_')[-1]}"
    elif rel == "lying_on":
        s = "The person is lying on the floor"
    else:
        s = "The person is on open floor"
    if ctx.get("at") and ctx["at"] != ctx.get("support_id"):
        s += f" at table {ctx['at'].split('_')[-1]}"
    elif ctx.get("near"):
        n = ctx["near"][0]
        if n["id"] != ctx.get("support_id"):
            s += f", {n['d']:.1f} m from {pretty(n['label'])} {n['id'].split('_')[-1]}"
    if ctx.get("occluder"):
        s += (
            f"; partly hidden from the robot by {pretty(ctx['occluder'])} {ctx['occluder_id'].split('_')[-1]}"
        )
    return s + "."


def _footprint_distance(o: Node, x: float, y: float, cell: float) -> float:
    ix, iy = o.cells >> 32, (o.cells & 0xFFFFFFFF).astype(np.int32)
    dx = (ix + 0.5) * cell - x
    dy = (iy + 0.5) * cell - y
    return float(max(0.0, np.sqrt(dx * dx + dy * dy).min() - cell / 2))


def _footprint_gap(a: Node, b: Node, cell: float) -> float:
    # cheap: compare centers' box distance first
    ra = 0.5 * float(np.hypot(*a.size[:2]))
    rb = 0.5 * float(np.hypot(*b.size[:2]))
    if float(np.linalg.norm(a.center[:2] - b.center[:2])) - ra - rb > 0.3:
        return 9.0
    ax, ay = a.cells >> 32, (a.cells & 0xFFFFFFFF).astype(np.int32)
    bx, by = b.cells >> 32, (b.cells & 0xFFFFFFFF).astype(np.int32)
    A = np.stack([ax, ay], 1).astype(np.float32)
    B = np.stack([bx, by], 1).astype(np.float32)
    if len(A) * len(B) > 4_000_000:
        A, B = A[:: max(1, len(A) // 2000)], B[:: max(1, len(B) // 2000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
    return float(max(0.0, (d - 1.0) * cell))


# ------------------------------------------------------------------------------------------ detector
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/"
    "efficientdet_lite0.tflite"
)


def default_detector_path() -> Path:
    from .config import home

    return home() / "models" / "efficientdet_lite0.tflite"


def download_detector(dest: Path | None = None) -> Path:
    import requests

    dest = Path(dest or default_detector_path())
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(DETECTOR_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return dest


class SemanticDetector:
    """Optional MediaPipe object detector (EfficientDet-Lite, COCO). Detections are back-projected with the
    aligned depth and voted onto graph objects. Disabled (returns []) if the model file is missing."""

    def __init__(self, model_path: str | Path, score: float = 0.4):
        self.ok = False
        self.det = None
        p = Path(model_path).expanduser()
        if not p.exists():
            log.info("scene graph: no detector model at %s (geometric labels only)", p)
            return
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            opts = vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(p)),
                score_threshold=score,
                max_results=12,
            )
            self.det = vision.ObjectDetector.create_from_options(opts)
            self.ok = True
        except Exception as e:  # pragma: no cover - depends on mediapipe build
            log.warning("scene graph: detector unavailable: %s", e)

    def detect(self, rgb: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        if not self.ok:
            return []
        import mediapipe as mp

        res = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for d in res.detections:
            c = d.categories[0]
            b = d.bounding_box
            out.append((c.category_name, float(c.score), (b.origin_x, b.origin_y, b.width, b.height)))
        return out


def backproject_box(
    box, depth_m: np.ndarray, intr: list, T_map_cam: np.ndarray, img_shape
) -> np.ndarray | None:
    """Map point of a detection: median depth of the box's central third, through the camera pose."""
    x, y, w, h = box
    H, W = img_shape[:2]
    dh, dw = depth_m.shape[:2]
    sx, sy = dw / W, dh / H
    u0, u1 = int((x + w / 3) * sx), int((x + 2 * w / 3) * sx)
    v0, v1 = int((y + h / 3) * sy), int((y + 2 * h / 3) * sy)
    win = depth_m[max(v0, 0) : max(v1, v0 + 1), max(u0, 0) : max(u1, u0 + 1)]
    valid = win[(win > 0.2) & (win < 6.0)]
    if valid.size < 5:
        return None
    z = float(np.median(valid))
    fx, fy, cx, cy = intr[:4]
    u, v = (x + w / 2), (y + h / 2)
    # intrinsics [fx, fy, cx, cy, w, h] are for the full-resolution upright image
    sxf, syf = (intr[4] / W) if len(intr) > 5 else 1.0, (intr[5] / H) if len(intr) > 5 else 1.0
    p = np.array([(u * sxf - cx) / fx * z, (v * syf - cy) / fy * z, z, 1.0])
    return (np.asarray(T_map_cam, float).reshape(4, 4) @ p)[:3]


# ------------------------------------------------------------------------------------------ offline
def read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = Path(path).read_bytes()
    n = struct.unpack_from("<I", b, 0)[0]
    o = 4
    P = np.frombuffer(b, "<f4", n * 3, o).reshape(n, 3).astype(float)
    o += n * 12
    C = np.frombuffer(b, np.uint8, n * 3, o).reshape(n, 3)
    o += n * 3
    o += (-n * 3) % 4
    m = struct.unpack_from("<I", b, o)[0]
    o += 4
    cam = np.frombuffer(b, "<f4", m * 3, o).reshape(m, 3).astype(float)
    o += m * 12
    k = struct.unpack_from("<I", b, o)[0]
    o += 4
    trk = np.frombuffer(b, "<f4", k * 3, o).reshape(k, 3).astype(float)
    return P, C, cam, trk


def build_scene_graph(
    session_dir: str | Path, cfg: dict | None = None, detector_model: str | None = None
) -> dict | None:
    """Offline: build <session>/scene/scene_graph.json from the registered map, the keyframes and the captions.
    Adds per-object posture statistics (REBA by where the person was) and attaches every captioned event."""
    sd = Path(session_dir)
    sc = sd / "scene"
    if not (sc / "points.bin").exists():
        log.warning("scene graph: %s has no scene/points.bin (build the scene first)", sd)
        return None
    cfg = cfg or {}
    P, _, _, _ = read_points_bin(sc / "points.bin")
    ceiling = None
    try:
        ceiling = json.loads((sc / "overlay.json").read_text()).get("ceiling_m")
    except Exception:
        pass
    g = SceneGraph(cfg)
    g.rebuild_static(P, ceiling_z=(ceiling + 0.15) if ceiling else None)

    frames = []
    if (sc / "frames.jsonl").exists():
        frames = [json.loads(ln) for ln in (sc / "frames.jsonl").read_text().splitlines() if ln.strip()]

    # optional semantic votes from the keyframes
    det = SemanticDetector(detector_model) if detector_model else None
    if det is not None and det.ok:
        for fr in frames[:: max(1, int(cfg.get("detector_stride", 4)))]:
            rgbp = sc / "rgb" / f"{fr['i']:08d}.jpg"
            dp = sc / "depth" / f"{fr['i']:08d}.png"
            if not (rgbp.exists() and dp.exists() and fr.get("T_map_cam")):
                continue
            img = cv2.cvtColor(cv2.imread(str(rgbp)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
            for name, score, box in det.detect(img):
                xyz = backproject_box(box, depth, fr["intr"], np.array(fr["T_map_cam"]), img.shape)
                if xyz is not None:
                    g.add_detection(name, xyz, score)

    # person positions per keyframe -> posture per object
    import csv

    rows = []
    mw = mp_ = mr = None
    mpath = sd / "metrics.csv"
    if mpath.exists():
        with open(mpath, newline="") as fh:
            rr = list(csv.DictReader(fh))

        def num(v):
            return float(v) if v not in ("", None) else np.nan

        mw = np.array([num(r.get("wall_time")) for r in rr])
        mp_ = np.array([num(r.get("present")) for r in rr])
        mr = np.array([num(r.get("reba")) for r in rr])
    for fr in frames:
        pc = fr.get("person_cam")
        T = fr.get("T_map_cam")
        if pc is None or T is None:
            continue
        xyz = (np.array(T, float).reshape(4, 4) @ np.array([*pc, 1.0]))[:3]
        cam = np.array(T, float).reshape(4, 4)[:3, 3]
        reba = None
        if mw is not None and len(mw) and "wall" in fr:
            j = int(np.nanargmin(np.abs(mw - fr["wall"])))
            if abs(mw[j] - fr["wall"]) < 0.5 and mp_[j] == 1 and not np.isnan(mr[j]):
                reba = float(mr[j])
        ctx = g.context(xyz, None, cam, posture=fr.get("posture"))
        rows.append({"t": fr.get("wall"), "xyz": xyz, "ctx": ctx, "reba": reba, "fall": fr.get("fall")})

    by_obj: dict[str, list[float]] = {}
    where: Counter = Counter()
    occluded = 0
    for r in rows:
        key = (
            r["ctx"].get("support_id")
            or r["ctx"].get("at")
            or (r["ctx"]["near"][0]["id"] if r["ctx"]["near"] else "open_floor")
        )
        where[key] += 1
        if r["ctx"].get("occluder"):
            occluded += 1
        if r["reba"] is not None:
            by_obj.setdefault(key, []).append(r["reba"])
    posture = {k: {"samples": len(v), "reba_mean": round(float(np.mean(v)), 2)} for k, v in by_obj.items()}

    # events (captions) -> graph events, with the camera position at that time for the occlusion test
    caps = []
    try:
        caps = json.loads((sc / "overlay.json").read_text()).get("captions", [])
    except Exception:
        pass
    walls = np.array([f.get("wall", 0.0) for f in frames]) if frames else np.zeros(0)
    falls = []
    for c in caps:
        if not c.get("pos"):
            continue
        ev = g.add_event(c.get("kind", "event"), c["pos"], c.get("t", 0.0), c.get("title", ""))
        if c.get("kind", "").startswith("fall") and len(walls):
            j = int(np.argmin(np.abs(walls - c.get("t", 0.0))))
            T = frames[j].get("T_map_cam")
            cam = np.array(T, float).reshape(4, 4)[:3, 3] if T else None
            ctx = g.context(np.array(c["pos"]), 0.4, cam, lying=True)  # someone who went down: hips ~0.4 m
            ev.attrs.update(
                support=ctx["support"],
                occluder=ctx.get("occluder_id"),
                near=ctx["near"][0]["id"] if ctx["near"] else None,
            )
            falls.append(
                {
                    "kind": c["kind"],
                    "text": c.get("text", ""),
                    "at": ev.attrs["at"],
                    "occluder": ctx.get("occluder_id"),
                    "near": ev.attrs["near"],
                }
            )

    out = g.to_json()
    out["posture_by_object"] = dict(sorted(posture.items(), key=lambda kv: -kv[1]["samples"]))
    out["presence_by_object"] = dict(where.most_common())
    out["person_samples"] = len(rows)
    out["person_occluded_samples"] = occluded
    out["fall_events"] = falls
    out["boxes"] = g.boxes()
    out["description"] = g.describe()
    (sc / "scene_graph.json").write_text(json.dumps(out, indent=1))
    log.info("scene graph: %s", out["summary"])
    return out
