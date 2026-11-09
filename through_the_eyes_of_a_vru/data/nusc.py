"""Minimal nuScenes-format reader: frames, boxes and lidar sweeps.

Only the JSON tables and numpy are used (no devkit). A scene folder looks
like ``<root>/<scene>/{annotation,methods/<id>,lidar,images}``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from through_the_eyes_of_a_vru.data.floor import Floor, read_ply

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]

AGENTS: dict[str, float] = {"car": 1.0, "bike": 2.0}
MIN_AGENT_POINTS = 3  # a GT box belongs to an agent if it holds >= this many


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def quat_to_matrix(q: Sequence[float]) -> F64:
    """Rotation matrix of a nuScenes quaternion ``[w, x, y, z]``."""
    w, x, y, z = (float(v) for v in q)
    s = 2.0 / (w * w + x * x + y * y + z * z)
    xx, yy, zz = s * x * x, s * y * y, s * z * z
    xy, xz, yz = s * x * y, s * x * z, s * y * z
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    return np.array(
        [
            [1 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1 - xx - yy],
        ]
    )


@dataclass(eq=False)
class Box:
    """A 3D box in the global frame (nuScenes size order: w, l, h)."""

    token: str
    category: str
    center: F64
    size: F64
    rot: F64
    score: float | None = None
    agent: str | None = None


@dataclass(eq=False)
class Frame:
    """One sample with its lidar sweep and pose chain."""

    index: int
    token: str
    time: float
    lidar: Path
    ego_t: F64
    ego_r: F64
    sensor_t: F64
    sensor_r: F64
    bike_t: F64 | None = None
    bike_r: F64 | None = None
    images: dict[str, Path] = field(default_factory=dict)

    def to_sensor(self, center: F64, rot: F64) -> tuple[F64, F64]:
        """Map a global pose into the lidar frame (centre, rotation)."""
        p = self.sensor_r.T @ (self.ego_r.T @ (center - self.ego_t))
        p = p - self.sensor_r.T @ self.sensor_t
        return p, self.sensor_r.T @ self.ego_r.T @ rot

    def affine(self) -> tuple[F64, F64]:
        """``(A, b)`` with ``p_sensor = A @ p_global + b``."""
        a = self.sensor_r.T @ self.ego_r.T
        return a, -a @ self.ego_t - self.sensor_r.T @ self.sensor_t

    def points_to_global(self, p: NDArray[np.floating]) -> F64:
        """Map ``(n, 3)`` lidar-frame points into the global frame."""
        a, b = self.affine()
        return np.asarray((p - b) @ a, dtype=np.float64)

    def base_in_sensor(self, agent: str) -> tuple[float, float, float] | None:
        """Planar pose (x, y, yaw) of an agent's base in the lidar frame."""
        if agent == "car":
            t, r = self.ego_t, self.ego_r
        elif self.bike_t is not None and self.bike_r is not None:
            t, r = self.bike_t, self.bike_r
        else:
            return None
        p, rs = self.to_sensor(t, r)
        return float(p[0]), float(p[1]), float(np.arctan2(rs[1, 0], rs[0, 0]))


@dataclass(eq=False)
class Method:
    id: str
    name: str
    boxes: dict[str, list[Box]]


@dataclass(eq=False)
class Scene:
    id: str
    root: Path
    frames: list[Frame]
    gt: dict[str, list[Box]]
    methods: dict[str, Method]
    floor: Floor | None = None

    def frame(self, index: int) -> Frame:
        for f in self.frames:
            if f.index == index:
                return f
        raise KeyError(index)

    @property
    def has_images(self) -> bool:
        return any(f.images for f in self.frames)


def load_boxes(folder: Path) -> dict[str, list[Box]]:
    """Read ``sample_annotation``/``instance``/``category`` of a folder."""
    cats = {c["token"]: c["name"] for c in read_json(folder / "category.json")}
    inst = {
        i["token"]: cats[i["category_token"]]
        for i in read_json(folder / "instance.json")
    }
    out: dict[str, list[Box]] = {}
    for a in read_json(folder / "sample_annotation.json"):
        out.setdefault(a["sample_token"], []).append(
            Box(
                token=a["token"],
                category=inst[a["instance_token"]],
                center=np.asarray(a["translation"], dtype=np.float64),
                size=np.asarray(a["size"], dtype=np.float64),
                rot=quat_to_matrix(a["rotation"]),
                score=a.get("score"),
                agent=a.get("agent"),
            )
        )
    return out


def _frames(ann: Path, root: Path) -> list[Frame]:
    samples = read_json(ann / "sample.json")
    ego = {e["token"]: e for e in read_json(ann / "ego_pose.json")}
    cal = {c["token"]: c for c in read_json(ann / "calibrated_sensor.json")}
    bike_path = ann / "ego_pose_helper.json"
    bike = (
        {e["sample_token"]: e for e in read_json(bike_path)}
        if bike_path.exists()
        else {}
    )
    data: dict[str, list[dict[str, Any]]] = {}
    for d in read_json(ann / "sample_data.json"):
        data.setdefault(d["sample_token"], []).append(d)
    samples.sort(key=lambda s: s["timestamp"])
    t0 = samples[0]["timestamp"]
    frames: list[Frame] = []
    for n, s in enumerate(samples):
        lid = next(
            (d for d in data.get(s["token"], []) if "lidar" in d["filename"]),
            None,
        )
        if lid is None or not (root / lid["filename"]).exists():
            continue
        e, c = ego[lid["ego_pose_token"]], cal[lid["calibrated_sensor_token"]]
        b = bike.get(s["token"])
        images = {
            Path(d["filename"]).parent.name: root / d["filename"]
            for d in data.get(s["token"], [])
            if d["fileformat"] in ("jpg", "png")
            and (root / d["filename"]).exists()
        }
        frames.append(
            Frame(
                index=int(s.get("index", n)),
                token=s["token"],
                time=float(s["timestamp"] - t0),
                lidar=root / lid["filename"],
                ego_t=np.asarray(e["translation"], dtype=np.float64),
                ego_r=quat_to_matrix(e["rotation"]),
                sensor_t=np.asarray(c["translation"], dtype=np.float64),
                sensor_r=quat_to_matrix(c["rotation"]),
                bike_t=None
                if b is None
                else np.asarray(b["translation"], dtype=np.float64),
                bike_r=None if b is None else quat_to_matrix(b["rotation"]),
                images=images,
            )
        )
    return frames


def load_scene(root: Path) -> Scene:
    """Load ``root`` (one scene folder) including all method folders."""
    ann = root / "annotation"
    methods: dict[str, Method] = {}
    for m in sorted((root / "methods").glob("*/sample_annotation.json")):
        meta_path = m.parent / "meta.json"
        name = (
            read_json(meta_path).get("name", m.parent.name)
            if meta_path.exists()
            else m.parent.name
        )
        methods[m.parent.name] = Method(
            m.parent.name, name, load_boxes(m.parent)
        )
    return Scene(
        id=root.name,
        root=root,
        frames=_frames(ann, root),
        gt=load_boxes(ann),
        methods=methods,
        floor=read_ply(root / "floor.ply")
        if (root / "floor.ply").exists()
        else None,
    )


def find_scenes(root: Path) -> dict[str, Scene]:
    return {
        p.parent.parent.name: load_scene(p.parent.parent)
        for p in sorted(root.glob("*/annotation/sample.json"))
    }


@functools.lru_cache(maxsize=4)
def load_points(path: Path) -> F32:
    """Raw sweep as ``(n, 5)``: x, y, z, intensity, agent (1 car, 2 bike)."""
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, 5)


def points_in_agents(points: F32, agents: Sequence[str]) -> F32:
    mask = np.isin(points[:, 4], [AGENTS[a] for a in agents])
    return points[mask]


def agent_counts(frame: Frame, boxes: Sequence[Box]) -> NDArray[np.int64]:
    """Lidar points of each agent inside each box: ``(n_boxes, 2)``."""
    pts = load_points(frame.lidar)
    xyz = pts[:, :3].astype(np.float64)
    out = np.zeros((len(boxes), len(AGENTS)), dtype=np.int64)
    for i, b in enumerate(boxes):
        c, r = frame.to_sensor(b.center, b.rot)
        rad = float(np.linalg.norm(b.size)) / 2.0
        near = np.linalg.norm(xyz[:, :2] - c[:2], axis=1) < rad
        local = (xyz[near] - c) @ r  # rows: R^T (p - c)
        half = np.array([b.size[1], b.size[0], b.size[2]]) / 2.0
        inside = np.all(np.abs(local) <= half, axis=1)
        ag = pts[near, 4][inside]
        for j, code in enumerate(AGENTS.values()):
            out[i, j] = int(np.sum(ag == code))
    return out
