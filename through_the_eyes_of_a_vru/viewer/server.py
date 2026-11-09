"""Flask backend of the viewer (JSON + one binary point endpoint)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

from through_the_eyes_of_a_vru.data.nusc import (
    AGENTS,
    MIN_AGENT_POINTS,
    Box,
    Frame,
    Scene,
    agent_counts,
    find_scenes,
    load_points,
    points_in_agents,
)
from through_the_eyes_of_a_vru.metrics.detection import (
    DISPLAY_THRESHOLD,
    EVAL_CLASSES,
    EvalResult,
    eval_class,
    evaluate,
    match_flags,
    merge_duplicates,
)

STATIC = Path(__file__).parent / "static"


def _agents() -> list[str]:
    got = [a for a in request.args.get("agents", "car,bike").split(",") if a]
    if not got or any(a not in AGENTS for a in got):
        abort(400, "agents must be a non-empty subset of car,bike")
    return got


def _result(r: EvalResult) -> dict[str, Any]:
    return {
        "m_ap": r.m_ap,
        "classes": {
            c: {"ap": v.ap, "n_gt": v.n_gt, "n_pred": v.n_pred}
            for c, v in r.classes.items()
        },
    }


def _box_json(frame: Frame, b: Box, tp: bool | None = None) -> dict[str, Any]:
    c, r = frame.to_sensor(b.center, b.rot)
    return {
        "x": float(c[0]),
        "y": float(c[1]),
        "z": float(c[2]),
        "w": float(b.size[0]),
        "l": float(b.size[1]),
        "h": float(b.size[2]),
        "yaw": float(np.arctan2(r[1, 0], r[0, 0])),
        "cat": b.category,
        "cls": eval_class(b.category),
        "score": b.score,
        "agent": b.agent,
        "tp": tp,
    }


def create_app(root: Path) -> Flask:
    app = Flask(__name__)
    scenes: dict[str, Scene] = find_scenes(root)

    def scene_of() -> Scene:
        sc = scenes.get(request.args.get("scene", ""))
        if sc is None:
            abort(404, "unknown scene")
        return sc

    def frame_of(sc: Scene) -> Frame:
        try:
            return sc.frame(int(request.args.get("frame", "")))
        except (KeyError, ValueError):
            abort(404, "unknown frame")

    @functools.lru_cache(maxsize=64)
    def gt_counts(scene: str, index: int) -> Any:
        sc = scenes[scene]
        f = sc.frame(index)
        return agent_counts(f, sc.gt.get(f.token, []))

    def gt_boxes(sc: Scene, f: Frame, agents: list[str]) -> list[Box]:
        boxes = sc.gt.get(f.token, [])
        cols = [list(AGENTS).index(a) for a in agents]
        n = gt_counts(sc.id, f.index)[:, cols].sum(axis=1)
        return [
            b for b, k in zip(boxes, n, strict=True) if k >= MIN_AGENT_POINTS
        ]

    def det_boxes(
        sc: Scene, f: Frame, method: str, agents: list[str], min_score: float
    ) -> list[Box]:
        boxes = [
            b
            for b in sc.methods[method].boxes.get(f.token, [])
            if b.agent in agents and (b.score or 0.0) >= min_score
        ]
        return merge_duplicates(boxes) if len(agents) > 1 else boxes

    @app.get("/")
    def index() -> Response:
        return send_file(STATIC / "index.html")

    @app.get("/api/scenes")
    def api_scenes() -> Response:
        return jsonify(
            [
                {
                    "id": s.id,
                    "frames": [
                        {"index": f.index, "time": round(f.time, 2)}
                        for f in s.frames
                    ],
                    "methods": [
                        {"id": m.id, "name": m.name}
                        for m in s.methods.values()
                    ],
                    "cameras": sorted({c for f in s.frames for c in f.images}),
                    "floor": None
                    if s.floor is None
                    else {
                        "verts": len(s.floor.verts),
                        "faces": len(s.floor.faces),
                    },
                }
                for s in scenes.values()
            ]
        )

    @app.get("/api/points")
    def api_points() -> Response:
        sc, agents = scene_of(), _agents()
        f = frame_of(sc)
        pts = points_in_agents(load_points(f.lidar), agents)
        limit = int(request.args.get("max", "160000"))
        pts = pts[:: max(1, len(pts) // limit)]
        hag = np.full(len(pts), np.nan, dtype=np.float32)
        if sc.floor is not None:
            g = f.points_to_global(pts[:, :3].astype(np.float64))
            hag = (g[:, 2] - sc.floor.height(g[:, :2])).astype(np.float32)
        out = np.ascontiguousarray(
            np.column_stack([pts[:, [0, 1, 2, 4]], hag]).astype(np.float32)
        )
        return Response(out.tobytes(), mimetype="application/octet-stream")

    @app.get("/api/floor")
    def api_floor() -> Response:
        floor = scene_of().floor
        if floor is None:
            abort(404, "scene has no floor mesh")
        part = request.args.get("part", "verts")
        data = (
            floor.verts.astype("<f4")
            if part == "verts"
            else floor.faces.astype("<u4")
        )
        return Response(data.tobytes(), mimetype="application/octet-stream")

    @app.get("/api/frame")
    def api_frame() -> Response:
        sc, agents = scene_of(), _agents()
        f = frame_of(sc)
        min_score = float(request.args.get("min_score", "0.1"))
        ids = [m for m in request.args.get("methods", "").split(",") if m]
        gt_all = {g.index: gt_boxes(sc, g, agents) for g in sc.frames}
        gt_now = gt_all[f.index]
        n_gt = {
            c: sum(1 for b in gt_now if eval_class(b.category) == c)
            for c in EVAL_CLASSES
        }
        methods: dict[str, Any] = {}
        for m in ids:
            if m not in sc.methods:
                abort(404, "unknown method")
            pred_all = {
                g.index: det_boxes(sc, g, m, agents, min_score)
                for g in sc.frames
            }
            pred = pred_all[f.index]
            flags = match_flags(gt_now, pred, DISPLAY_THRESHOLD)
            methods[m] = {
                "boxes": [
                    _box_json(f, b, ok if eval_class(b.category) else None)
                    for b, ok in zip(pred, flags, strict=True)
                ],
                "frame": _result(evaluate({f.index: gt_now}, {f.index: pred})),
                "scene": _result(evaluate(gt_all, pred_all)),
            }
        floor_tf = None
        if sc.floor is not None:
            a, b = f.affine()
            base = f.base_in_sensor("car")
            z_ref = None
            if base is not None:
                g = f.points_to_global(np.array([[base[0], base[1], 0.0]]))
                z = float(sc.floor.height(g[:, :2])[0])
                if not np.isnan(z):
                    z_ref = float(
                        a[2] @ np.array([g[0, 0], g[0, 1], z]) + b[2]
                    )
            floor_tf = {"A": a.ravel().tolist(), "b": b.tolist(), "ref": z_ref}
        return jsonify(
            {
                "floor": floor_tf,
                "markers": {a: f.base_in_sensor(a) for a in AGENTS},
                "gt": [_box_json(f, b) for b in gt_now],
                "gt_counts": n_gt,
                "methods": methods,
                "n_frames": len(sc.frames),
            }
        )

    @app.get("/api/image")
    def api_image() -> Response:
        f = frame_of(scene_of())
        path = f.images.get(request.args.get("cam", ""))
        if path is None:
            abort(404, "no such image")
        return send_file(path)

    return app
