"""nuScenes-style detection AP over a set of frames.

Matching is by BEV centre distance (thresholds 0.5, 1, 2, 4 m), AP is the
nuScenes 101-point interpolated AP (min recall / precision 0.1), and mAP is
the mean over the classes that have ground truth in scope. Ground truth
categories the detector cannot output (e.g. motorcycles) are ignored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from through_the_eyes_of_a_vru.data.nusc import Box

EVAL_CLASSES = ("car", "bicycle", "pedestrian")
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DISPLAY_THRESHOLD = 2.0
MERGE_RADIUS = 1.0

# Ground-truth category -> evaluated class (everything else is ignored).
CLASS_OF: dict[str, str] = {
    "vehicle.car": "car",
    "car": "car",
    "vehicle.bicycle": "bicycle",
    "bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "pedestrian": "pedestrian",
}


def eval_class(category: str) -> str | None:
    return CLASS_OF.get(category)


@dataclass
class ClassResult:
    ap: float | None
    n_gt: int
    n_pred: int


@dataclass
class EvalResult:
    m_ap: float | None
    classes: dict[str, ClassResult]


def _xy(boxes: Iterable[Box]) -> NDArray[np.float64]:
    return np.array([b.center[:2] for b in boxes], dtype=np.float64).reshape(
        -1, 2
    )


def _of_class(boxes: Sequence[Box], cls: str) -> list[Box]:
    return [b for b in boxes if eval_class(b.category) == cls]


def match_flags(
    gt: Sequence[Box], pred: Sequence[Box], thr: float
) -> list[bool]:
    """True-positive flag per prediction (same class, greedy by score)."""
    flags = [False] * len(pred)
    for cls in EVAL_CLASSES:
        g = _xy(_of_class(gt, cls))
        used = np.zeros(len(g), dtype=bool)
        idx = [i for i, p in enumerate(pred) if eval_class(p.category) == cls]
        idx.sort(key=lambda i: -(pred[i].score or 0.0))
        for i in idx:
            if not len(g):
                break
            d = np.linalg.norm(g - pred[i].center[:2], axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] < thr:
                used[j] = True
                flags[i] = True
    return flags


def _ap(tp: NDArray[np.bool_], n_gt: int) -> float:
    if not len(tp):
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    grid = np.linspace(0.0, 1.0, 101)
    p = np.interp(grid, rec, prec, right=0.0)[11:]
    p = np.clip(p - 0.1, 0.0, None)
    return float(p.mean() / 0.9)


def evaluate(
    gt: Mapping[int, Sequence[Box]], pred: Mapping[int, Sequence[Box]]
) -> EvalResult:
    """mAP and per-class AP; ``gt``/``pred`` map frame index -> boxes."""
    classes: dict[str, ClassResult] = {}
    for cls in EVAL_CLASSES:
        n_gt = sum(len(_of_class(g, cls)) for g in gt.values())
        n_pred = sum(len(_of_class(p, cls)) for p in pred.values())
        if n_gt == 0:
            classes[cls] = ClassResult(None, 0, n_pred)
            continue
        aps = []
        for thr in THRESHOLDS:
            scores: list[float] = []
            tps: list[bool] = []
            for f, p in pred.items():
                flags = match_flags(gt.get(f, []), p, thr)
                for b, ok in zip(p, flags, strict=True):
                    if eval_class(b.category) == cls:
                        scores.append(b.score or 0.0)
                        tps.append(ok)
            order = np.argsort(-np.asarray(scores), kind="stable")
            tp = np.asarray(tps, dtype=bool)[order]
            aps.append(_ap(tp, n_gt))
        classes[cls] = ClassResult(float(np.mean(aps)), n_gt, n_pred)
    vals = [c.ap for c in classes.values() if c.ap is not None]
    return EvalResult(float(np.mean(vals)) if vals else None, classes)


def merge_duplicates(boxes: Sequence[Box]) -> list[Box]:
    """Class-wise centre-distance NMS to fuse two agents' detections."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: -(x.score or 0.0)):
        cls = eval_class(b.category)
        if not any(
            eval_class(k.category) == cls
            and np.linalg.norm(k.center[:2] - b.center[:2]) < MERGE_RADIUS
            for k in kept
        ):
            kept.append(b)
    return kept
