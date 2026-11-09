"""Detection metric sanity checks."""

import numpy as np
import pytest

from through_the_eyes_of_a_vru.data.nusc import Box
from through_the_eyes_of_a_vru.metrics.detection import (
    evaluate,
    merge_duplicates,
)


def box(x: float, cat: str, score: float | None = None) -> Box:
    return Box(
        token="t",
        category=cat,
        center=np.array([x, 0.0, 0.0]),
        size=np.array([2.0, 4.0, 1.5]),
        rot=np.eye(3),
        score=score,
    )


def test_perfect_detector_scores_one() -> None:
    gt = {0: [box(0, "vehicle.car"), box(10, "vehicle.car")]}
    pred = {0: [box(0.1, "car", 0.9), box(10.1, "car", 0.8)]}
    r = evaluate(gt, pred)
    assert r.classes["car"].ap == pytest.approx(1.0)
    assert r.classes["bicycle"].ap is None  # no GT -> excluded from mAP
    assert r.m_ap == pytest.approx(1.0)


def test_missing_detections_score_zero_and_ignored_classes() -> None:
    gt = {0: [box(0, "vehicle.car"), box(5, "vehicle.motorcycle")]}
    r = evaluate(gt, {0: []})
    assert r.classes["car"].ap == 0.0
    assert r.classes["car"].n_gt == 1
    assert r.m_ap == 0.0


def test_merge_duplicates_keeps_best() -> None:
    kept = merge_duplicates(
        [box(0, "car", 0.5), box(0.2, "car", 0.9), box(8, "car", 0.4)]
    )
    assert [b.score for b in kept] == [0.9, 0.4]
