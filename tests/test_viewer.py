"""The shipped samples load and the API answers."""

from pathlib import Path

import pytest

from through_the_eyes_of_a_vru.app import DEFAULT_ROOT
from through_the_eyes_of_a_vru.viewer.server import create_app

pytestmark = pytest.mark.skipif(
    not any(Path(DEFAULT_ROOT).glob("*/annotation")), reason="no samples"
)


def test_api_roundtrip() -> None:
    c = create_app(DEFAULT_ROOT).test_client()
    scenes = c.get("/api/scenes").get_json()
    sc = scenes[0]
    frame = sc["frames"][0]["index"]
    base = f"scene={sc['id']}&frame={frame}"
    pts = c.get(f"/api/points?{base}&agents=car,bike").data
    assert len(pts) % 20 == 0
    mid = sc["methods"][0]["id"] if sc["methods"] else ""
    r = c.get(f"/api/frame?{base}&agents=car&methods={mid}").get_json()
    assert r["gt"]
    assert c.get(f"/api/frame?{base}&agents=").status_code == 400
