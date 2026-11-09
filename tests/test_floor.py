"""Floor mesh round trip and height lookup."""

from pathlib import Path

import numpy as np

from through_the_eyes_of_a_vru.data.floor import (
    grid_to_floor,
    read_ply,
    write_ply,
)


def test_ply_roundtrip_and_height(tmp_path: Path) -> None:
    ys, xs = np.mgrid[0:5, 0:6]
    z = 0.1 * xs  # a ramp along x
    keep = np.ones((4, 5), dtype=bool)
    keep[:, 4] = False  # last column of cells has no faces
    floor = grid_to_floor(10.0, -2.0, 1.0, z, keep)
    path = tmp_path / "floor.ply"
    write_ply(floor, path)
    back = read_ply(path)
    assert (back.nx, back.ny, back.cell) == (6, 5, 1.0)
    assert back.faces.shape == (2 * 4 * 4, 3)
    h = back.height(np.array([[12.5, 0.3], [50.0, 0.0], [14.5, 0.0]]))
    assert np.isclose(h[0], 0.25)
    assert np.isnan(h[1])  # off the grid
    assert np.isnan(h[2])  # cell without faces
