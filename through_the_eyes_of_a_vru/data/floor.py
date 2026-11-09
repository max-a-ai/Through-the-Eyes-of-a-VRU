"""A scene's floor as a triangle mesh (binary PLY, global frame).

The mesh is a regular height grid: ``comment grid x0 y0 cell nx ny`` in
the header gives the grid, vertices are row-major (``y`` rows, ``x``
columns), and vertices that no face uses carry ``z = NaN``. That makes
the height under any point a bilinear lookup while the file stays an
ordinary mesh any viewer can open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
I32 = NDArray[np.int32]


@dataclass(eq=False)
class Floor:
    """Floor mesh plus its grid (global frame)."""

    verts: F32  # (nx * ny, 3)
    faces: I32  # (m, 3)
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    @property
    def z(self) -> F32:
        return self.verts[:, 2].reshape(self.ny, self.nx)

    def height(self, xy: NDArray[np.floating]) -> F64:
        """Bilinear floor height at global ``xy``; NaN off the mesh."""
        z = self.z.astype(np.float64)
        gx = (xy[:, 0] - self.x0) / self.cell
        gy = (xy[:, 1] - self.y0) / self.cell
        i = np.floor(gx).astype(np.int64)
        j = np.floor(gy).astype(np.int64)
        ok = (i >= 0) & (j >= 0) & (i < self.nx - 1) & (j < self.ny - 1)
        i, j = np.clip(i, 0, self.nx - 2), np.clip(j, 0, self.ny - 2)
        fx, fy = gx - i, gy - j
        out = (
            z[j, i] * (1 - fx) * (1 - fy)
            + z[j, i + 1] * fx * (1 - fy)
            + z[j + 1, i] * (1 - fx) * fy
            + z[j + 1, i + 1] * fx * fy
        )
        out[~ok] = np.nan
        return np.asarray(out, dtype=np.float64)


def grid_to_floor(
    x0: float, y0: float, cell: float, z: NDArray[np.floating], keep: object
) -> Floor:
    """Triangulate the cells of ``z`` (ny, nx) where ``keep`` is True.

    ``keep`` is a boolean ``(ny - 1, nx - 1)`` cell mask. Vertices outside
    every kept cell get ``z = NaN``.
    """
    ny, nx = z.shape
    mask = np.asarray(keep, dtype=bool)
    jj, ii = np.nonzero(mask)
    v00 = (jj * nx + ii).astype(np.int32)
    v10, v01, v11 = v00 + 1, v00 + nx, v00 + nx + 1
    faces = np.concatenate(
        [np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)]
    ).astype(np.int32)
    xs = x0 + cell * np.arange(nx)
    ys = y0 + cell * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)
    zz = np.asarray(z, dtype=np.float32).copy()
    used = np.zeros(nx * ny, dtype=bool)
    used[faces.ravel()] = True
    zz.reshape(-1)[~used] = np.nan
    verts = np.stack([gx.ravel(), gy.ravel(), zz.ravel()], 1)
    return Floor(verts.astype(np.float32), faces, x0, y0, cell, nx, ny)


def write_ply(floor: Floor, path: Path) -> None:
    head = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment grid {float(floor.x0)!r} {float(floor.y0)!r} "
        f"{float(floor.cell)!r} {floor.nx} {floor.ny}\n"
        f"element vertex {len(floor.verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(floor.faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    rec = np.zeros(len(floor.faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rec["n"] = 3
    rec["i"] = floor.faces
    with path.open("wb") as f:
        f.write(head.encode("ascii"))
        f.write(floor.verts.astype("<f4").tobytes())
        f.write(rec.tobytes())


def read_ply(path: Path) -> Floor:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    grid = next(ln for ln in lines if ln.startswith("comment grid"))
    x0, y0, cell, nx, ny = grid.split()[2:]
    n_v = int(next(ln for ln in lines if "element vertex" in ln).split()[2])
    n_f = int(next(ln for ln in lines if "element face" in ln).split()[2])
    verts = np.frombuffer(raw, dtype="<f4", count=n_v * 3, offset=end)
    off = end + n_v * 12
    rec = np.frombuffer(
        raw,
        dtype=[("n", "u1"), ("i", "<i4", (3,))],
        count=n_f,
        offset=off,
    )
    return Floor(
        verts.reshape(-1, 3).astype(np.float32),
        rec["i"].astype(np.int32),
        float(x0),
        float(y0),
        float(cell),
        int(nx),
        int(ny),
    )
