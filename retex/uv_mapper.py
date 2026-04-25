"""3D-pick to UV(pixel) lookup using barycentric interpolation.

Given a vtkCellPicker hit (cell id + 3D world point), interpolate the per-vertex
UV coords of that cell at the hit position and convert to texture pixel coords.

Also builds a flat array of triangle UV edges for the wireframe overlay in the
UV pane.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv


class UVMapper:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh
        tcoords = mesh.active_texture_coordinates
        self.has_uv = tcoords is not None
        self.tcoords = np.asarray(tcoords) if self.has_uv else None
        self.points = np.asarray(mesh.points)
        # mesh.faces is a flat array: [n0, p0_0, p0_1, ..., n1, p1_0, ...].
        # For typical OBJ scans n is 3.
        self.faces_flat = np.asarray(mesh.faces)

    def cell_vertex_indices(self, cell_id: int) -> np.ndarray | None:
        """Return point indices for cell `cell_id`, or None on failure.

        Uses VTK directly to handle non-uniform cell sizes safely.
        """
        cell = self.mesh.GetCell(cell_id)
        n = cell.GetNumberOfPoints()
        if n < 3:
            return None
        ids = cell.GetPointIds()
        return np.array([ids.GetId(i) for i in range(n)], dtype=np.int64)

    def world_to_uv_px(
        self, cell_id: int, world_xyz: np.ndarray, tex_w: int, tex_h: int
    ) -> tuple[int, int] | None:
        """Hit point on a triangle -> texture pixel coords, or None if no UVs."""
        if not self.has_uv or self.tcoords is None:
            return None
        idx = self.cell_vertex_indices(cell_id)
        if idx is None or len(idx) < 3:
            return None
        # Use the first three vertices of the cell (Artec scans are triangles).
        i0, i1, i2 = int(idx[0]), int(idx[1]), int(idx[2])
        p0, p1, p2 = self.points[i0], self.points[i1], self.points[i2]
        bary = _barycentric(world_xyz, p0, p1, p2)
        if bary is None:
            return None
        uv0, uv1, uv2 = self.tcoords[i0], self.tcoords[i1], self.tcoords[i2]
        u = bary[0] * uv0[0] + bary[1] * uv1[0] + bary[2] * uv2[0]
        v = bary[0] * uv0[1] + bary[1] * uv1[1] + bary[2] * uv2[1]
        # OBJ UVs are y-up; image is y-down.
        px = int(np.clip(u * tex_w, 0, tex_w - 1))
        py = int(np.clip((1.0 - v) * tex_h, 0, tex_h - 1))
        return px, py

    def uv_px_to_uv01(self, px: int, py: int, tex_w: int, tex_h: int) -> tuple[float, float]:
        """Inverse of pixel mapping (for hover labels)."""
        return px / tex_w, 1.0 - (py / tex_h)

    def triangle_uv_edges_px(
        self, tex_w: int, tex_h: int
    ) -> np.ndarray | None:
        """Return (E, 2, 2) array of edge endpoints in pixel coords for overlay.

        E = total edges across all triangles (with duplicates - drawing dupes is fine).
        """
        if not self.has_uv or self.tcoords is None:
            return None
        flat = self.faces_flat
        if flat.size == 0:
            return None
        edges: list[np.ndarray] = []
        i = 0
        n_total = flat.size
        uv = self.tcoords
        while i < n_total:
            n = int(flat[i])
            ids = flat[i + 1 : i + 1 + n]
            i += 1 + n
            if n < 3:
                continue
            pts = uv[ids]  # (n, 2) in [0,1] with y-up
            # Convert to image pixels y-down
            px = pts[:, 0] * tex_w
            py = (1.0 - pts[:, 1]) * tex_h
            xy = np.stack([px, py], axis=1)
            for k in range(n):
                a = xy[k]
                b = xy[(k + 1) % n]
                edges.append(np.stack([a, b]))
        if not edges:
            return None
        return np.stack(edges, axis=0)


def _barycentric(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray | None:
    """Barycentric coords of p w.r.t. triangle (a,b,c) using projected solve.

    Returns array [w0, w1, w2] where p ~= w0*a + w1*b + w2*c, or None if degenerate.
    Coords may go slightly outside [0,1] for off-triangle hits; we clamp at the caller.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)
