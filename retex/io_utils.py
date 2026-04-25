"""Shared OBJ/MTL loading helpers used by resize.py and retexture.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


def find_obj(source_dir: Path) -> Path:
    """Pick <dir>.obj if present, otherwise the first .obj in the folder."""
    preferred = source_dir / f"{source_dir.name}.obj"
    if preferred.exists():
        return preferred
    objs = sorted(source_dir.glob("*.obj"))
    if not objs:
        raise SystemExit(f"No .obj file in {source_dir}")
    return objs[0]


def read_obj_vertices(obj_path: Path) -> np.ndarray:
    """Parse 'v x y z' lines from the OBJ in file order. Returns (N, 3) float array.

    1-based indexing into this array matches OBJ vertex numbering.
    """
    verts: list[list[float]] = []
    with obj_path.open() as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (IndexError, ValueError):
                    continue
    return np.asarray(verts, dtype=float)


def parse_mtl_textures(mtl_path: Path, source_dir: Path) -> list[tuple[str, Path | None]]:
    """Return [(material_name, texture_path_or_None), ...] in MTL order.

    Material names are returned without surrounding quotes.
    """
    out: list[tuple[str, Path | None]] = []
    if not mtl_path.exists():
        return out
    cur: str | None = None
    cur_tex: Path | None = None
    for raw in mtl_path.read_text().splitlines():
        line = raw.strip()
        if line.lower().startswith("newmtl"):
            if cur is not None:
                out.append((cur, cur_tex))
            cur = line.split(None, 1)[1].strip().strip('"')
            cur_tex = None
        elif line.lower().startswith("map_kd"):
            rel = line.split(None, 1)[1].strip()
            p = source_dir / rel
            cur_tex = p if p.exists() else None
    if cur is not None:
        out.append((cur, cur_tex))
    return out


def read_obj_with_tcoords(
    obj_path: Path, textured_materials: set[str] | None = None
) -> pv.PolyData:
    """Load an OBJ via vtkOBJReader and activate the TCoords for the textured material."""
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(obj_path))
    reader.Update()
    mesh = pv.wrap(reader.GetOutput())
    pd = mesh.GetPointData()

    candidates: list[str] = []
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        if arr is not None and arr.GetNumberOfComponents() == 2:
            candidates.append(arr.GetName())

    chosen: str | None = None
    if textured_materials:
        for name in candidates:
            if name.strip('"') in textured_materials:
                chosen = name
                break
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is not None:
        pd.SetActiveTCoords(chosen)
    return mesh


def find_texture(source_dir: Path, name: str, mtl_path: Path) -> Path | None:
    """Return the first existing texture path referenced by the MTL, or <name>.jpg fallback."""
    for _, p in parse_mtl_textures(mtl_path, source_dir):
        if p and p.exists():
            return p
    jpg = source_dir / f"{name}.jpg"
    return jpg if jpg.exists() else None


def textured_material_names(mtl_path: Path, source_dir: Path) -> set[str]:
    return {name for name, p in parse_mtl_textures(mtl_path, source_dir) if p}


class LoadedMesh:
    """Bundle of OBJ + MTL + texture loaded from a source directory."""

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.obj_path = find_obj(source_dir)
        self.name = self.obj_path.stem
        self.mtl_path = source_dir / f"{self.name}.mtl"
        self.tex_path = find_texture(source_dir, self.name, self.mtl_path)
        self.mesh = read_obj_with_tcoords(
            self.obj_path,
            textured_materials=textured_material_names(self.mtl_path, self.source_dir),
        )
        self.original_vertices = read_obj_vertices(self.obj_path)
