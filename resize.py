"""Mesh rescaler.

Pick two points on a textured OBJ, type the real-world distance between them,
and save a uniformly rescaled copy with the texture preserved.

Usage:
    python resize.py objects/ball
    python resize.py objects/ball --output objects

Keys (in the viewer):
    P  pick a surface point (twice)
    M  enter the real-world distance for the current pair
    S  save scaled mesh to <output>/<name>_<v1>_<v2>_<mm>mm/
    R  reset picks and scale
    Q  quit

The save folder name encodes the picked vertex indices and entered measurement,
so the original is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

import numpy as np
import pyvista as pv
import vtk

from retex.io_utils import (
    find_obj as _find_obj,
    parse_mtl_textures as _parse_mtl_textures,
    read_obj_vertices as _read_obj_vertices,
    read_obj_with_tcoords as _read_obj_with_tcoords,
)

UNITS = "mm"


class MeshResizer:
    def __init__(self, source_dir: Path, output_root: Path):
        self.source_dir = source_dir
        self.obj_path = _find_obj(source_dir)
        self.name = self.obj_path.stem
        self.mtl_path = source_dir / f"{self.name}.mtl"
        self.tex_path = self._find_texture()
        self.output_root = output_root
        self.original_vertices = _read_obj_vertices(self.obj_path)
        print(f"OBJ has {len(self.original_vertices)} unique vertex positions")

        print(f"Loading {self.obj_path} ...")
        self.mesh = _read_obj_with_tcoords(
            self.obj_path, textured_materials=self._textured_material_names()
        )
        self.texture = pv.read_texture(str(self.tex_path)) if self.tex_path else None
        if self.texture is None:
            print("Warning: no texture found.")
        elif self.mesh.active_texture_coordinates is None:
            print("Warning: mesh has no texture coordinates; texture will not display.")
        else:
            active_name = self.mesh.GetPointData().GetTCoords().GetName()
            print(f"Active TCoords: {active_name!r}")

        self.picks: list[np.ndarray] = []
        self.pick_vertex_indices: list[int] = []  # 1-based, into original_vertices
        self.pick_actors: list = []
        self.line_actor = None
        self.hud_actor = None
        self.scale_factor: float = 1.0
        self.real_distance: float | None = None

        self.plotter = pv.Plotter(window_size=(1280, 860))
        self.plotter.set_background("black")
        mesh_kwargs = dict(reset_camera=True, smooth_shading=True)
        if self.texture is not None and self.mesh.active_texture_coordinates is not None:
            mesh_kwargs["texture"] = self.texture
        else:
            mesh_kwargs["color"] = "lightgray"
        self.plotter.add_mesh(self.mesh, **mesh_kwargs)
        self.plotter.add_axes()
        self.plotter.enable_surface_point_picking(
            callback=self._on_pick,
            show_message=False,
            show_point=False,
        )
        self.plotter.add_key_event("r", self.reset)
        self.plotter.add_key_event("m", self.prompt_measurement)
        self.plotter.add_key_event("s", self.save)
        self._update_hud()

    def _find_texture(self) -> Path | None:
        for _, p in _parse_mtl_textures(self.mtl_path, self.source_dir):
            if p and p.exists():
                return p
        jpg = self.source_dir / f"{self.name}.jpg"
        return jpg if jpg.exists() else None

    def _textured_material_names(self) -> set[str]:
        return {name for name, p in _parse_mtl_textures(self.mtl_path, self.source_dir) if p}

    def run(self):
        self.plotter.show()

    def _pick_radius(self) -> float:
        b = self.mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
        return diag * 0.005

    def _on_pick(self, point):
        if point is None:
            return
        if len(self.picks) >= 2:
            print("Two points already picked. Press R to reset.")
            return
        pt = np.asarray(point, dtype=float).reshape(3)
        self.picks.append(pt)
        v_idx = self._nearest_vertex_index(pt)
        self.pick_vertex_indices.append(v_idx)
        color = "red" if len(self.picks) == 1 else "blue"
        actor = self.plotter.add_mesh(
            pv.Sphere(radius=self._pick_radius(), center=pt),
            color=color,
            reset_camera=False,
            pickable=False,
        )
        self.pick_actors.append(actor)
        print(f"Pick {len(self.picks)}: nearest vertex index = {v_idx}")
        if len(self.picks) == 2:
            self._draw_line()
            print(f"Measured distance: {self._measured_distance():.6f} {UNITS}")
            print("Press M to enter the real-world distance, R to redo.")
        self._update_hud()

    def _nearest_vertex_index(self, pt: np.ndarray) -> int:
        diffs = self.original_vertices - pt
        # 1-based to match OBJ vertex numbering
        return int(np.argmin(np.einsum("ij,ij->i", diffs, diffs))) + 1

    def _draw_line(self):
        if self.line_actor is not None:
            self.plotter.remove_actor(self.line_actor)
        a, b = self.picks
        self.line_actor = self.plotter.add_mesh(
            pv.Line(a, b),
            color="yellow",
            line_width=4,
            reset_camera=False,
            pickable=False,
        )

    def _measured_distance(self) -> float | None:
        if len(self.picks) < 2:
            return None
        return float(np.linalg.norm(self.picks[1] - self.picks[0]))

    def _update_hud(self):
        if self.hud_actor is not None:
            self.plotter.remove_actor(self.hud_actor)
        n = len(self.picks)
        if n == 0:
            step = "Step 1/3 - Press P to pick the FIRST point on the mesh"
        elif n == 1:
            step = "Step 1/3 - Press P to pick the SECOND point on the mesh"
        elif self.real_distance is None:
            step = f"Step 2/3 - Press M to enter the real distance ({UNITS})"
        else:
            step = "Step 3/3 - Press S to save the rescaled mesh"

        lines = [
            f"Mesh Rescaler  -  [{self.name}]  -  units: {UNITS}",
            "",
            step,
            "",
            "Keys:  P pick   M enter real   S save   R reset   Q quit",
            "",
        ]
        d = self._measured_distance()
        if self.pick_vertex_indices:
            vs = ", ".join(f"v{v}" for v in self.pick_vertex_indices)
            lines.append(f"picks:    {vs}")
        if d is not None:
            lines.append(f"measured: {d:.4f} {UNITS}")
        if self.real_distance is not None:
            lines.append(f"real:     {self.real_distance:.4f} {UNITS}")
            lines.append(f"factor:   {self.scale_factor:.6f}  (uniform)")
            lines.append(f"will save -> {self._save_folder_name()}/")
        self.hud_actor = self.plotter.add_text(
            "\n".join(lines),
            position="upper_left",
            font_size=12,
            color="white",
            shadow=True,
        )

    def reset(self):
        for a in self.pick_actors:
            self.plotter.remove_actor(a)
        self.pick_actors = []
        if self.line_actor is not None:
            self.plotter.remove_actor(self.line_actor)
            self.line_actor = None
        self.picks = []
        self.pick_vertex_indices = []
        self.scale_factor = 1.0
        self.real_distance = None
        self._update_hud()
        print("Reset.")

    def prompt_measurement(self):
        d = self._measured_distance()
        if d is None:
            print("Pick two points first (press P twice).")
            return
        root = tk.Tk()
        root.withdraw()
        try:
            val = simpledialog.askfloat(
                f"Rescale ({UNITS})",
                f"Measured on mesh: {d:.4f} {UNITS}\n\n"
                f"Real-world distance in {UNITS}:",
                minvalue=1e-9,
            )
        finally:
            root.destroy()
        if val is None:
            return
        self.real_distance = float(val)
        self.scale_factor = self.real_distance / d
        print(f"Real distance: {self.real_distance:.4f} {UNITS}")
        print(f"Scale factor: {self.scale_factor:.6f}")
        self._update_hud()

    def _save_folder_name(self) -> str:
        v1, v2 = self.pick_vertex_indices
        mm = f"{self.real_distance:g}"
        return f"{self.name}_{v1}_{v2}_{mm}mm"

    def save(self):
        if self.real_distance is None or len(self.pick_vertex_indices) != 2:
            print("Pick two points and press M to set the real distance first.")
            return
        out_dir = self.output_root / self._save_folder_name()
        if out_dir.resolve() == self.source_dir.resolve():
            print(f"Refusing to overwrite the source folder: {out_dir}")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        out_obj = out_dir / self.obj_path.name
        scale_obj_text(self.obj_path, out_obj, self.scale_factor)
        if self.mtl_path.exists():
            shutil.copy2(self.mtl_path, out_dir / self.mtl_path.name)
        if self.tex_path and self.tex_path.exists():
            shutil.copy2(self.tex_path, out_dir / self.tex_path.name)
        v1, v2 = self.pick_vertex_indices
        meta = {
            "factor": self.scale_factor,
            "measured_mm": self._measured_distance(),
            "real_mm": self.real_distance,
            "v1_index": v1,
            "v2_index": v2,
            "v1_xyz_original": self.original_vertices[v1 - 1].tolist(),
            "v2_xyz_original": self.original_vertices[v2 - 1].tolist(),
            "p1_picked_xyz": self.picks[0].tolist(),
            "p2_picked_xyz": self.picks[1].tolist(),
            "source": str(self.obj_path),
            "units": UNITS,
        }
        (out_dir / "scale.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved -> {out_dir}")


def scale_obj_text(src: Path, dst: Path, factor: float) -> None:
    """Multiply x/y/z on every 'v ' line; pass everything else through unchanged.

    Texture coords (vt), normals (vn), faces (f), mtllib, usemtl, comments and
    any Artec-specific headers are written byte-for-byte. Trailing color
    components on vertex lines (v x y z r g b) are preserved unscaled.
    """
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                try:
                    x = float(parts[1]) * factor
                    y = float(parts[2]) * factor
                    z = float(parts[3]) * factor
                except (IndexError, ValueError):
                    fout.write(line)
                    continue
                rest = parts[4:]
                fout.write(
                    "v "
                    + " ".join([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", *rest])
                    + "\n"
                )
            else:
                fout.write(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="Folder containing <name>.obj")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: parent of <source>, e.g. objects/)",
    )
    args = ap.parse_args()
    src = args.source.resolve()
    if not src.is_dir():
        sys.exit(f"Not a directory: {src}")
    out_root = (args.output.resolve() if args.output else src.parent)
    MeshResizer(src, out_root).run()


if __name__ == "__main__":
    main()
