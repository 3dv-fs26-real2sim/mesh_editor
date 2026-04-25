# mesh_editor

Two small interactive tools for working with textured OBJ scans (Artec exports
in particular):

- **`resize.py`** — pick two surface points, type the real-world distance
  between them, save a uniformly rescaled copy of the mesh with texture and
  MTL preserved.
- **`retexture.py`** — paint the texture of a mesh by sampling colors from a
  library of reference images. 3D viewport and UV/texture pane stay in sync;
  brush strokes on the model write into the texture buffer in real time.

Both tools leave the source folder untouched and write a new asset folder with
provenance metadata.

## Setup

```bash
conda env create -f environment.yml
conda activate mesh_editor
```

## Usage

Each tool takes a folder containing `<name>.obj`, `<name>.mtl` and the texture
image referenced by the MTL (or `<name>.jpg` as a fallback).

```bash
python resize.py    objects/ball
python retexture.py objects/ball
```

Output defaults to the parent of the source folder; override with `--output`.

### resize.py keys

| key | action                                          |
|-----|-------------------------------------------------|
| P   | pick a surface point (twice)                    |
| M   | enter the real-world distance for the pair      |
| S   | save the rescaled mesh                          |
| R   | reset picks                                     |
| Q   | quit                                            |

Output folder name encodes the picked vertex indices and the entered
measurement (e.g. `ball_5_8_25mm/`), so the original is never overwritten.

### retexture.py controls

- **B** toggle paint mode, **Space** toggle camera mode
- **[** / **]** shrink / grow brush
- **Ctrl+Z / Ctrl+Shift+Z** undo / redo strokes
- **Ctrl+S** save retextured asset
- Middle-mouse drag pans the UV pane; wheel zooms
- Click an image in the library panel, then click in the enlarged view to
  eyedrop a color

Output folder is timestamped (e.g. `ball_20260425_204300/`) and contains the
unchanged OBJ, a rewritten MTL pointing at a freshly saved PNG, and a
`retex.json` provenance file.

## Layout

```
resize.py        rescale tool entrypoint
retexture.py     retexture tool entrypoint
retex/           shared modules
  io_utils.py    OBJ/MTL loading + texture discovery
  texture_state.py   central HxWx3 uint8 paint buffer + undo/redo
  uv_mapper.py   3D-pick -> UV pixel via barycentric interpolation
  viewport_3d.py pyvistaqt 3D viewport with paint-mode event filter
  viewport_uv.py QGraphicsView UV/texture pane
  library_panel.py   reference-image grid + eyedropper
objects/         sample meshes
```
