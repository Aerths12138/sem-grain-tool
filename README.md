# SEM Grain Tool

SEM Grain Tool is a local Windows desktop workflow for SEM particle and grain
segmentation, measurement, correction, and COMSOL-oriented geometry export.

## Features

- CPSAM/Cellpose grain segmentation on CPU
- automatic and manual two-point scale-bar calibration
- particle count, centroid, area, and equivalent-diameter statistics
- particle-size distribution CSV and chart output
- manual mask correction: delete, merge, add, split, undo, and redo
- non-overlapping circle reconstruction with configurable area scale and spacing
- millimeter-calibrated circle-only DXF export with a matching metadata JSON file
- cancellable helper processes with per-task timeouts and process-tree cleanup

## Source layout

- `pyinstaller_packaging/sem_grain_app_frozen.py`: packaged desktop application
- `analyze_particles.py`: particle measurements and distributions
- `detect_scale_bar.py`: automatic scale-bar detection
- `generate_shapes_from_particles.py`: circle reconstruction and DXF export
- `generate_sintered_agglomerates.py`: sintered-agglomerate geometry generation
- `pyinstaller_packaging/`: CPU build, tests, and packaged acceptance tooling

## Large files

The repository intentionally excludes SEM images, generated results, the portable
Python environment, PyInstaller output, and the CPSAM model. For a local build, place
the model at:

```text
cellpose_cache/models/cpsam
```

The complete USB-portable release is generated under `dist/SEMGrainTool/` and must be
copied as a whole directory. It is not stored in normal Git history.

## CPU build

The established build expects a project-local CPU environment at
`SEM_Grain_Tool_Portable/env/`:

```powershell
.\SEM_Grain_Tool_Portable\env\python.exe pyinstaller_packaging\build_cpu_release.py
```

The build stops if Torch is CUDA-enabled. After PyInstaller completes, packaged
acceptance verifies CPU Torch, CPSAM, Tcl/Tk, scale calibration, particle CSV output,
circle preview, and millimeter DXF output.

See [pyinstaller_packaging/README.md](pyinstaller_packaging/README.md) for focused test
and acceptance commands.
