# SEM Grain Tool Portable Package

## Goal

Build a folder that can be copied to another Windows computer and run locally without installing Python manually.

## Recommended folder layout

```text
SEM_Grain_Tool/
  env/
    python.exe
    Lib/
    Scripts/
    ...
  cellpose_cache/
    models/
      cpsam
  cellpose_dataset/
  cellpose_results_pretrained/
  coordinate_shape_outputs/
  particle_analysis/
  sem_grain_app.py
  make_cellpose_overlay.py
  analyze_particles.py
  generate_shapes_from_particles.py
  start_sem_grain_app_portable.bat
```

## Build method

Use a portable conda environment. Do not just copy `E:\conda_envs\sem_sam_env` directly unless you have tested it, because some conda packages store absolute paths.

Preferred method:

```powershell
conda activate base
conda install -c conda-forge conda-pack
conda-pack -p E:\conda_envs\sem_sam_env -o sem_sam_env.zip
```

Then unpack `sem_sam_env.zip` into:

```text
SEM_Grain_Tool/env/
```

On the target computer, run once:

```powershell
SEM_Grain_Tool\env\Scripts\conda-unpack.exe
```

Then start the app:

```text
start_sem_grain_app_portable.bat
```

## Important notes

- Keep `cellpose_cache\models\cpsam` inside the package. The file is large, about 1.2 GB.
- The current environment uses CPU PyTorch, so recognition can take 10-25 minutes per full image.
- For GPU acceleration, the target computer needs a compatible NVIDIA driver and a CUDA-enabled PyTorch environment packed instead.
- The app itself does not need a server. It runs as a local Tkinter desktop program.
- The output files are written into the same package folder.
