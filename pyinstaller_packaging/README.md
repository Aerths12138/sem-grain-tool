# PyInstaller Packaging Attempt

This folder is isolated from the existing portable conda package.

Build CPU version:

```powershell
cd /d E:\deskup\工作\sem识图\pyinstaller_packaging
E:\conda_envs\sem_sam_env\Scripts\pyinstaller.exe --clean SEMGrainTool.spec
```

Expected output:

```text
E:\deskup\工作\sem识图\pyinstaller_packaging\dist\SEMGrainTool\SEMGrainTool.exe
```

Notes:

- This build uses the current CPU PyTorch environment.
- The large `cpsam` model is included as app data.
- The app is built as `onedir`, not `onefile`, because PyTorch and Cellpose
  ship many dynamic libraries and a large model.
- GPU support should be built from a separate CUDA-enabled environment, then
  exposed as a second runtime option or a second installer.
