# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules


datas = [
    (r"E:\deskup\工作\sem识图\make_cellpose_overlay.py", "."),
    (r"E:\deskup\工作\sem识图\analyze_particles.py", "."),
    (r"E:\deskup\工作\sem识图\detect_scale_bar.py", "."),
    (r"E:\deskup\工作\sem识图\generate_shapes_from_particles.py", "."),
    (r"E:\deskup\工作\sem识图\generate_sintered_agglomerates.py", "."),
    (r"E:\deskup\工作\sem识图\cellpose_cache\models\cpsam", r"cellpose_cache\models"),
    (r"E:\deskup\工作\sem识图\SEM_Grain_Tool_Portable\env\Library\lib\tcl8.6", r"_tcl_data\tcl8.6"),
    (r"E:\deskup\工作\sem识图\SEM_Grain_Tool_Portable\env\Library\lib\tk8.6", r"_tcl_data\tk8.6"),
]
binaries = [
    (r"E:\deskup\工作\sem识图\SEM_Grain_Tool_Portable\env\Library\bin\tcl86t.dll", "."),
    (r"E:\deskup\工作\sem识图\SEM_Grain_Tool_Portable\env\Library\bin\tk86t.dll", "."),
]

hiddenimports = [
    "cellpose.__main__",
    "torch",
    "cv2",
    "tifffile",
    "skimage.measure",
    "skimage.segmentation",
    "ezdxf",
    "PIL.ImageTk",
]

for package in ("cellpose", "torch", "skimage", "cv2", "tifffile", "ezdxf"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("cellpose")

a = Analysis(
    ["sem_grain_app_frozen.py"],
    pathex=[r"E:\deskup\工作\sem识图\pyinstaller_packaging"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SEMGrainToolConsole",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SEMGrainToolConsole",
)
