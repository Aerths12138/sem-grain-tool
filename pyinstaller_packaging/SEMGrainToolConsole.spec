# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


spec_dir = Path(SPECPATH).resolve()
sys.path.insert(0, str(spec_dir))
from spec_common import prepare_build

config = prepare_build(SPECPATH, DISTPATH, "SEMGrainToolConsole")

a = Analysis(
    [config["script"]],
    pathex=config["pathex"],
    binaries=config["binaries"],
    datas=config["datas"],
    hiddenimports=config["hiddenimports"],
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
    console=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
