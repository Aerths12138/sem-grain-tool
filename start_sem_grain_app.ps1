$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "E:\conda_envs\sem_sam_env\python.exe"

Set-Location $Root
& $Python (Join-Path $Root "pyinstaller_packaging\sem_grain_web_app.py")
