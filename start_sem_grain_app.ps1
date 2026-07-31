$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "E:\conda_envs\sem_sam_env\python.exe"

Set-Location $Root
& $Python (Join-Path $Root "sem_grain_app.py")
