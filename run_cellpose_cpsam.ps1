$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "E:\conda_envs\sem_sam_env\python.exe"
$Cache = Join-Path $Root "cellpose_cache"
$ModelCache = Join-Path $Cache "models"
$Dataset = Join-Path $Root "cellpose_dataset"
$Out = Join-Path $Root "cellpose_results_pretrained"

New-Item -ItemType Directory -Path $Cache -Force | Out-Null
New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null
New-Item -ItemType Directory -Path $Out -Force | Out-Null

$env:USERPROFILE = $Cache
$env:CELLPOSE_LOCAL_MODELS_PATH = $ModelCache

& $Python -m cellpose `
  --image_path (Join-Path $Dataset "1_i311_sem.png") `
  --pretrained_model cpsam `
  --diameter 90 `
  --save_tif `
  --save_png `
  --save_outlines `
  --savedir $Out `
  --verbose
