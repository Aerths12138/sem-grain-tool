$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "E:\conda_envs\sem_sam_env\python.exe"
$Cache = Join-Path $Root "cellpose_cache"
$ModelCache = Join-Path $Cache "models"
$Dataset = Join-Path $Root "cellpose_dataset"

New-Item -ItemType Directory -Path $Cache -Force | Out-Null
New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null

$env:USERPROFILE = $Cache
$env:CELLPOSE_LOCAL_MODELS_PATH = $ModelCache

& $Python -m cellpose `
  --train `
  --dir $Dataset `
  --pretrained_model cpsam `
  --mask_filter _masks `
  --learning_rate 1e-5 `
  --n_epochs 50 `
  --train_batch_size 1 `
  --min_train_masks 5 `
  --model_name_out sem_i311_cpsam `
  --verbose
