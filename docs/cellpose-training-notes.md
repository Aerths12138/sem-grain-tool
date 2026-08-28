# Cellpose-SAM SEM Segmentation Notes

## Current status

- Working environment: `E:\conda_envs\sem_sam_env`
- Cellpose installed in that environment: `cellpose 4.1.1`
- PyTorch in that environment: CPU-only (`torch 2.12.0+cpu`)
- Prepared dataset:
  - image: `E:\deskup\工作\sem识图\cellpose_dataset\1_i311_sem.png`
  - mask: `E:\deskup\工作\sem识图\cellpose_dataset\1_i311_sem_masks.tif`
  - preview: `E:\deskup\工作\sem识图\cellpose_dataset\1_i311_sem_mask_preview.png`
- Prepared scripts:
  - `E:\deskup\工作\sem识图\prepare_cellpose_dataset.py`
  - `E:\deskup\工作\sem识图\run_cellpose_cpsam.ps1`
  - `E:\deskup\工作\sem识图\train_cellpose_cpsam.ps1`

## Blocker

Cellpose-SAM needs the `cpsam` weight file from:

`https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam`

The current local file at:

`E:\deskup\工作\sem识图\cellpose_cache\models\cpsam`

is corrupted/incomplete. PyTorch reports:

`PytorchStreamReader failed reading zip archive: failed finding central directory`

## Next strategies

1. Download `cpsam` outside Codex with a stable browser/download manager, then replace:
   `E:\deskup\工作\sem识图\cellpose_cache\models\cpsam`

2. Use a Hugging Face mirror or proxy if direct Hugging Face downloads keep failing.

3. Install a CUDA-enabled PyTorch build in `E:\conda_envs\sem_sam_env`; otherwise Cellpose-SAM can run, but full-size inference and training will be slow on CPU.

4. After the valid `cpsam` file is in place, run:
   `powershell -ExecutionPolicy Bypass -File E:\deskup\工作\sem识图\run_cellpose_cpsam.ps1`

5. For quick smoke-test fine-tuning after inference works, run:
   `powershell -ExecutionPolicy Bypass -File E:\deskup\工作\sem识图\train_cellpose_cpsam.ps1`

6. For better results, prepare at least 5-20 more SEM images with corresponding instance masks. One annotated image is enough to test the pipeline, but not enough for a robust model.
