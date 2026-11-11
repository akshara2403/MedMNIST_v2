# MedMNIST pre-warm-up exercise: complete code kit (Steps a, b, c)

Prof. Liang's exercise, based on MedMNIST v2 (Yang et al., *Sci Data* 2023):
- **(a)** ResNet-50 (224) on ChestMNIST, 5 runs: mean, std, gap to the paper.
- **(b)** the same for all 12 2D datasets (ResNet-50 at 224) and all 6 3D datasets (ResNet-50+3D).
- **(c)** one model for all 12 2D datasets, one for all 6 3D datasets (P3), and one for all 18 (P4).
- **(d)** the Overleaf report. `summarize.py` writes `results_table.tex` and the figures for it.

## Steps (a) and (b) use the authors' own code
As the professor asked, (a) and (b) run the scripts released by the authors
(github.com/MedMNIST/experiments, pinned at commit `70b6b3a7`):
- `MedMNIST2D/train_and_eval_pytorch.py --model_flag resnet50 --resize --as_rgb` (batch 128, 100 epochs)
- `MedMNIST3D/train_and_eval_pytorch.py --model_flag resnet50 --conv Conv3d --as_rgb` (batch 32, 100 epochs),
  plus `--shape_transform` for AdrenalMNIST3D and VesselMNIST3D (as stated in the paper)

`setup_official.sh` clones the repo and `patch_official.py` applies a **small, documented patch**: about 55
lines per script, each marked `[warm-up patch]`, every option off by default. The full diff is saved to
`official_patch.diff`. Without the new flags the scripts behave exactly as the originals. We switch on:

| flag | why |
|---|---|
| `--amp` | mixed precision: about 2× faster on the GPU |
| `--eval_every_epoch val` | the original also scores train and test after every epoch, but selects the model by val AUC only, so the same model is selected. Final train/val/test evaluation is unchanged |
| `--num_workers N` | the original resizes 28→224 on one CPU core; parallel loading keeps the GPU busy |
| `--seed s` | seeds 0–4 for reproducible runs (the original sets none) |
| (always) | the original overwrites `CUDA_VISIBLE_DEVICES`, which would put all 8 parallel jobs on GPU 0 |

`setup_official.sh` also fixes one line in the installed ACSConv package. Its converter crashes on current
PyTorch: it passes BatchNorm's bias tensor where a True/False flag is expected.

## Files
| file | step | what it does |
|---|---|---|
| `setup_official.sh`, `patch_official.py` | a, b | get, pin and patch the authors' code |
| `run_official.py` | a, b | runs the authors' script for one (dataset, seed); `--smoke` = 2-minute check |
| `medmnist_joint.py` | c | Ark-style joint training: P3-2D, P3-3D, P4. The method is described in its docstring |
| `run_queue.py`, `launch_workers.sh` | all | 93 jobs (90 runs + 3 joint models), one worker per GPU, longest first |
| `check_env.py` | — | GPU check, data download, speed, and **time and cost estimate** |
| `summarize.py` | a–d | tables, gaps to the paper, P3/P4 gains, curves, `results_table.tex`, fills the spreadsheet |
| `medmnist_replicate.py`, `medmnist3d_replicate.py` | (c) | GPU data pipeline and 3D model used by the joint code; also a faster stand-alone engine (`--engine fast`), not used by default |
| `performance_table_Akshara.xlsx` | — | the professor's sheet: P1 pre-filled, P2/std/gains are formulas |

## Run everything on RunPod (one multi-GPU pod)
Open the pod's **Jupyter Lab** (pod page → *HTTP Services*), drag this zip into the file browser, then
**File → New → Terminal**:
```bash
mkdir -p kit && cd kit
python -m zipfile -e ../medmnist_warmup_kit.zip .
apt-get update -qq && apt-get install -y -qq tmux git
pip install -q medmnist scikit-learn pandas matplotlib openpyxl tqdm
bash setup_official.sh                 # authors' code + patch
nvidia-smi -L
python check_env.py --3d               # ~3 min: speed + time and cost estimate
python run_official.py --smoke         # ~2 min: the authors' 2D and 3D scripts run one epoch each
```
Both must end without errors (the smoke test prints `SMOKE TEST PASSED`). **Send the output to Claude.**
**Make sure your RunPod balance covers the estimate plus ~$10**: if the balance hits $0 the pod stops and the
container disk is erased. Then:
```bash
bash launch_workers.sh                 # Steps a, b and c on all GPUs
#  cheaper/faster Step (c):  bash launch_workers.sh all "--joint_rounds 60"
```
You can close the browser now. To check on it later (from the `kit` folder):
```bash
tmux attach -t workers                 # Ctrl-b n = next GPU window, Ctrl-b d = detach
wc -l out/results.jsonl                # 93 lines = everything finished
tail -n 3 logs/worker_gpu*.log
```
If anything crashes, run `bash launch_workers.sh` again. Finished jobs are skipped and the joint models
resume from their last round. An individual official run restarts from epoch 1, because the authors'
script has no checkpoints.

## When it is finished (93 lines in results.jsonl)
```bash
python summarize.py --out_dir out --xlsx performance_table_Akshara.xlsx
tar czf results_all.tgz out/results.jsonl out/joint_results.jsonl out/summary_*.csv out/results_table.tex \
    out/*.png out/history out/predictions out/logs_official logs official_patch.diff performance_table_Akshara.xlsx
```
Download `results_all.tgz` (right-click → Download), check that it opens on your laptop, then
**Terminate** the pod (the trash icon). *Stop* still bills for storage.

## Notes for the report
- MedMNIST v2.2.4 removed a few blank samples from OrganA/C/S, Organ3D and Vessel3D, so today's data
  differs very slightly from the data used for the paper's tables.
- Step (c) runs once (seed 0). P3 for the 2D columns comes from `joint_p3_2d`, for the 3D columns from
  `joint_p3_3d`; P4 comes from `joint_p4`.
- The joint code feeds images through the same pre-processing as the authors' code, done on the GPU.
  `check_env.py` checks that it matches the official pipeline up to float rounding.
