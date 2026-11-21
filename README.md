# MedMNIST v2: Reproducing the Baselines and Training One Model on Many Datasets

**Akshara Ramprasad** · M.S. Data Science, Boston University

This project has two parts:

1. **Reproducing a paper.** I reproduced the ResNet-50 results of the MedMNIST v2 paper
   ([Yang et al., *Scientific Data*, 2023](https://doi.org/10.1038/s41597-022-01721-8)) on all
   18 datasets, using the authors' own code, with 5 runs per dataset (90 runs in total).
2. **Training one model on many datasets.** Building on Foundation Ark
   ([Ma et al., MICCAI 2023](https://arxiv.org/abs/2310.09507)), I trained a single model on
   all 12 2D datasets, one on all 6 3D datasets, and one on all 18 together.

The full write-up is in the report: `MedMNIST_v2_Report_Akshara.pdf`.

---

## Key results

| | Result |
|---|---|
| **ChestMNIST** (5 runs) | test AUC **0.7695 ± 0.0016**, paper 0.773 |
| **All 18 datasets** | **15 of 18** within 1% of the paper |
| **One model, 12 2D datasets** | mean AUC **0.926**, vs. 0.918 for 12 separate models (paper: 0.923) |
| **One model, all 18 datasets** | mean AUC **0.895**, vs. 0.900 for 18 separate models |
| **Biggest gain** | BreastMNIST: 0.828 → **0.911** when trained together with the other datasets |

---

## Part 1: Reproducing the baselines

I ran the authors' training scripts from
[MedMNIST/experiments](https://github.com/MedMNIST/experiments) (commit `70b6b3a7`) with the
same settings as the paper:

- **2D:** ResNet-50, images resized from 28×28 to 224×224
- **3D:** ResNet-50 converted to 3D
- **Training:** Adam, learning rate 0.001, 100 epochs; the model is picked by its best validation AUC
- **Runs:** 5 per dataset (seeds 0 to 4)

To run 90 trainings in parallel on 9 GPUs, I made a **small patch** to the scripts
(`official_patch.diff`). It adds faster mixed-precision training, fixed random seeds,
parallel data loading, and validation-only scoring during training. Every option is off by default,
and the patch does not change the data, the model, or how the best epoch is picked.

**Results (test AUC):**

| Dataset | Paper | Mine (mean ± std of 5 runs) | Within 1%? |
|---|---|---|---|
| PathMNIST | 0.989 | 0.988 ± 0.005 | ✅ |
| ChestMNIST | 0.773 | 0.769 ± 0.002 | ✅ |
| DermaMNIST | 0.912 | 0.911 ± 0.002 | ✅ |
| OCTMNIST | 0.958 | 0.959 ± 0.003 | ✅ |
| PneumoniaMNIST | 0.962 | 0.957 ± 0.008 | ✅ |
| RetinaMNIST | 0.716 | 0.717 ± 0.016 | ✅ |
| BreastMNIST | 0.866 | 0.828 ± 0.019 | ❌ |
| BloodMNIST | 0.997 | 0.997 ± 0.000 | ✅ |
| TissueMNIST | 0.932 | 0.928 ± 0.002 | ✅ |
| OrganAMNIST | 0.998 | 0.997 ± 0.000 | ✅ |
| OrganCMNIST | 0.993 | 0.993 ± 0.001 | ✅ |
| OrganSMNIST | 0.975 | 0.973 ± 0.003 | ✅ |
| OrganMNIST3D | 0.994 | 0.993 ± 0.002 | ✅ |
| NoduleMNIST3D | 0.875 | 0.892 ± 0.020 | ✅ |
| FractureMNIST3D | 0.725 | 0.690 ± 0.033 | ❌ |
| AdrenalMNIST3D | 0.828 | 0.842 ± 0.043 | ✅ |
| VesselMNIST3D | 0.907 | 0.894 ± 0.025 | ❌ |
| SynapseMNIST3D | 0.851 | 0.877 ± 0.022 | ✅ |

The three datasets that miss are very small (156 to 382 test images). Their results vary more
from run to run than the 1% margin itself, so the gap is within normal run-to-run variation.

---

## Part 2: One model for many datasets

**The idea.** Instead of 18 separate models, one shared ResNet-50 learns from every dataset.
Each dataset keeps its own small output layer (its own "head"), so no labels need to be merged.
Following Ark, a slowly updated **teacher** copy of the model guides the **student** during training.

**What went wrong first.** Training the datasets one after another, as in Ark, failed. The model
forgot almost everything except the last datasets it saw: the mean validation AUC was about 0.6.

**What I changed:**

1. **Mix the datasets.** Each training step uses a batch from a randomly chosen dataset, so no
   dataset is forgotten.
2. **Give each dataset its own batch normalization.** X-rays, pathology slides and CT scans look
   very different, so each dataset keeps its own normalization statistics. All the
   other weights are shared.
3. **Refresh the batch-norm statistics before every evaluation,** so they match the current model.
4. **Balance the datasets.** Each dataset gets at most 300 training steps per round, so large
   datasets cannot take over.
5. **Use 2D and 3D together.** ACS convolutions
   ([Yang et al., 2021](https://doi.org/10.1109/JBHI.2021.3049452)) let the same 2D filters process
   3D volumes, so all 18 datasets can share one model.

**Results (mean test AUC):**

| Model | Separate models | One joint model |
|---|---|---|
| 12 2D datasets | 0.918 | **0.926** |
| 6 3D datasets | 0.865 | 0.837 |
| All 18 datasets | 0.900 | 0.895 |

- **Small datasets gain the most.** BreastMNIST rises from 0.828 to 0.911, DermaMNIST by +0.022,
  and PneumoniaMNIST by +0.013.
- **2D helps 3D.** In the all-18 model, OrganMNIST3D and NoduleMNIST3D do better than in the
  3D-only model.
- **One limitation: negative transfer.** SynapseMNIST3D, the only electron-microscopy dataset,
  drops from 0.877 to 0.744. Without it, the 3D gap is only −0.007.

---

## Files

| File | What it does |
|---|---|
| `setup_official.sh`, `patch_official.py` | Download the authors' code at a fixed version and apply the patch |
| `official_patch.diff` | Every line I changed in the authors' scripts |
| `run_official.py` | Runs the authors' script for one dataset and seed, and records the result |
| `run_queue.py`, `launch_workers.sh` | Run all jobs on several GPUs, one job per GPU |
| `medmnist_joint.py` | **Part 2:** the joint model and its training |
| `medmnist_replicate.py`, `medmnist3d_replicate.py` | Fast GPU data loading and the 3D model used by `medmnist_joint.py` |
| `summarize.py` | Builds the result tables, figures and spreadsheet |
| `check_env.py` | Checks the GPU and estimates training time |
| `performance_table_Akshara.xlsx` | All results: paper (P1), my runs (P2), joint models (P3, P4), and the gains |
| `summary_93.txt` | Text summary of all results |
| `results_final.tgz` | Raw results, training histories and logs |
| `logs/` | Training logs from the GPU workers |

## How to run

```bash
pip install medmnist scikit-learn pandas matplotlib openpyxl tqdm
bash setup_official.sh          # get and patch the authors' code
python run_official.py --smoke  # 2-minute check that everything works
bash launch_workers.sh          # run all 93 jobs on every available GPU
python summarize.py --out_dir out --xlsx performance_table_Akshara.xlsx
```

I ran everything on a cloud machine with 9 NVIDIA A40 GPUs.

## References

- J. Yang et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification.* Scientific Data, 2023.
- D. Ma, J. Pang, M. B. Gotway, J. Liang. *Foundation Ark: Accruing and reusing knowledge for
  superior and robust performance.* MICCAI, 2023.
- J. Yang et al. *Reinventing 2D convolutions for 3D images.* IEEE JBHI, 2021.
- W.-G. Chang et al. *Domain-specific batch normalization for unsupervised domain adaptation.*
  CVPR, 2019.
