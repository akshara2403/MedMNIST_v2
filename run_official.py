"""
run_official.py - run the AUTHORS' scripts (github.com/MedMNIST/experiments, patched by patch_official.py)
for one (dataset, seed), then record the result in out/results.jsonl for summarize.py.

  2D:  MedMNIST2D/train_and_eval_pytorch.py --model_flag resnet50 --resize --as_rgb --batch_size 128
  3D:  MedMNIST3D/train_and_eval_pytorch.py --model_flag resnet50 --conv Conv3d --as_rgb --batch_size 32
       (+ --shape_transform for AdrenalMNIST3D and VesselMNIST3D, as stated in the paper)
  both: --num_epochs 100 --seed <s> --amp --eval_every_epoch val --num_workers <w>

  python run_official.py --smoke          # 1-epoch BreastMNIST + VesselMNIST3D check (~2 min on a GPU)
  python run_official.py --data_flag chestmnist --seed 0
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time

import medmnist
from medmnist import INFO
from medmnist_replicate import REPORTED_2D, REPORTED_3D, done_runs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "official_repo")
SHAPE = {"adrenalmnist3d", "vesselmnist3d"}


def ensure_data(flag):
    """The official scripts read from medmnist's default folder (~/.medmnist). Download there once."""
    from medmnist.info import DEFAULT_ROOT
    if not os.path.exists(os.path.join(DEFAULT_ROOT, f"{flag}.npz")):
        getattr(medmnist, INFO[flag]["python_class"])(split="train", download=True, size=28)


def command(flag, seed, out_root, epochs, workers, amp, eval_every):
    is3d = flag.endswith("3d")
    script = os.path.join(REPO, "MedMNIST3D" if is3d else "MedMNIST2D", "train_and_eval_pytorch.py")
    cmd = [sys.executable, "-u", script, "--data_flag", flag, "--output_root", out_root,
           "--num_epochs", str(epochs), "--gpu_ids", os.environ.get("OFFICIAL_GPU_IDS", "0"), "--model_flag", "resnet50", "--as_rgb",
           "--run", f"seed{seed}", "--seed", str(seed), "--num_workers", str(workers),
           "--eval_every_epoch", eval_every]
    bs = os.environ.get("OFFICIAL_BS") or ("32" if is3d else "128")      # env override only for testing
    cmd += ["--conv", "Conv3d", "--batch_size", bs] if is3d else ["--resize", "--batch_size", bs]
    if is3d and flag in SHAPE:
        cmd.append("--shape_transform")
    if amp:
        cmd.append("--amp")
    return cmd, os.path.dirname(script)


def run_official(flag, seed, out_dir, epochs=100, workers=6, amp=True, eval_every="val", record=True):
    epochs = int(os.environ.get("OFFICIAL_EPOCHS", epochs))       # env override only for testing
    results_path = os.path.join(out_dir, "results.jsonl")
    if record and (flag, seed) in done_runs(results_path):
        print(f"[skip] {flag} seed {seed} already finished"); return None
    for sub in ("logs_official", "history", "predictions", "official_runs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    ensure_data(flag)
    tag = f"{flag}_seed{seed}"
    out_root = os.path.abspath(os.path.join(out_dir, "official_runs", f"seed{seed}"))
    cmd, cwd = command(flag, seed, out_root, epochs, workers, amp, eval_every)
    log_path = os.path.join(out_dir, "logs_official", f"{tag}.log")
    print(f"[official] {tag}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    history, best_epoch, final = [], None, {}
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "PYTHONWARNINGS": "ignore"})
        for line in p.stdout:
            log.write(line); log.flush()
            i = line.find("EPOCH_LOG ")            # may follow a tqdm progress bar on the same line
            if i >= 0:
                e = json.loads(line[i + len("EPOCH_LOG "):])
                e = {k: v for k, v in e.items() if not (isinstance(v, float) and v != v)}   # drop NaN
                history.append(e)
                print(f"{tag} ep {e['epoch']+1:3d}/{epochs} loss {e['train_loss']:.4f} val_auc {e['val_auc']:.4f}",
                      flush=True)
            elif re.search(r"cur_best_epoch\s+(\d+)", line):
                best_epoch = int(re.search(r"cur_best_epoch\s+(\d+)", line).group(1))
            else:
                m = re.search(r"(train|val|test)\s+auc:\s*([\d.na]+)\s+acc:\s*([\d.na]+)", line)
                if m:
                    final[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        p.wait()
    if p.returncode != 0 or "test" not in final:
        print(f"[FAILED] {tag} (exit code {p.returncode}) - see {log_path}")
        print("".join(open(log_path).readlines()[-15:]))
        return None
    run_dirs = sorted(glob.glob(os.path.join(out_root, flag, "*")))
    pred_dir = os.path.join(out_dir, "predictions", tag); os.makedirs(pred_dir, exist_ok=True)
    if run_dirs:
        for f in glob.glob(os.path.join(run_dirs[-1], "*.csv")) + glob.glob(os.path.join(run_dirs[-1], "*_log.txt")):
            shutil.copy(f, pred_dir)
    rep = {**REPORTED_2D, **REPORTED_3D}[flag]
    result = {"data_flag": flag, "seed": seed, "engine": "official+patch",
              "model": "resnet50_3d" if flag.endswith("3d") else "resnet50_224", "epochs": epochs,
              "best_epoch": (best_epoch + 1) if best_epoch is not None else None,
              "val_auc": final["val"][0], "val_acc": final["val"][1],
              "test_auc": final["test"][0], "test_acc": final["test"][1],
              "reported_auc": rep[0], "reported_acc": rep[1], "amp": amp, "minutes": (time.time() - t0) / 60,
              "official_output": run_dirs[-1] if run_dirs else None,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if record:
        with open(os.path.join(out_dir, "history", f"{tag}.json"), "w") as f:
            json.dump(history, f)
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    print(f"[done] {tag}: best epoch {result['best_epoch']} | test AUC {result['test_auc']:.4f} (paper {rep[0]}) "
          f"| ACC {result['test_acc']:.4f} (paper {rep[1]}) | {result['minutes']:.0f} min", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_flag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="1 epoch of BreastMNIST and VesselMNIST3D, not recorded")
    a = ap.parse_args()
    if a.smoke:
        ok = True
        for f in ("breastmnist", "vesselmnist3d"):
            r = run_official(f, 0, os.path.join(a.out_dir, "smoke"), epochs=1, workers=a.workers, record=False)
            ok &= r is not None
        print("\nSMOKE TEST", "PASSED - the official code runs on this machine" if ok else "FAILED - send the log to Claude")
    else:
        run_official(a.data_flag, a.seed, a.out_dir, a.epochs, a.workers)
