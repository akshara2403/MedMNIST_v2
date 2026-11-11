"""
medmnist_replicate.py  -  free-GPU-friendly replication of the official MedMNIST v2
ResNet-50 (224) baseline  (github.com/MedMNIST/experiments, MedMNIST2D/train_and_eval_pytorch.py)

PROTOCOL KEPT IDENTICAL TO THE OFFICIAL SCRIPT
  * data: 28x28 MedMNIST (size=28), upsampled to 224x224 with NEAREST, grayscale copied to 3 ch,
          normalised with mean=.5 / std=.5          (verified bit-identical to the official PIL pipeline)
  * model: torchvision resnet50(weights=None, num_classes=K)  - trained from scratch
  * loss: BCEWithLogits (multi-label: ChestMNIST) / CrossEntropy (everything else)
  * Adam lr=1e-3, batch 128, 100 epochs, MultiStepLR x0.1 at epochs 50 and 75
  * shuffle=True, no drop_last (same batches-per-epoch as the official DataLoader)
  * model selection: the epoch with the highest *validation* AUC  -> report its *test* AUC/ACC
  * metrics: medmnist.Evaluator (the official AUC / ACC code)

ENGINEERING CHANGES (so it fits a free T4 / P100; none of them changes which model is selected)
  1. The whole dataset sits on the GPU as uint8; resize + normalise happen on the GPU.
     (the official code resizes on the CPU with PIL - Colab only has 2 CPU cores, so the GPU starves)
  2. Mixed precision (fp16 autocast) + channels_last. Needed anyway: fp32 at batch 128 / 224px
     does not fit in a 16 GB T4.
  3. The official loop evaluates train+val+test after every epoch, but only val is used for
     selection. We evaluate val every epoch, test only for the selected model (optionally every
     epoch with --log_test, for learning curves).
  4. Checkpoint after every epoch + auto-resume, so Colab disconnects lose at most one epoch.
  5. Fixed seed per run (the official script sets none).
"""
import argparse, json, os, time, math, random, copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import medmnist
from medmnist import INFO, Evaluator

# ResNet-50 (224) numbers from MedMNIST v2, Table 3   (AUC, ACC)
REPORTED_2D = {
    "pathmnist": (0.989, 0.892), "chestmnist": (0.773, 0.948), "dermamnist": (0.912, 0.731),
    "octmnist": (0.958, 0.776), "pneumoniamnist": (0.962, 0.884), "retinamnist": (0.716, 0.511),
    "breastmnist": (0.866, 0.842), "bloodmnist": (0.997, 0.950), "tissuemnist": (0.932, 0.680),
    "organamnist": (0.998, 0.947), "organcmnist": (0.993, 0.911), "organsmnist": (0.975, 0.785),
}
# ResNet-50 + 3D numbers from MedMNIST v2, Table 4   (AUC, ACC)
REPORTED_3D = {
    "organmnist3d": (0.994, 0.883), "nodulemnist3d": (0.875, 0.847), "fracturemnist3d": (0.725, 0.494),
    "adrenalmnist3d": (0.828, 0.745), "vesselmnist3d": (0.907, 0.918), "synapsemnist3d": (0.851, 0.795),
}


# ----------------------------------------------------------------------------------------- data
class GPUSplit:
    """One split of a 28x28 MedMNIST2D dataset held on the GPU as uint8."""

    def __init__(self, npz, split, device):
        imgs = npz[f"{split}_images"]                      # (N,28,28) or (N,28,28,3), uint8
        self.imgs = torch.from_numpy(imgs).to(device)
        self.labels = torch.from_numpy(npz[f"{split}_labels"].astype(np.int64)).to(device)
        self.n = len(self.imgs)

    def batch(self, idx):
        x = self.imgs[idx].float().div_(255.0)
        if x.ndim == 3:                                    # grayscale -> 3 channels (as_rgb)
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        else:                                              # NHWC -> NCHW
            x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=8, mode="nearest")   # 28 -> 224, == PIL.Image.NEAREST
        x = (x - 0.5) / 0.5
        return x.contiguous(memory_format=torch.channels_last), self.labels[idx]


# ------------------------------------------------------------------------------------- helpers
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict(model, split, task, bs=512):
    model.eval()
    out = []
    for s in range(0, split.n, bs):
        idx = torch.arange(s, min(s + bs, split.n), device=split.imgs.device)
        x, _ = split.batch(idx)
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):  # eval in fp16 too
            logits = model(x)
        logits = logits.float()
        out.append(torch.sigmoid(logits) if task == "multi-label, binary-class" else torch.softmax(logits, 1))
    return torch.cat(out).cpu().numpy()


def done_runs(results_path):
    """Set of (data_flag, seed) already finished, read from results.jsonl."""
    s = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            if line.strip():
                r = json.loads(line)
                s.add((r["data_flag"], r["seed"]))
    return s


# ---------------------------------------------------------------------------------------- main
def run(data_flag, seed, out_dir, data_root, num_epochs=100, batch_size=128, lr=1e-3,
        amp=True, log_test=False, max_minutes=None, keep_best_weights=True, verbose=True):
    """Train one (dataset, seed) run. Resumes automatically from out_dir/ckpt if present.
    Returns the result dict when finished, or None if it paused because of max_minutes."""
    t_start = time.time()
    if not torch.cuda.is_available() and not os.environ.get("MEDMNIST_ALLOW_CPU"):
        raise RuntimeError("No GPU - in Colab: Runtime > Change runtime type > T4 GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    os.makedirs(data_root, exist_ok=True)
    for sub in ("ckpt", "predictions", "weights", "history"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    tag = f"{data_flag}_seed{seed}"
    ckpt_path = os.path.join(out_dir, "ckpt", f"{tag}.pt")
    results_path = os.path.join(out_dir, "results.jsonl")
    if (data_flag, seed) in done_runs(results_path):
        print(f"[skip] {tag} already finished"); return None

    info = INFO[data_flag]
    task, n_classes = info["task"], len(info["label"])
    if not os.path.exists(os.path.join(data_root, f"{data_flag}.npz")):
        getattr(medmnist, info["python_class"])(split="train", download=True, root=data_root, size=28)
    npz = np.load(os.path.join(data_root, f"{data_flag}.npz"))
    train, val, test = (GPUSplit(npz, s, device) for s in ("train", "val", "test"))
    val_eval = Evaluator(data_flag, "val", size=28, root=data_root)
    test_eval = Evaluator(data_flag, "test", size=28, root=data_root)

    set_seed(seed)                                        # seeds weight init + batch order
    model = torchvision.models.resnet50(weights=None, num_classes=n_classes)
    model = model.to(device).to(memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * num_epochs), int(0.75 * num_epochs)], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_auc, best_epoch, best_state, history = 0, -1.0, -1, None, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, best_epoch = ck["epoch"] + 1, ck["best_auc"], ck["best_epoch"]
        best_state, history = ck["best_state"], ck["history"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if ck.get("rng_cuda") is not None: torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"[resume] {tag} from epoch {start_epoch}  (best val AUC so far {best_auc:.4f} @ ep {best_epoch})")
    else:
        print(f"[start] {tag}: {train.n} train / {val.n} val / {test.n} test, task={task}, classes={n_classes}")

    multilabel = task == "multi-label, binary-class"
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(train.n, device=device)
        losses = []
        for s in range(0, train.n, batch_size):          # no drop_last, like the official loader
            x, y = train.batch(perm[s:s + batch_size])
            y = y.float() if multilabel else y.squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            losses.append(loss.detach())
        scheduler.step()
        train_loss = torch.stack(losses).mean().item()

        val_auc, val_acc = val_eval.evaluate(predict(model, val, task))
        rec = {"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "val_acc": val_acc,
               "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        if log_test:
            rec["test_auc"], rec["test_acc"] = test_eval.evaluate(predict(model, test, task))
        if not math.isnan(val_auc) and val_auc > best_auc:   # strict '>', same as official
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        history.append(rec)
        if verbose:
            extra = f" test_auc {rec['test_auc']:.4f}" if log_test else ""
            print(f"{tag} ep {epoch+1:3d}/{num_epochs} loss {train_loss:.4f} val_auc {val_auc:.4f}{extra}"
                  f" | best {best_auc:.4f}@{best_epoch+1} | {rec['sec']:.0f}s", flush=True)

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_auc": best_auc, "best_epoch": best_epoch,
                     "best_state": best_state, "history": history,
                     "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ckpt_path)

        if max_minutes is not None and epoch + 1 < num_epochs:
            elapsed = (time.time() - t_start) / 60
            if elapsed + 1.5 * rec["sec"] / 60 > max_minutes:
                print(f"[pause] {tag} after epoch {epoch+1} - time budget reached. Re-run to resume.")
                return None

    # ---- final: evaluate the selected (best-val) model on val + test, save official-format CSVs
    if best_state is None:                                # only if val AUC was NaN every epoch
        print("[warn] val AUC never valid - using the last epoch's weights")
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred_dir = os.path.join(out_dir, "predictions", tag)
    os.makedirs(pred_dir, exist_ok=True)
    v_auc, v_acc = val_eval.evaluate(predict(model, val, task), pred_dir, f"resnet50_224_seed{seed}")
    t_auc, t_acc = test_eval.evaluate(predict(model, test, task), pred_dir, f"resnet50_224_seed{seed}")
    rep_auc, rep_acc = REPORTED_2D[data_flag]
    result = {"data_flag": data_flag, "seed": seed, "model": "resnet50_224", "epochs": num_epochs,
              "best_epoch": best_epoch + 1, "val_auc": v_auc, "val_acc": v_acc,
              "test_auc": t_auc, "test_acc": t_acc, "reported_auc": rep_auc, "reported_acc": rep_acc,
              "amp": amp, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "history", f"{tag}.json"), "w") as f:
        json.dump(history, f)
    if keep_best_weights:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                   os.path.join(out_dir, "weights", f"{tag}_best_fp16.pth"))
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    os.remove(ckpt_path)                                  # free Drive space
    print(f"[done] {tag}: best epoch {best_epoch+1} | test AUC {t_auc:.4f} (reported {rep_auc}) "
          f"| test ACC {t_acc:.4f} (reported {rep_acc})")
    return result


def benchmark(data_root, batch_size=128, steps=30, amp=True):
    """Measure training images/sec for ResNet-50 @224 on this GPU (synthetic data)."""
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), 1e-3); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(batch_size, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (batch_size,), device=device)
    for i in range(steps + 5):
        if i == 5: torch.cuda.synchronize(); t = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    ips = steps * batch_size / (time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2**30
    return ips, mem


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_flag", default="chestmnist")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out_dir", default="./medmnist_out")
    p.add_argument("--data_root", default="./medmnist_data")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_test", action="store_true")
    p.add_argument("--max_minutes", type=float, default=None)
    a = p.parse_args()
    for sd in [int(s) for s in a.seeds.split(",")]:
        r = run(a.data_flag, sd, a.out_dir, a.data_root, a.num_epochs, a.batch_size,
                amp=not a.no_amp, log_test=a.log_test, max_minutes=a.max_minutes)
        if r is None and a.max_minutes is not None and (a.data_flag, sd) not in done_runs(
                os.path.join(a.out_dir, "results.jsonl")):
            break
