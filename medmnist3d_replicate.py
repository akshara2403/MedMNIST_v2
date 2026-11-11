"""
medmnist3d_replicate.py  -  replication of the official MedMNIST v2 "ResNet-50 + 3D" baseline
(github.com/MedMNIST/experiments, MedMNIST3D/train_and_eval_pytorch.py  with  --model_flag resnet50 --conv Conv3d)

PROTOCOL KEPT IDENTICAL (paper, "Methods": Baselines + the official script)
  * data: 28x28x28 volumes, value/255, single channel copied to 3 channels (as_rgb), no normalisation
  * shape datasets (AdrenalMNIST3D, VesselMNIST3D): x * U(0,1) per volume when training, x * 0.5 at evaluation
  * model: the official CIFAR-style ResNet-50 (3x3 stem, no max-pool), converted 2D -> 3D exactly like
           acsconv's Conv3dConverter(model, i3d_repeat_axis=-3): Conv2d->Conv3d (k,k,k), BN2d->BN3d,
           AdaptiveAvgPool2d->3d, and the random 2D init "inflated" (repeated k times along depth, / k)
  * CrossEntropy, Adam lr=1e-3, batch 32, 100 epochs, MultiStepLR x0.1 at epochs 50 and 75
  * model selection: best validation AUC -> report test AUC/ACC with medmnist.Evaluator

WHY WE DON'T IMPORT ACSConv: its converter crashes on PyTorch >= 2.x versions that added a `bias`
argument to BatchNorm (it passes the bias *tensor* as the constructor flag). `convert_2d_to_3d` below
does the same conversion; `test_converter_matches_acsconv()` checks it against a patched acsconv.
SyncBatchNorm in the official code is identical to BatchNorm on one GPU, so we keep plain BN.
Engineering changes are the same as the 2D engine: data on GPU, AMP, val-only per-epoch eval,
per-epoch checkpoints + auto-resume, fixed seeds.
"""
import os, json, time, math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import medmnist
from medmnist import INFO, Evaluator

from medmnist_replicate import REPORTED_3D, set_seed, atomic_save, done_runs

SHAPE_DATASETS = {"adrenalmnist3d", "vesselmnist3d"}


# ------------------------------------------------- official model (MedMNIST/experiments, Apache-2.0)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out)
        return self.linear(out.view(out.size(0), -1))


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


# --------------------------------------------------------- 2D -> 3D (== acsconv Conv3dConverter)
def _t(v):
    return v if isinstance(v, tuple) and len(v) == 3 else (v[0] if isinstance(v, tuple) else v,) * 3


def _convert_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new = nn.Conv3d(child.in_channels, child.out_channels, _t(child.kernel_size), _t(child.stride),
                            _t(child.padding), _t(child.dilation), child.groups, child.bias is not None,
                            child.padding_mode)
        elif isinstance(child, nn.BatchNorm2d):
            new = nn.BatchNorm3d(child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats)
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            new = nn.AdaptiveAvgPool3d(_t(child.output_size))
        else:
            _convert_modules(child)
            continue
        setattr(module, name, new)
    return module


def convert_2d_to_3d(model2d, repeat_axis=-3):
    """Conv3dConverter(model, i3d_repeat_axis=-3): convert layers, then inflate the 2D weights."""
    sd = model2d.state_dict()
    model3d = _convert_modules(model2d)
    present = model3d.state_dict()
    for k in list(sd.keys()):
        if sd[k].dim() == 4:
            r = present[k].shape[repeat_axis]
            sd[k] = torch.stack([sd[k]] * r, dim=repeat_axis) / r
    model3d.load_state_dict(sd, strict=True)
    return model3d


def build_model(n_classes, seed):
    set_seed(seed)
    return convert_2d_to_3d(ResNet50(in_channels=3, num_classes=n_classes))


# --------------------------------------------------------------------------------------- data
class GPUSplit3D:
    def __init__(self, npz, split, device):
        self.vols = torch.from_numpy(npz[f"{split}_images"]).to(device)          # (N,28,28,28) uint8
        self.labels = torch.from_numpy(npz[f"{split}_labels"].astype(np.int64)).to(device)
        self.n = len(self.vols)

    def batch(self, idx, mul=None):
        x = self.vols[idx].float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1, -1)  # as_rgb
        if mul == "random":                     # Transform3D(mul='random'): one U(0,1) per volume
            x = x * torch.rand(len(idx), 1, 1, 1, 1, device=x.device)
        elif mul == "0.5":
            x = x * 0.5
        return x.contiguous(), self.labels[idx]


@torch.no_grad()
def predict(model, split, mul, amp, bs=64):
    model.eval()
    out = []
    for s in range(0, split.n, bs):
        idx = torch.arange(s, min(s + bs, split.n), device=split.vols.device)
        x, _ = split.batch(idx, mul)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(torch.softmax(model(x).float(), 1))
    return torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------------------------------- main
def run3d(data_flag, seed, out_dir, data_root, num_epochs=100, batch_size=32, lr=1e-3,
          amp=True, log_test=False, max_minutes=None, keep_best_weights=True, verbose=True):
    t_start = time.time()
    if not torch.cuda.is_available() and not os.environ.get("MEDMNIST_ALLOW_CPU"):
        raise RuntimeError("No GPU visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    for sub in ("ckpt", "predictions", "weights", "history"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    os.makedirs(data_root, exist_ok=True)
    tag = f"{data_flag}_seed{seed}"
    ckpt_path = os.path.join(out_dir, "ckpt", f"{tag}.pt")
    results_path = os.path.join(out_dir, "results.jsonl")
    if (data_flag, seed) in done_runs(results_path):
        print(f"[skip] {tag} already finished"); return None

    info = INFO[data_flag]; n_classes = len(info["label"])
    if not os.path.exists(os.path.join(data_root, f"{data_flag}.npz")):
        getattr(medmnist, info["python_class"])(split="train", download=True, root=data_root, size=28)
    npz = np.load(os.path.join(data_root, f"{data_flag}.npz"))
    train, val, test = (GPUSplit3D(npz, s, device) for s in ("train", "val", "test"))
    val_eval = Evaluator(data_flag, "val", size=28, root=data_root)
    test_eval = Evaluator(data_flag, "test", size=28, root=data_root)
    train_mul, eval_mul = ("random", "0.5") if data_flag in SHAPE_DATASETS else (None, None)

    model = build_model(n_classes, seed).to(device)
    criterion = nn.CrossEntropyLoss()
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
        print(f"[resume] {tag} from epoch {start_epoch} (best val AUC {best_auc:.4f} @ ep {best_epoch+1})")
    else:
        print(f"[start] {tag}: {train.n}/{val.n}/{test.n} train/val/test, classes={n_classes}, "
              f"shape_transform={data_flag in SHAPE_DATASETS}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time(); model.train(); losses = []
        perm = torch.randperm(train.n, device=device)
        for s in range(0, train.n, batch_size):
            x, y = train.batch(perm[s:s + batch_size], train_mul)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y.squeeze(1))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(loss.detach())
        scheduler.step()
        val_auc, val_acc = val_eval.evaluate(predict(model, val, eval_mul, amp))
        rec = {"epoch": epoch, "train_loss": torch.stack(losses).mean().item(), "val_auc": val_auc,
               "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        if log_test:
            rec["test_auc"], rec["test_acc"] = test_eval.evaluate(predict(model, test, eval_mul, amp))
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        history.append(rec)
        if verbose:
            extra = f" test_auc {rec['test_auc']:.4f}" if log_test else ""
            print(f"{tag} ep {epoch+1:3d}/{num_epochs} loss {rec['train_loss']:.4f} val_auc {val_auc:.4f}{extra}"
                  f" | best {best_auc:.4f}@{best_epoch+1} | {rec['sec']:.0f}s", flush=True)
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_auc": best_auc, "best_epoch": best_epoch,
                     "best_state": best_state, "history": history, "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ckpt_path)
        if max_minutes is not None and epoch + 1 < num_epochs:
            if (time.time() - t_start) / 60 + 1.5 * rec["sec"] / 60 > max_minutes:
                print(f"[pause] {tag} after epoch {epoch+1}. Re-run to resume."); return None

    if best_state is None:
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred_dir = os.path.join(out_dir, "predictions", tag); os.makedirs(pred_dir, exist_ok=True)
    v_auc, v_acc = val_eval.evaluate(predict(model, val, eval_mul, amp), pred_dir, f"resnet50_3d_seed{seed}")
    t_auc, t_acc = test_eval.evaluate(predict(model, test, eval_mul, amp), pred_dir, f"resnet50_3d_seed{seed}")
    rep_auc, rep_acc = REPORTED_3D[data_flag]
    result = {"data_flag": data_flag, "seed": seed, "model": "resnet50_3d", "epochs": num_epochs,
              "best_epoch": best_epoch + 1, "val_auc": v_auc, "val_acc": v_acc, "test_auc": t_auc,
              "test_acc": t_acc, "reported_auc": rep_auc, "reported_acc": rep_acc, "amp": amp,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "history", f"{tag}.json"), "w") as f:
        json.dump(history, f)
    if keep_best_weights:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                   os.path.join(out_dir, "weights", f"{tag}_best_fp16.pth"))
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    os.remove(ckpt_path)
    print(f"[done] {tag}: best epoch {best_epoch+1} | test AUC {t_auc:.4f} (reported {rep_auc}) "
          f"| test ACC {t_acc:.4f} (reported {rep_acc})")
    return result


def benchmark3d(batch_size=32, steps=10, amp=True):
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    model = build_model(11, 0).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.rand(batch_size, 3, 28, 28, 28, device=device); y = torch.randint(0, 11, (batch_size,), device=device)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):
        if i == 3: torch.cuda.synchronize(); t = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    return steps * batch_size / (time.time() - t), torch.cuda.max_memory_allocated() / 2**30
