"""
medmnist_joint.py  -  Step (c): ONE model trained on many MedMNIST datasets together.

  P3-2D : one model, all 12 MedMNIST2D datasets      (python medmnist_joint.py --mode p3_2d)
  P3-3D : one model, all  6 MedMNIST3D datasets      (python medmnist_joint.py --mode p3_3d)
  P4    : one model, all 18 datasets (2D + 3D)       (python medmnist_joint.py --mode p4)

METHOD  ("Ark-MedMNIST"), built on two GitHub code bases:
  * Foundation Ark (Ma et al., MICCAI 2023; github.com/JLiangLab/Ark), Ark_MICCAI2023/{models,trainer,engine}.py
      - shared backbone + linear projector (1376-d) + ONE HEAD PER DATASET (no label merging):
        ChestMNIST keeps its 14 multi-label outputs, the others keep their own classes
      - CYCLIC training: every round visits the datasets one after another ("one task epoch" each)
      - TEACHER = EMA of the student, updated after every task epoch; momentum cosine 0.9 -> 1
      - CONSISTENCY loss: MSE(student projection, teacher projection); weight cosine 0 -> 0.5;
        loss = (1-c) * task loss + c * consistency          (exactly Ark's trainer.py)
  * ACSConv (Yang et al., IEEE JBHI 2021; github.com/M3DV/ACSConv): 2D kernels applied to 3D volumes by
    splitting the output channels into axial / coronal / sagittal groups.

OUR ADDITIONS (the "new" part, to discuss in the report)
  1. INTERLEAVED schedule (default): within a round every step draws a batch from a randomly chosen dataset
     (each dataset keeps its share of steps). Ark's cyclic schedule (--schedule cyclic) trains one dataset
     after another; trained from scratch on these very different image types it forgets: in our first runs
     only the last datasets of the round (the three CT organ sets, val AUC 0.94-0.99) survived and the
     others fell to ~0.5 (mean val AUC ~0.6). The teacher still gets one EMA update per dataset share of
     each round, with Ark's momentum and consistency schedules.
  2. Dataset-specific batch norm (DSBN, Chang et al., CVPR 2019): each dataset has its own BatchNorm
     statistics and affine parameters (<0.5% of the weights); all convolutions are shared. One shared BN
     would mix the statistics of colour pathology, X-ray, OCT and CT images.
  3. Task-balanced rounds: each dataset gets at most --max_iters_per_task iterations per round, so the
     165k-image TissueMNIST cannot dominate the 546-image BreastMNIST, and the cost of a round stays bounded.
  4. P4: one backbone for 2D AND 3D. Every conv keeps 2D weights; 2D images use them as normal 2D
     convolutions, 3D volumes use them as ACS convolutions -> all conv weights are shared across the 18
     datasets (2D datasets get 2D BNs, 3D datasets 3D BNs).
  6. BN recalibration before every evaluation (recalibrate_bn): fresh BatchNorm statistics for the current
     weights, because per-dataset BN statistics are updated only ~1/T of the steps and go stale.
  5. One checkpoint for all datasets: after each round, the student and the teacher are evaluated on all
     validation sets; the single model with the best MEAN validation AUC is kept and reported on test.

KEPT FROM THE BASELINES so P3/P4 are comparable with P2: same data and pre-processing (2D: 28->224 nearest,
3 channels; 3D: 28^3, 3 channels, shape-transform for Adrenal/Vessel), Adam lr 1e-3, batch 128 (2D) / 32 (3D),
LR x0.1 at 50% and 75% of the rounds, no data augmentation, medmnist.Evaluator metrics.
Backbones: P3-2D torchvision ResNet-50 (as the 2D baseline); P3-3D the official 3D ResNet-50 (as the 3D
baseline); P4 torchvision ResNet-50 with dual 2D/ACS convs, 3D volumes upsampled 28->56 (nearest, x2).
"""
import argparse, copy, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import medmnist
from medmnist import INFO, Evaluator

from medmnist_replicate import REPORTED_2D, REPORTED_3D, GPUSplit, set_seed, atomic_save, done_runs
from medmnist3d_replicate import GPUSplit3D, ResNet50 as ResNet50_3D_official, convert_2d_to_3d, SHAPE_DATASETS

AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

TASKS_2D = list(REPORTED_2D)
TASKS_3D = list(REPORTED_3D)
MODES = {"p3_2d": TASKS_2D, "p3_3d": TASKS_3D, "p4": TASKS_2D + TASKS_3D}


# ------------------------------------------------- ACS convolution (from github.com/M3DV/ACSConv, Apache-2.0)
def _conv3d_out(i, shape, k, d, p, s):
    return math.floor((shape[i] - k[i] - (d[i] - 1) * (k[i] - 1) + 2 * p[i]) / s[i]) + 1


def acs_conv_f(x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, split):
    """Verbatim logic of acsconv.operators.functional.acs_conv_f (standard convs, groups=1)."""
    _, _, *ishape = x.shape
    oshape = tuple(_conv3d_out(i, ishape, kernel_size, dilation, padding, stride) for i in range(3))
    wa = weight[0:split[0]].unsqueeze(2)
    wc = weight[split[0]:split[0] + split[1]].unsqueeze(3)
    ws = weight[split[0] + split[1]:].unsqueeze(4)
    k, p, s = kernel_size, padding, stride
    out = []
    if split[0] > 0:
        xa = x if oshape[0] == ishape[0] or 2 * oshape[0] == ishape[0] else \
            F.pad(x, (0, 0, 0, 0, p[0], p[0]))[:, :, k[0] // 2:k[0] // 2 + (oshape[0] - 1) * s[0] + 1]
        out.append(F.conv3d(xa, wa, None, s, (0, p[1], p[2]), dilation, groups))
    if split[1] > 0:
        xc = x if oshape[1] == ishape[1] or 2 * oshape[1] == ishape[1] else \
            F.pad(x, (0, 0, p[1], p[1]))[:, :, :, k[1] // 2:k[1] // 2 + s[1] * (oshape[1] - 1) + 1]
        out.append(F.conv3d(xc, wc, None, s, (p[0], 0, p[2]), dilation, groups))
    if split[2] > 0:
        xs = x if oshape[2] == ishape[2] or 2 * oshape[2] == ishape[2] else \
            F.pad(x, (p[2], p[2]))[:, :, :, :, k[2] // 2:k[2] // 2 + s[2] * (oshape[2] - 1) + 1]
        out.append(F.conv3d(xs, ws, None, s, (p[0], p[1], 0), dilation, groups))
    f = torch.cat(out, 1)
    if bias is not None:
        f = f + bias.view(1, out_channels, 1, 1, 1)
    return f


class DualConv(nn.Module):
    """One set of 2D kernels: Conv2d for 4D input, ACS conv for 5D input."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert conv.groups == 1
        self.weight, self.bias = conv.weight, conv.bias
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        self.k3 = (conv.kernel_size[0],) * 3
        self.s3, self.p3, self.d3 = (conv.stride[0],) * 3, (conv.padding[0],) * 3, (conv.dilation[0],) * 3
        o = conv.out_channels
        self.out_channels = o
        self.split = (o // 3 + (o % 3 > 0), o // 3 + (o % 3 > 1), o // 3)   # == ACSConv's default split

    def forward(self, x):
        if x.dim() == 4:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return acs_conv_f(x, self.weight, self.bias, self.k3, self.d3, self.p3, self.s3, 1, self.out_channels, self.split)


class ModalityBN(nn.Module):
    """Separate BatchNorm (statistics + affine) for 2D and 3D inputs."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.bn2d = bn
        self.bn3d = nn.BatchNorm3d(bn.num_features, bn.eps, bn.momentum)

    def forward(self, x):
        return self.bn2d(x) if x.dim() == 4 else self.bn3d(x)


class DualMaxPool(nn.Module):
    def __init__(self, mp):
        super().__init__(); self.k, self.s, self.p = mp.kernel_size, mp.stride, mp.padding

    def forward(self, x):
        return (F.max_pool2d if x.dim() == 4 else F.max_pool3d)(x, self.k, self.s, self.p)


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=tuple(range(2, x.dim())), keepdim=True)


def make_dual(module):
    for name, child in module.named_children():
        if isinstance(child, TaskBN):            # already 2D/3D-specific: leave it alone
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, DualConv(child))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, ModalityBN(child))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, DualMaxPool(child))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, GlobalAvgPool())
        else:
            make_dual(child)
    return module


# ------------------------------------------------------------------------------- Ark-style network
class TaskBN(nn.Module):
    """Domain-specific batch norm (Chang et al., CVPR 2019): one BatchNorm (running statistics + affine) per
    dataset; every convolution stays shared. Needed because with cyclic training a single BN's running
    statistics end up describing only the LAST dataset of the round, which breaks evaluation on the others.
    (Ark itself uses a Swin Transformer with LayerNorm, so it does not have this problem.)"""

    def __init__(self, bn, task_is3d, ctx):
        super().__init__()
        self.ctx = ctx                                  # shared dict {"task": index}, set by ArkNet.forward
        self.bns = nn.ModuleList((nn.BatchNorm3d if d else nn.BatchNorm2d)(bn.num_features, bn.eps, bn.momentum)
                                 for d in task_is3d)

    def forward(self, x):
        return self.bns[self.ctx["task"]](x)


def make_task_bn(module, task_is3d, ctx):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm3d)):
            setattr(module, name, TaskBN(child, task_is3d, ctx))
        else:
            make_task_bn(child, task_is3d, ctx)
    return module


class ArkNet(nn.Module):
    def __init__(self, backbone, feat_dim, n_classes_list, proj_dim=1376, ctx=None):
        super().__init__()
        self.ctx = ctx if ctx is not None else {"task": 0}
        self.backbone = backbone
        self.projector = nn.Linear(feat_dim, proj_dim)                       # Ark: Linear projector
        self.heads = nn.ModuleList(nn.Linear(proj_dim, k) for k in n_classes_list)   # Ark: omni_heads

    def forward(self, x, head):
        self.ctx["task"] = head                             # selects the task-specific BatchNorms
        z = self.projector(torch.flatten(self.backbone(x), 1))
        return z, self.heads[head](z)


def build_model(mode, n_classes_list, seed, task_is3d=None):
    set_seed(seed)
    task_is3d = task_is3d or [mode == "p3_3d"] * len(n_classes_list)
    ctx = {"task": 0}
    if mode == "p3_2d":
        bb = torchvision.models.resnet50(weights=None); bb.fc = nn.Identity()
    elif mode == "p3_3d":
        bb = convert_2d_to_3d(ResNet50_3D_official(in_channels=3, num_classes=2)); bb.linear = nn.Identity()
    else:   # P4: task-specific BN (2D BNs for 2D tasks, 3D BNs for 3D tasks), then dual 2D/ACS convs
        bb = torchvision.models.resnet50(weights=None); bb.fc = nn.Identity()
    bb = make_task_bn(bb, task_is3d, ctx)
    if mode == "p4":
        bb = make_dual(bb)
    return ArkNet(bb, 2048, n_classes_list, ctx=ctx)


# ------------------------------------------------------------------------------------- tasks / data
class Task:
    def __init__(self, flag, head, npz, device, mode, data_root, bs2d=128, bs3d=32):
        info = INFO[flag]
        self.flag, self.head = flag, head
        self.is3d = flag.endswith("3d")
        self.multilabel = info["task"] == "multi-label, binary-class"
        self.n_classes = len(info["label"])
        Split = GPUSplit3D if self.is3d else GPUSplit
        self.train, self.val, self.test = (Split(npz, s, device) for s in ("train", "val", "test"))
        self.bs = bs3d if self.is3d else bs2d
        self.shape = flag in SHAPE_DATASETS
        self.upsample3d = 2 if (mode == "p4" and self.is3d) else 1
        self.val_eval = Evaluator(flag, "val", size=28, root=data_root)
        self.test_eval = Evaluator(flag, "test", size=28, root=data_root)

    def batch(self, split, idx, train):
        if self.is3d:
            x, y = split.batch(idx, ("random" if train else "0.5") if self.shape else None)
            if self.upsample3d > 1:
                x = F.interpolate(x, scale_factor=self.upsample3d, mode="nearest")
        else:
            x, y = split.batch(idx)
        return x, y

    def loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y.float()) if self.multilabel \
            else F.cross_entropy(logits, y.squeeze(1))

    @torch.no_grad()
    def predict(self, model, split, amp):
        model.eval(); out = []
        bs = 2 * self.bs if self.is3d else 512
        for s in range(0, split.n, bs):
            idx = torch.arange(s, min(s + bs, split.n), device=split.labels.device)
            x, _ = self.batch(split, idx, train=False)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                _, logits = model(x, self.head)
            logits = logits.float()
            out.append(torch.sigmoid(logits) if self.multilabel else torch.softmax(logits, 1))
        return torch.cat(out).cpu().numpy()


def cosine_scheduler(base, final, epochs, niter):          # Ark utils.cosine_scheduler (no warm-up)
    it = np.arange(epochs * niter)
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / len(it)))


@torch.no_grad()
def recalibrate_bn(model, tasks, amp, n_batches=20):
    """BN recalibration: recompute every BatchNorm's running statistics for the CURRENT weights.
    Each dataset's BN is updated only when that dataset is trained (~1 step in T) while the shared
    convolutions change at every step, so its running statistics go stale; evaluation then uses
    statistics of an older network and predictions collapse (observed: student val AUC exactly 0.500
    while the slowly-moving EMA teacher was fine). No weights are changed."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None             # None = exact cumulative average
    model.train()
    for t in tasks:
        perm = torch.randperm(t.train.n, device=t.train.labels.device)
        for b in range(min(n_batches, math.ceil(t.train.n / t.bs))):
            x, _ = t.batch(t.train, perm[b * t.bs:(b + 1) * t.bs], train=False)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                model(x, t.head)
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    model.eval()


def evaluate_all(model, tasks, amp):
    res = {}
    for t in tasks:
        auc, acc = safe_eval(t.val_eval, t.predict(model, t.val, amp))
        res[t.flag] = (auc, acc)
    return res


def safe_eval(evaluator, pred, *args):
    """medmnist's Evaluator crashes on NaN scores; report NaN AUC instead (that round is then never selected)."""
    if not np.isfinite(pred).all():
        print(f"   [warn] non-finite predictions for {evaluator.flag} {evaluator.split} - AUC set to NaN", flush=True)
        return float("nan"), float("nan")
    return evaluator.evaluate(pred, *args)



# ------------------------------------------------------------------------------------------- main
def run_joint(mode, out_dir, data_root, rounds=100, max_iters_per_task=300, lr=1e-3, seed=0, schedule="interleaved",
              bn_recal=True,
              amp=True, max_minutes=None, keep_best_weights=True, bs2d=128, bs3d=32):
    t_start = time.time()
    if not torch.cuda.is_available() and not os.environ.get("MEDMNIST_ALLOW_CPU"):
        raise RuntimeError("No GPU visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    tag = f"joint_{mode}"
    for sub in ("ckpt", "predictions", "weights", "history"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    if (tag, seed) in done_runs(results_path):
        print(f"[skip] {tag} already finished"); return None
    ckpt_path = os.path.join(out_dir, "ckpt", f"{tag}.pt")

    flags = MODES[mode]
    tasks = []
    for h, f in enumerate(flags):
        if not os.path.exists(os.path.join(data_root, f"{f}.npz")):
            getattr(medmnist, INFO[f]["python_class"])(split="train", download=True, root=data_root, size=28)
        tasks.append(Task(f, h, np.load(os.path.join(data_root, f"{f}.npz")), device, mode, data_root, bs2d, bs3d))
    T = len(tasks)

    student = build_model(mode, [t.n_classes for t in tasks], seed, [t.is3d for t in tasks]).to(device)
    if mode == "p3_2d":
        student = student.to(memory_format=torch.channels_last)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5 * rounds), int(0.75 * rounds)], 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16)
    if amp:
        print(f"[{tag}] mixed precision: {str(AMP_DTYPE).replace('torch.', '')}", flush=True)
    coef = cosine_scheduler(0.0, 0.5, rounds, T)          # consistency weight 0 -> 0.5   (Ark)
    mom = cosine_scheduler(0.9, 1.0, rounds, T)           # teacher EMA momentum 0.9 -> 1 (Ark)

    start, best, history = 0, {"mean_val_auc": -1.0}, []
    best_state = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if set(ck["student"]) != set(student.state_dict()) or ck.get("schedule", "cyclic") != schedule:
            os.replace(ckpt_path, ckpt_path + ".old_version")             # from an older model/schedule version
            print(f"[{tag}] old checkpoint has a different architecture or schedule - moved aside, starting fresh")
            ck = None
    else:
        ck = None
    if ck is not None:
        student.load_state_dict(ck["student"]); teacher.load_state_dict(ck["teacher"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"]); start, best, history = ck["round"] + 1, ck["best"], ck["history"]
        best_state = ck["best_state"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if ck.get("rng_cuda") is not None: torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"[resume] {tag} from round {start + 1}/{rounds} (best mean val AUC {best['mean_val_auc']:.4f})")
    else:
        sizes = ", ".join(f"{t.flag}:{t.train.n}" for t in tasks)
        print(f"[start] {tag}: {T} tasks, {rounds} rounds, cap {max_iters_per_task} it/task, {schedule} | {sizes}")

    for r in range(start, rounds):
        t0 = time.time(); rec = {"round": r + 1, "lr": optimizer.param_groups[0]["lr"], "loss": {}}
        n_its = [math.ceil(t.train.n / t.bs) for t in tasks]
        if max_iters_per_task:
            n_its = [min(n, max_iters_per_task) for n in n_its]
        order = [ti for ti in range(T) for _ in range(n_its[ti])]   # cyclic: task after task (Ark)
        if schedule == "cyclic":
            ends = list(np.cumsum(n_its))                             # EMA after each task epoch (Ark)
        else:                                                         # interleaved: a random task each step
            order = [order[i] for i in torch.randperm(len(order)).tolist()]
            ends = [round(len(order) * (k + 1) / T) for k in range(T)]   # same number of EMA updates
        perms = [torch.randperm(t.train.n, device=device) for t in tasks]
        ptr = [0] * T
        lc, lk = [[] for _ in range(T)], [[] for _ in range(T)]
        student.train(); teacher.train()              # as in Ark: teacher forward in train mode
        k = 0                                         # index of the current EMA segment in this round
        for step, ti in enumerate(order):
            task = tasks[ti]
            c = float(coef[r * T + k])
            b = ptr[ti]; ptr[ti] += 1
            idx = perms[ti][b * task.bs:(b + 1) * task.bs]
            x, y = task.batch(task.train, idx, train=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                zs, logits = student(x, task.head)
                with torch.no_grad():
                    zt, _ = teacher(x, task.head)
            loss_cls = task.loss(logits.float(), y)
            loss_con = F.mse_loss(zs.float(), zt.float())
            loss = (1 - c) * loss_cls + c * loss_con
            if torch.isfinite(loss):                  # never step on a non-finite loss
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                lc[ti].append(loss_cls.detach()); lk[ti].append(loss_con.detach())
            else:
                rec.setdefault("skipped", {}).setdefault(task.flag, 0)
                rec["skipped"][task.flag] += 1
            while k < T and step + 1 == ends[k]:      # end of a segment: teacher <- EMA(student)
                m = float(mom[r * T + k])
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.mul_(m).add_((1 - m) * ps.detach())
                k += 1
        for ti, task in enumerate(tasks):
            if lc[ti]:
                rec["loss"][task.flag] = [torch.stack(lc[ti]).mean().item(), torch.stack(lk[ti]).mean().item()]
        scheduler.step()

        for who, model in (("student", student), ("teacher", teacher)):
            if bn_recal:
                recalibrate_bn(model, tasks, amp)
            res = evaluate_all(model, tasks, amp)
            mean_auc = float(np.mean([a for a, _ in res.values()]))
            rec[f"{who}_val"] = {k: v[0] for k, v in res.items()}
            rec[f"{who}_mean_val_auc"] = mean_auc
            if not math.isnan(mean_auc) and mean_auc > best["mean_val_auc"]:
                best = {"mean_val_auc": mean_auc, "round": r + 1, "model": who}
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        rec["sec"] = time.time() - t0
        history.append(rec)
        print(f"{tag} round {r+1:3d}/{rounds} | mean val AUC student {rec['student_mean_val_auc']:.4f} "
              f"teacher {rec['teacher_mean_val_auc']:.4f} | best {best['mean_val_auc']:.4f} "
              f"(r{best.get('round')}, {best.get('model')}) | worst: {min(rec['student_val'], key=lambda k: rec['student_val'][k] if rec['student_val'][k] == rec['student_val'][k] else -1)} "
              f"{min(v for v in rec['student_val'].values() if v == v) if any(v == v for v in rec['student_val'].values()) else float('nan'):.3f} "
              f"| {rec['sec']/60:.1f} min", flush=True)
        atomic_save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "round": r, "best": best, "best_state": best_state, "schedule": schedule,
                     "history": history, "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ckpt_path)
        if max_minutes is not None and r + 1 < rounds:
            if (time.time() - t_start) / 60 + 1.5 * rec["sec"] / 60 > max_minutes:
                print(f"[pause] {tag} after round {r+1}. Re-run to resume."); return None

    # -------- final: the single best model (by mean val AUC) on every test set
    if best_state is None:
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in student.state_dict().items()}
    student.load_state_dict(best_state)
    pred_dir = os.path.join(out_dir, "predictions", tag); os.makedirs(pred_dir, exist_ok=True)
    rows = []
    for t in tasks:
        v_auc, v_acc = safe_eval(t.val_eval, t.predict(student, t.val, amp), pred_dir, tag)
        te_auc, te_acc = safe_eval(t.test_eval, t.predict(student, t.test, amp), pred_dir, tag)
        rep = {**REPORTED_2D, **REPORTED_3D}[t.flag]
        rows.append({"mode": mode, "data_flag": t.flag, "val_auc": v_auc, "val_acc": v_acc,
                     "test_auc": te_auc, "test_acc": te_acc, "reported_auc": rep[0], "reported_acc": rep[1]})
        print(f"   {t.flag:16s} test AUC {te_auc:.4f}  ACC {te_acc:.4f}   (paper ResNet-50: {rep[0]:.3f})")
    with open(os.path.join(out_dir, "joint_results.jsonl"), "a") as f:
        for row in rows:
            f.write(json.dumps({**row, "best_round": best["round"], "best_model": best["model"],
                                "rounds": rounds, "max_iters_per_task": max_iters_per_task,
                                "schedule": schedule}) + "\n")
    with open(os.path.join(out_dir, "history", f"{tag}.json"), "w") as f:
        json.dump(history, f)
    if keep_best_weights:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                   os.path.join(out_dir, "weights", f"{tag}_best_fp16.pth"))
    summary = {"data_flag": tag, "seed": seed, "model": f"ark_{mode}", "best_round": best["round"],
               "best_model": best["model"], "mean_val_auc": best["mean_val_auc"],
               "mean_test_auc": float(np.mean([x["test_auc"] for x in rows])),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(results_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    os.remove(ckpt_path)
    print(f"[done] {tag}: best round {best['round']} ({best['model']}), mean test AUC {summary['mean_test_auc']:.4f}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--out_dir", default="out")
    p.add_argument("--data_root", default="data")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--max_iters_per_task", type=int, default=300, help="0 = full epoch per task (pure Ark)")
    p.add_argument("--max_minutes", type=float, default=None)
    p.add_argument("--schedule", default="interleaved", choices=["interleaved", "cyclic"])
    a = p.parse_args()
    run_joint(a.mode, a.out_dir, a.data_root, a.rounds, a.max_iters_per_task, schedule=a.schedule,
              max_minutes=a.max_minutes)
