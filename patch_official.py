"""
patch_official.py - apply a SMALL, documented patch to the authors' code (github.com/MedMNIST/experiments,
pinned commit 70b6b3a7) so that 90 runs fit on rented GPUs. Usage:  python patch_official.py official_repo

Every option defaults to the ORIGINAL behaviour; we only switch them on from the command line.
  1. --amp                 mixed precision (fp16 autocast + GradScaler) for train and test forward passes
  2. --eval_every_epoch    'all' (original: train+val+test every epoch) or 'val' (val only; model selection
                           uses val AUC only, so the selected model is the same; final train/val/test
                           evaluation of the best model is unchanged)
  3. --num_workers         DataLoader workers (original: 0) - PIL resizing on the CPU was the bottleneck
  4. --seed                fixed random seed (the original sets none)
  5. per-epoch JSON line   "EPOCH_LOG {...}" printed for learning curves (no effect on training);
                           metrics that were not computed (NaN) are not written to TensorBoard
  6. CUDA_VISIBLE_DEVICES  the original overwrites it with --gpu_ids, which sends every parallel job to
                           GPU 0; the patch keeps the value set by the caller (one job per GPU)
Also fixes the installed ACSConv package (one line): its converter passes BatchNorm's `bias` *tensor*
as a constructor flag on recent PyTorch and crashes. With the fix it works unchanged.
The full diff is written to official_patch.diff for the report.
"""
import os, subprocess, sys

COMMIT = "70b6b3a7ad7afddff1df2a3b735235830fbdb142"

HEADER = '''
# ---- [warm-up patch] options; defaults reproduce the original behaviour exactly ----
import json as _json
import random as _random
PATCH = {"amp": False, "eval_every_epoch": "all", "num_workers": 0, "seed": None}
_scaler = None
# ---------------------------------------------------------------------------------------
'''

SEED = '''
    if PATCH["seed"] is not None:  # [warm-up patch] fixed seed
        _random.seed(PATCH["seed"]); np.random.seed(PATCH["seed"])
        torch.manual_seed(PATCH["seed"]); torch.cuda.manual_seed_all(PATCH["seed"])
'''

ARGS = '''    parser.add_argument('--amp', action="store_true", help='[warm-up patch] mixed precision')
    parser.add_argument('--eval_every_epoch', default='all', choices=['all', 'val'], help='[warm-up patch]')
    parser.add_argument('--num_workers', default=0, type=int, help='[warm-up patch] DataLoader workers')
    parser.add_argument('--seed', default=None, type=int, help='[warm-up patch] random seed')

    args = parser.parse_args()
    PATCH.update(amp=args.amp, eval_every_epoch=args.eval_every_epoch, num_workers=args.num_workers, seed=args.seed)'''

AUTOCAST_TRAIN = '''        with torch.autocast("cuda", dtype=torch.float16, enabled=PATCH["amp"]):  # [warm-up patch]
            outputs = model(inputs.to(device))
        outputs = outputs.float()'''
AUTOCAST_TEST = '''            with torch.autocast("cuda", dtype=torch.float16, enabled=PATCH["amp"]):  # [warm-up patch]
                outputs = model(inputs.to(device))
            outputs = outputs.float()'''
BACKWARD = '''        _scaler.scale(loss).backward()  # [warm-up patch] GradScaler (no-op without --amp)
        _scaler.step(optimizer)
        _scaler.update()'''


def epoch_eval(args_train, args_val, args_test):
    return (f'''        if PATCH["eval_every_epoch"] == "all":  # [warm-up patch] original: train+val+test every epoch
            train_metrics = test({args_train})
            val_metrics = test({args_val})
            test_metrics = test({args_test})
        else:                                   # val only (model selection uses val AUC only)
            val_metrics = test({args_val})
            train_metrics = test_metrics = [float("nan")] * 3''')


def edits(is3d):
    if is3d:
        a_tr = "model, train_evaluator, train_loader_at_eval, criterion, device, run"
        a_va = "model, val_evaluator, val_loader, criterion, device, run"
        a_te = "model, test_evaluator, test_loader, criterion, device, run"
        train_sig = "def train(model, train_loader, criterion, optimizer, device, writer):"
    else:
        a_tr = "model, train_evaluator, train_loader_at_eval, task, criterion, device, run"
        a_va = "model, val_evaluator, val_loader, task, criterion, device, run"
        a_te = "model, test_evaluator, test_loader, task, criterion, device, run"
        train_sig = "def train(model, train_loader, task, criterion, optimizer, device, writer):"
    orig_eval = (f"        train_metrics = test({a_tr})\n        val_metrics = test({a_va})\n"
                 f"        test_metrics = test({a_te})")
    return [  # (old, new, expected count)
        ("from tqdm import trange\n", "from tqdm import trange\n" + HEADER, 1),
        ('        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])',
         '        pass  # [warm-up patch] keep the caller\'s CUDA_VISIBLE_DEVICES (one job per GPU)', 1),
        ("    lr = 0.001\n", "    lr = 0.001\n" + SEED, 1),
        ("shuffle=True)", 'shuffle=True, num_workers=PATCH["num_workers"])', 1),
        ("shuffle=False)", 'shuffle=False, num_workers=PATCH["num_workers"])', 3),
        ("    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
         "    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n"
         "    global _scaler\n    _scaler = torch.amp.GradScaler(\"cuda\", enabled=PATCH[\"amp\"])  # [warm-up patch]\n", 1),
        (orig_eval, epoch_eval(a_tr, a_va, a_te), 1),
        ("        scheduler.step()\n",
         "        scheduler.step()\n"
         "        print('EPOCH_LOG ' + _json.dumps({'epoch': epoch, 'train_loss': train_loss,  # [warm-up patch]\n"
         "              'val_auc': val_metrics[1], 'val_acc': val_metrics[2], 'test_auc': test_metrics[1],\n"
         "              'lr': optimizer.param_groups[0]['lr']}), flush=True)\n", 1),
        (train_sig + "\n", train_sig + "\n", 1),
        ("        optimizer.zero_grad()\n        outputs = model(inputs.to(device))\n",
         "        optimizer.zero_grad()\n" + AUTOCAST_TRAIN + "\n", 1),
        ("enumerate(data_loader):\n            outputs = model(inputs.to(device))\n",
         "enumerate(data_loader):\n" + AUTOCAST_TEST + "\n", 1),
        ("        loss.backward()\n        optimizer.step()", BACKWARD, 1),
        ("\n    args = parser.parse_args()", "\n" + ARGS, 1),
        ("            writer.add_scalar(key, value, epoch)\n",
         "            if value == value:  # [warm-up patch] skip NaN (metric not computed this epoch)\n"
         "                writer.add_scalar(key, value, epoch)\n", 1),
    ]


def patch_file(path, is3d):
    raw = open(path, newline="").read()
    crlf = "\r\n" in raw                     # the authors' files use Windows line endings: keep them
    s = raw.replace("\r\n", "\n")
    if "[warm-up patch]" in s:
        print("already patched:", path); return
    for old, new, n in edits(is3d):
        c = s.count(old)
        if c != n:
            sys.exit(f"PATCH FAILED in {path}: expected {n} x {old[:60]!r}, found {c}. Is the repo at {COMMIT}?")
        s = s.replace(old, new)
    open(path, "w", newline="").write(s.replace("\n", "\r\n") if crlf else s)
    print("patched:", path)


def fix_acsconv():
    import acsconv
    p = os.path.join(os.path.dirname(acsconv.__file__), "converters", "base_converter.py")
    s = open(p).read()
    old = "arguments = [a for a in arguments if a not in ['device', 'dtype', 'factory_kwargs']]"
    new = "arguments = [a for a in arguments if a not in ['device', 'dtype', 'factory_kwargs', 'bias']]"
    if new in s:
        print("acsconv already fixed")
    elif s.count(old) == 1:
        open(p, "w").write(s.replace(old, new)); print("fixed acsconv:", p)
    else:
        sys.exit("could not find the acsconv line to fix: " + p)


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "official_repo"
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        sys.exit(f"{repo} is at {head}, expected {COMMIT}. Run setup_official.sh")
    patch_file(os.path.join(repo, "MedMNIST2D", "train_and_eval_pytorch.py"), is3d=False)
    patch_file(os.path.join(repo, "MedMNIST3D", "train_and_eval_pytorch.py"), is3d=True)
    fix_acsconv()
    diff = subprocess.run(["git", "-C", repo, "diff"], capture_output=True, text=True).stdout
    open("official_patch.diff", "w").write(diff)
    print(f"wrote official_patch.diff ({diff.count(chr(10))} lines) - attach/quote it in the report")
