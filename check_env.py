"""
check_env.py - run this first on the server.
  python check_env.py --data_root data
Checks: GPU(s), library versions, internet access to the MedMNIST download, the pre-processing
equivalence test, and training speed -> estimated hours per dataset.
"""
import argparse, os, sys, subprocess
import numpy as np, torch, torchvision, medmnist
import torch.nn.functional as F
from medmnist import INFO
import medmnist_replicate as mr

p = argparse.ArgumentParser()
from medmnist.info import DEFAULT_ROOT
p.add_argument("--data_root", default=DEFAULT_ROOT)
p.add_argument("--skip_download", action="store_true")
p.add_argument("--3d", dest="bench3d", action="store_true", help="also benchmark the 3D ResNet-50")
a = p.parse_args()

print("=" * 70)
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} "
      f"| medmnist {medmnist.__version__}")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible. Check nvidia-smi / CUDA_VISIBLE_DEVICES / that torch was installed with CUDA.")
for i in range(torch.cuda.device_count()):
    pr = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: {pr.name} | {total/2**30:.1f} GB total, {free/2**30:.1f} GB free right now")
try:
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                         capture_output=True, text=True).stdout.strip() or "(no other processes on GPUs)")
except FileNotFoundError:
    pass

print("=" * 70, "\n1) pre-processing equivalence vs the official PIL pipeline")
import PIL, torchvision.transforms as T
from PIL import Image
off = T.Compose([T.Resize((224, 224), interpolation=PIL.Image.NEAREST), T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
for shape in [(28, 28), (28, 28, 3)]:
    x = np.random.randint(0, 256, size=(8,) + shape, dtype=np.uint8)
    ref = torch.stack([off(Image.fromarray(i).convert("RGB")) for i in x]).cuda()
    ours, _ = mr.GPUSplit({"s_images": x, "s_labels": np.zeros((8, 1))}, "s", "cuda").batch(torch.arange(8, device="cuda"))
    d = (ref - ours.float()).abs().max().item()
    # <= 1e-6: float32 rounding only (GPU division vs CPU division differ in the last bit)
    print(f"   input {shape}: max |official - ours| = {d:.2e}  ->  {'OK' if d <= 1e-6 else 'MISMATCH!'}")

if not a.skip_download:
    print("=" * 70, "\n2) downloading ChestMNIST (tests internet access to zenodo.org)")
    os.makedirs(a.data_root, exist_ok=True)
    try:
        medmnist.ChestMNIST(split="train", download=True, root=a.data_root, size=28)
        print("   OK:", os.path.join(a.data_root, "chestmnist.npz"))
    except Exception as e:
        print("   FAILED - no internet from the server? Download the .npz files on your laptop from\n"
              "   https://zenodo.org/records/10519652 and copy them into", a.data_root, "\n  ", repr(e)[:200])

print("=" * 70, "\n3) training speed, ResNet-50 @224, batch 128, AMP (GPU 0)")
ips, mem = mr.benchmark(a.data_root)
print(f"   {ips:.0f} train images/s | peak memory {mem:.1f} GB")
print(f"\n   {'dataset':16s} {'h/run':>7s} {'h/5 runs':>9s}")
tot = 0
for f in mr.REPORTED_2D:
    n = INFO[f]["n_samples"]
    h = 100 * (n["train"] / ips + n["val"] / (3 * ips)) / 3600
    tot += 5 * h
    print(f"   {f:16s} {h:7.1f} {5*h:9.1f}")
print(f"   {'TOTAL 2D':16s} {'':7s} {tot:9.1f}  hours on one GPU")
if a.bench3d:
    print("=" * 70, "\n4) training speed, ResNet-50 + 3D, 28^3, batch 32, AMP (GPU 0)")
    from medmnist3d_replicate import benchmark3d
    ips3, mem3 = benchmark3d()
    print(f"   {ips3:.1f} train volumes/s | peak memory {mem3:.1f} GB")
    tot3 = 0
    for f in mr.REPORTED_3D:
        n = INFO[f]["n_samples"]
        h = 100 * (n["train"] + n["val"] / 3) / ips3 / 3600
        tot3 += 5 * h
        print(f"   {f:16s} {h:7.1f} {5*h:9.1f}")
    print(f"   {'TOTAL 3D':16s} {'':7s} {tot3:9.1f}  hours")
# ---- whole project on this machine: all GPUs in parallel (launch_workers.sh)
import run_queue as rq
ngpu = torch.cuda.device_count()
ips3 = ips3 if a.bench3d else None
jobs = [(f, s) for f in rq.PLANS["all"] for s in ([0] if f.startswith("joint_") else range(5))]
hours = [rq.est_hours(f, ips2d=ips, ips3d=ips3) or 1.0 for f, _ in jobs]
total, longest = sum(hours), max(hours)
wall = max(total / ngpu, longest) * 1.1          # +10% for imbalance at the end
print("=" * 70)
print(f"WHOLE PROJECT (Steps a+b: 90 runs, Step c: 3 joint models) on {ngpu} x {torch.cuda.get_device_name(0)}:")
print(f"   total GPU-hours ~{total:.0f} | longest single job ~{longest:.1f} h | expected wall-clock ~{wall:.1f} h")
print(f"   cost estimate: wall-clock hours x your pod's $/hour")
print(f"   CPU cores usable: {rq.usable_cpus()} (~{rq.usable_cpus() // max(1, ngpu)} per GPU for the official scripts' image loading;"
      f" below ~4 per GPU the estimate is optimistic)")
if not a.bench3d:
    print("   (3D jobs counted as 1 h each - add --3d for a real estimate)")
print("=" * 70, "\nPaste this whole output back to Claude.")
