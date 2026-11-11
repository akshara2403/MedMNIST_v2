"""
run_queue.py - run many jobs one after another; skips finished ones, resumes paused ones.
Jobs: (dataset, seed) for Steps (a)+(b), and joint_p3_2d / joint_p3_3d / joint_p4 for Step (c).

  python run_queue.py --plan all                       # everything, in the recommended order
  python run_queue.py --plan small2d,3d                # only some groups
  python run_queue.py --datasets tissuemnist --seeds 0,1   # explicit jobs
  python run_queue.py --plan all --list                # just print the jobs and estimated hours

Splitting across machines (e.g. Datycs T4 + Kaggle): give each machine DIFFERENT jobs, e.g.
  Datycs:  python run_queue.py --datasets tissuemnist,octmnist,pathmnist
  Kaggle:  python run_queue.py --plan small2d,3d --out_dir out_kaggle
and later merge the results.jsonl files:  cat out_kaggle/results.jsonl >> out/results.jsonl
"""
import argparse, os
from medmnist import INFO
from medmnist.info import DEFAULT_ROOT
from medmnist_replicate import run, done_runs, REPORTED_2D, REPORTED_3D

PLANS = {   # order = cheapest first, so the most datasets are finished early
    "chest":   ["chestmnist"],
    "small2d": ["breastmnist", "retinamnist", "pneumoniamnist", "dermamnist", "bloodmnist",
                "organcmnist", "organsmnist"],
    "3d":      ["organmnist3d", "nodulemnist3d", "fracturemnist3d", "adrenalmnist3d",
                "vesselmnist3d", "synapsemnist3d"],
    "mid2d":   ["organamnist"],
    "big2d":   ["pathmnist", "octmnist", "tissuemnist"],
    "joint":   ["joint_p3_2d", "joint_p3_3d", "joint_p4"],          # Step (c): one model per job
}
PLANS["all"] = PLANS["chest"] + PLANS["small2d"] + PLANS["3d"] + PLANS["mid2d"] + PLANS["big2d"] + PLANS["joint"]
def usable_cpus():
    """CPUs this container may really use (RunPod pods report the host's 200+ cores in os.cpu_count)."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]          # cgroup v2
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except Exception:
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())          # cgroup v1
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


IPS_2D, IPS_3D = 421, None   # measured on the Datycs T4 (2D); 3D: run check_env.py --3d


def est_hours(flag, ips2d=IPS_2D, ips3d=None, rounds=100, cap=300):
    if flag.startswith("joint_"):           # per round: capped task epochs (+1/3 teacher) + val of 2 models
        mode = flag[len("joint_"):]
        fl2 = [] if mode == "p3_3d" else list(REPORTED_2D)
        img = sum(min(INFO[f]["n_samples"]["train"], cap * 128) * 4 / 3 + 2 * INFO[f]["n_samples"]["val"] / 3
                  for f in fl2)
        h3 = 0.0 if mode == "p3_2d" else (5.0 if not ips3d else
             rounds * sum(INFO[f]["n_samples"]["train"] * 4 / 3 for f in REPORTED_3D) / ips3d / 3600)
        return rounds * img / ips2d / 3600 + h3
    n = INFO[flag]["n_samples"]
    if flag.endswith("3d"):
        return None if not ips3d else 100 * (n["train"] + n["val"] / 3) / ips3d / 3600
    return 100 * (n["train"] + n["val"] / 3) / ips2d / 3600


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default=None, help="comma list of: " + ", ".join(PLANS))
    p.add_argument("--datasets", default=None, help="comma list of data flags (overrides --plan)")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--data_root", default=DEFAULT_ROOT, help="shared by all engines (the official code reads ~/.medmnist)")
    p.add_argument("--engine", default="official", choices=["official", "fast"],
                   help="Steps (a)+(b): 'official' = the authors' scripts + small patch (default); "
                        "'fast' = our re-implementation (medmnist_replicate.py / medmnist3d_replicate.py)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers per job (official engine)")
    p.add_argument("--log_test", action="store_true", help="also log test AUC every epoch (curves)")
    p.add_argument("--max_minutes", type=float, default=None, help="stop cleanly before this (Kaggle: 690)")
    p.add_argument("--ips3d", type=float, default=None, help="3D samples/s from check_env.py, for estimates")
    p.add_argument("--list", action="store_true")
    p.add_argument("--worker", action="store_true",
                   help="multi-GPU mode: start one of these per GPU; each claims the next free job "
                        "(biggest jobs first). Use launch_workers.sh")
    p.add_argument("--download_only", action="store_true", help="download all datasets of the plan, then exit")
    p.add_argument("--joint_rounds", type=int, default=100, help="Step (c): number of Ark rounds")
    p.add_argument("--joint_cap", type=int, default=300, help="Step (c): max iterations per task per round")
    p.add_argument("--joint_schedule", default="interleaved", choices=["interleaved", "cyclic"])
    a = p.parse_args()

    if a.datasets:
        flags = a.datasets.split(",")
    else:
        flags = [f for g in (a.plan or "all").split(",") for f in PLANS[g]]
    seeds = [int(s) for s in a.seeds.split(",")]
    results_path = os.path.join(a.out_dir, "results.jsonl")

    if a.download_only:
        import medmnist
        os.makedirs(a.data_root, exist_ok=True)
        from medmnist_joint import MODES
        need = []
        for f in flags:
            for g in (MODES[f[len("joint_"):]] if f.startswith("joint_") else [f]):
                if g not in need:
                    need.append(g)
        for f in need:
            if not os.path.exists(os.path.join(a.data_root, f"{f}.npz")):
                print("downloading", f, flush=True)
                getattr(medmnist, INFO[f]["python_class"])(split="train", download=True, root=a.data_root, size=28)
        print("all datasets present in", a.data_root); raise SystemExit

    done = done_runs(results_path)
    jobs = [(f, s) for f in flags for s in (seeds if not f.startswith("joint_") else [0]) if (f, s) not in done]
    if a.worker:   # longest jobs first = shortest total wall-clock time across GPUs (3D treated as ~1 h)
        jobs.sort(key=lambda j: -(est_hours(j[0], ips3d=a.ips3d, rounds=a.joint_rounds, cap=a.joint_cap) or 1.0))
    tot = 0
    print(f"{len(jobs)} jobs to run ({len(done)} already finished in {a.out_dir}):")
    for f, s in jobs:
        h = est_hours(f, ips3d=a.ips3d, rounds=a.joint_rounds, cap=a.joint_cap)
        tot += h or 0
        print(f"   {f:16s} seed {s}   ~{h:.1f} h" if h else f"   {f:16s} seed {s}   (3D: estimate unknown)")
    print(f"   estimated total (2D on a T4, plus 3D if --ips3d given): {tot:.0f} h")
    if a.list:
        raise SystemExit

    import time, socket
    claim_dir = os.path.join(a.out_dir, "claims"); os.makedirs(claim_dir, exist_ok=True)
    me = f"{socket.gethostname()}:{os.getpid()}"

    def claim(f, s):
        """Atomically claim a job. Re-claims jobs whose worker on this machine has died."""
        path = os.path.join(claim_dir, f"{f}_seed{s}.claim")
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode()); os.close(fd); return True
            except FileExistsError:
                host, _, pid = open(path).read().partition(":")
                alive = True
                if host == socket.gethostname() and pid.isdigit():
                    try: os.kill(int(pid), 0)
                    except ProcessLookupError: alive = False
                    except PermissionError: pass
                if alive: return False
                os.remove(path)                       # stale claim from a crashed worker
        return False

    t0 = time.time()
    for f, s in jobs:
        if (f, s) in done_runs(results_path):
            continue
        if a.worker and not claim(f, s):
            continue
        left = None if a.max_minutes is None else a.max_minutes - (time.time() - t0) / 60
        if left is not None and left < 10:
            print("[queue] time budget used up - re-run to continue"); break
        print(f"[queue] {me} -> {f} seed {s}", flush=True)
        if a.workers is None:     # split the CPU cores between the GPUs (PIL resizing runs on the CPU)
            a.workers = max(2, min(8, usable_cpus() // int(os.environ.get("NGPU", "1")) - 1))
        try:
            if f.startswith("joint_"):
                from medmnist_joint import run_joint
                r = run_joint(f[len("joint_"):], a.out_dir, a.data_root, rounds=a.joint_rounds,
                              max_iters_per_task=a.joint_cap, schedule=a.joint_schedule, max_minutes=left)
            elif a.engine == "official":
                from run_official import run_official
                r = run_official(f, s, a.out_dir, workers=a.workers)
            elif f.endswith("3d"):
                from medmnist3d_replicate import run3d
                r = run3d(f, s, a.out_dir, a.data_root, log_test=a.log_test, max_minutes=left)
            else:
                r = run(f, s, a.out_dir, a.data_root, log_test=a.log_test, max_minutes=left)
        except Exception:                   # a crashing job must not take its worker (and GPU) down
            import traceback
            traceback.print_exc()
            print(f"[queue] CRASHED: {f} seed {s} - see the traceback above; moving on", flush=True)
            r = None
            import torch; torch.cuda.empty_cache()
        if a.worker:
            try: os.remove(os.path.join(claim_dir, f"{f}_seed{s}.claim"))
            except FileNotFoundError: pass
        if r is None and (f, s) not in done_runs(results_path):
            if left is not None:
                print("[queue] paused - re-run the same command to continue"); break
            print(f"[queue] {f} seed {s} did NOT finish - continuing with the next job (re-run later to retry)")
    print("[queue] nothing left for this worker. When all workers are done:  python summarize.py --out_dir", a.out_dir)
