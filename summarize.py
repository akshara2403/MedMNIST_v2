"""
summarize.py - collect everything, compare with the paper, draw curves, fill the spreadsheet, write LaTeX tables.
  python summarize.py --out_dir out --xlsx performance_table_Akshara.xlsx
Writes into out/:
  summary_runs.csv          every individual run (Steps a, b)
  summary_by_dataset.csv    per dataset: mean, std, gap to paper, P3, P4
  results_table.tex         booktabs table for the Overleaf report
  curves_<dataset>.png      learning curves of the 5 runs
  curves_joint_<mode>.png   mean validation AUC per round for the joint models (Step c)
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist_replicate import REPORTED_2D, REPORTED_3D

REPORTED = {**REPORTED_2D, **REPORTED_3D}
ORDER = list(REPORTED)
NICE = {"chestmnist": "ChestMNIST", "pathmnist": "PathMNIST", "dermamnist": "DermaMNIST", "octmnist": "OCTMNIST",
        "pneumoniamnist": "PneumoniaMNIST", "retinamnist": "RetinaMNIST", "breastmnist": "BreastMNIST",
        "bloodmnist": "BloodMNIST", "tissuemnist": "TissueMNIST", "organamnist": "OrganAMNIST",
        "organcmnist": "OrganCMNIST", "organsmnist": "OrganSMNIST", "organmnist3d": "OrganMNIST3D",
        "nodulemnist3d": "NoduleMNIST3D", "fracturemnist3d": "FractureMNIST3D", "adrenalmnist3d": "AdrenalMNIST3D",
        "vesselmnist3d": "VesselMNIST3D", "synapsemnist3d": "SynapseMNIST3D"}

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default="out")
p.add_argument("--xlsx", default=None, help="spreadsheet to fill (Run 1-5, P3, P4 rows); edited in place")
a = p.parse_args()
O = a.out_dir
pd.set_option("display.width", 220)

# ------------------------------------------------------------------ Steps (a)+(b): individual runs
rows = [json.loads(l) for l in open(os.path.join(O, "results.jsonl")) if l.strip()] \
    if os.path.exists(os.path.join(O, "results.jsonl")) else []
runs = pd.DataFrame([r for r in rows if not r["data_flag"].startswith("joint_")])
summ = pd.DataFrame(columns=["dataset"])
if len(runs):
    runs = runs.sort_values(["data_flag", "seed"]).drop_duplicates(["data_flag", "seed"], keep="last")
    runs.to_csv(os.path.join(O, "summary_runs.csv"), index=False)
    print("\nINDIVIDUAL RUNS (test metrics of the best-validation-AUC epoch)")
    print(runs[["data_flag", "seed", "best_epoch", "val_auc", "test_auc", "test_acc"]].round(4).to_string(index=False))
    out = []
    for flag in [f for f in ORDER if f in set(runs.data_flag)]:
        g = runs[runs.data_flag == flag]
        r = {"dataset": flag, "n_runs": len(g)}
        for m, rep in (("auc", REPORTED[flag][0]), ("acc", REPORTED[flag][1])):
            mu, sd = g[f"test_{m}"].mean(), g[f"test_{m}"].std(ddof=1)
            r |= {f"{m}_mean": mu, f"{m}_std": sd, f"{m}_reported": rep, f"{m}_diff": mu - rep}
        r["auc_within_1%"] = "yes" if r["auc_diff"] > -0.01 else "NO"
        out.append(r)
    summ = pd.DataFrame(out)
    print("\nSTEPS (a)+(b) PER DATASET  (std = sample std, n-1;  diff = mean - paper)")
    print(summ[["dataset", "n_runs", "auc_mean", "auc_std", "auc_reported", "auc_diff", "auc_within_1%",
                "acc_mean", "acc_reported", "acc_diff"]].round(4).to_string(index=False))
    for flag in summ.dataset:
        g = runs[runs.data_flag == flag]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for seed in g.seed:
            hp = os.path.join(O, "history", f"{flag}_seed{seed}.json")
            if not os.path.exists(hp):
                continue
            h = json.load(open(hp))
            ax[0].plot([e["val_auc"] for e in h], label=f"seed {seed} val")
            if "test_auc" in h[0]:
                ax[0].plot([e["test_auc"] for e in h], "--", alpha=.6, label=f"seed {seed} test")
            ax[1].plot([e["train_loss"] for e in h], label=f"seed {seed}")
        ax[0].axhline(REPORTED[flag][0], color="k", ls=":", label="paper test AUC")
        for x, t in zip(ax, ["AUC", "training loss"]):
            x.set_xlabel("epoch"); x.set_title(f"{NICE[flag]}: {t}"); x.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(O, f"curves_{flag}.png"), dpi=200); plt.close()

# ------------------------------------------------------------------ Step (c): joint models P3 / P4
P3, P4 = {}, {}
jp = os.path.join(O, "joint_results.jsonl")
if os.path.exists(jp):
    j = pd.DataFrame([json.loads(l) for l in open(jp) if l.strip()]).drop_duplicates(["mode", "data_flag"], keep="last")
    for _, r in j.iterrows():
        (P4 if r["mode"] == "p4" else P3)[r["data_flag"]] = r["test_auc"]
    for mode in j["mode"].unique():
        hp = os.path.join(O, "history", f"joint_{mode}.json")
        if os.path.exists(hp):
            h = json.load(open(hp))
            plt.figure(figsize=(6, 4))
            plt.plot([e["round"] for e in h], [e["student_mean_val_auc"] for e in h], label="student")
            plt.plot([e["round"] for e in h], [e["teacher_mean_val_auc"] for e in h], label="teacher (EMA)")
            plt.xlabel("round"); plt.ylabel("mean validation AUC"); plt.title(f"joint {mode}"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(O, f"curves_joint_{mode}.png"), dpi=200); plt.close()

table = []
for flag in ORDER:
    s = summ[summ.dataset == flag] if len(summ) else []
    p2 = float(s.auc_mean.iloc[0]) if len(s) else np.nan
    sd = float(s.auc_std.iloc[0]) if len(s) else np.nan
    table.append({"dataset": flag, "P1": REPORTED[flag][0], "P2": p2, "P2_std": sd,
                  "P3": P3.get(flag, np.nan), "P4": P4.get(flag, np.nan)})
table = pd.DataFrame(table)
for x, y in (("P2", "P1"), ("P3", "P1"), ("P3", "P2"), ("P4", "P1"), ("P4", "P2"), ("P4", "P3")):
    table[f"{x}-{y}"] = table[x] - table[y]
table.to_csv(os.path.join(O, "summary_by_dataset.csv"), index=False)
print("\nALL STEPS (test AUC):  P1 paper | P2 mean of 5 runs | P3 2D-only or 3D-only joint | P4 2D+3D joint")
print(table.round(4).to_string(index=False))
for k in ("P2", "P3", "P4"):
    for part, fl in (("2D", list(REPORTED_2D)), ("3D", list(REPORTED_3D))):
        v = table[table.dataset.isin(fl)][k]
        if v.notna().all():
            print(f"   mean {k} over {part}: {v.mean():.4f}   (paper P1: {table[table.dataset.isin(fl)].P1.mean():.4f})")

# ------------------------------------------------------------------ LaTeX table for the report
def f3(v, sd=None):
    if pd.isna(v): return "--"
    return f"{v:.3f}" + (f" $\\pm$ {sd:.3f}" if sd is not None and not pd.isna(sd) else "")
def g3(v):
    return "--" if pd.isna(v) else f"{v:+.3f}"
lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Test AUC. P1: reported by Yang et al.\ (2023), ResNet-50 (224) for 2D and ResNet-50+3D for 3D. "
         r"P2: mean $\pm$ std of our 5 runs. P3: one model per dimensionality (12 2D or 6 3D datasets). "
         r"P4: one model for all 18 datasets.}", r"\label{tab:results}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Dataset & P1 & P2 & P2$-$P1 & P3 & P4 & P4$-$P2 \\", r"\midrule"]
for i, r in table.iterrows():
    if r.dataset == "organmnist3d":
        lines.append(r"\midrule")
    lines.append(f"{NICE[r.dataset]} & {r.P1:.3f} & {f3(r.P2, r.P2_std)} & {g3(r['P2-P1'])} & {f3(r.P3)} & "
                 f"{f3(r.P4)} & {g3(r['P4-P2'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(O, "results_table.tex"), "w").write("\n".join(lines) + "\n")
print(f"\nSaved CSVs, results_table.tex and curves_*.png in {O}")

# ------------------------------------------------------------------ spreadsheet
if a.xlsx:
    import openpyxl
    wb = openpyxl.load_workbook(a.xlsx); ws = wb.active
    col = {str(ws.cell(1, c).value).lower(): c for c in range(4, 22)}
    if len(runs):
        for flag, g in runs.groupby("data_flag"):
            for i, auc in enumerate(g.sort_values("seed").test_auc.tolist()[:5]):
                ws.cell(3 + i, col[flag], round(float(auc), 4))
    for flag, v in P3.items():
        ws.cell(10, col[flag], round(float(v), 4))
    for flag, v in P4.items():
        ws.cell(11, col[flag], round(float(v), 4))
    wb.save(a.xlsx)
    print("Filled Run 1-5, P3 and P4 in", a.xlsx, "(P2, std and all gains are formulas; open in Excel)")
