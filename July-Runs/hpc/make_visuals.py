#!/usr/bin/env python3
"""
Regenerate the dtpo_testing_8-style visual artifacts for every HPC run locally.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HPC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HPC)
PY = sys.executable


def find_ckpt(rundir):
    cdir = os.path.join(rundir, "checkpoint")
    if not os.path.isdir(cdir):
        return None
    tars = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".tar")]
    finals = [t for t in tars if os.path.basename(t).startswith("final_")]
    pool = finals or tars
    return max(pool, key=os.path.getmtime) if pool else None


def winner_iter(rundir):
    log = os.path.join(rundir, "honest_eval.log")
    if not os.path.isfile(log):
        return None
    m = None
    with open(log) as f:
        for line in f:
            hit = re.search(r"WINNER iter=(\d+)", line)
            if hit:
                m = int(hit.group(1))
    return m


def export_tree(rundir, ckpt, name):
    import torch as th
    rec = th.load(ckpt, map_location="cpu", weights_only=False)
    tree = rec.get("tree")
    wit = winner_iter(rundir)
    cands = rec.get("candidates") or []
    if wit is not None and cands:
        for c in cands:
            if c.get("iter") == wit:
                tree = c["tree"]
                break
    if tree is None:
        print(f"  [tree] no tree in {os.path.basename(ckpt)} — skipped")
        return
    kept = rec.get("action_kept")
    src = ckpt
    if tree is not rec.get("tree") or "action_map" not in rec:
        tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
        th.save({"tree": tree, "action_map": kept}, tmp.name)
        src = tmp.name
    for extra, base in ((["--landscape"], "tree_landscape"), ([], "tree")):
        out = os.path.join(rundir, base)
        r = subprocess.run([PY, os.path.join(ROOT, "viz_dtpo_tree.py"),
                            "--ckpt", src, "--out", out] + extra,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [tree] {base} FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
        else:
            print(f"  [tree] {base}.svg" + (f" (honest winner iter={wit})" if wit is not None else ""))
    if src != ckpt:
        os.unlink(src)


def make_curve(rundir, name):
    log = os.path.join(rundir, "train.log")
    if not os.path.isfile(log):
        return
    with open(log) as f:
        if "[DTPO] it=" not in f.read():
            return                    # curves only exist for DTPO-family runs
    r = subprocess.run([PY, os.path.join(ROOT, "plot_dtpo_curve.py"),
                        "--log", log, "--out", os.path.join(rundir, "curve.png"),
                        "--title", name],
                       capture_output=True, text=True)
    print(f"  [curve] curve.png" if r.returncode == 0 else
          f"  [curve] FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")


def make_frontier(results_csv, out_png):
    import csv
    from collections import defaultdict
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = defaultdict(list)   # (env, method, leaves) -> [report..]
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not row.get("leaves_budget"):
                continue
            cells[(row["envtag"], row["method"], int(row["leaves_budget"]))].append(float(row["report"]))

    envs = sorted({k[0] for k in cells})
    if not envs:
        print("[frontier] no leaf-budget rows in results.csv yet")
        return
    colors = {"distill": "#898781", "viper": "#2a78d6", "dtpo-full": "#e45756",
              "dtpo": "#1baf7a", "dtpo-ws": "#eda100"}
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.2), squeeze=False)
    for ax, env in zip(axes[0], envs):
        methods = sorted({k[1] for k in cells if k[0] == env})
        for m in methods:
            pts = sorted((k[2], np.median(v), np.percentile(v, 25), np.percentile(v, 75))
                         for k, v in cells.items() if k[0] == env and k[1] == m)
            if not pts:
                continue
            x, med, q1, q3 = zip(*pts)
            c = colors.get(m, "#0b0b0b")
            ax.plot(x, med, "o-", color=c, lw=2, ms=5, label=m)
            ax.fill_between(x, q1, q3, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({k[2] for k in cells if k[0] == env}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("leaf budget")
        ax.set_ylabel("honest REPORT survival %")
        ax.set_title(env)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Survival vs interpretability budget (median, IQR band, >=5 seeds)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[frontier] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(HPC, "runs"))
    ap.add_argument("--only", default=None, help="single run name")
    ap.add_argument("--skip-trees", action="store_true")
    ap.add_argument("--skip-curves", action="store_true")
    ap.add_argument("--skip-frontier", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(a.runs_dir):
        names = [a.only] if a.only else sorted(os.listdir(a.runs_dir))
        for name in names:
            rundir = os.path.join(a.runs_dir, name)
            if not os.path.isdir(rundir):
                continue
            print(f"== {name}")
            if not a.skip_curves:
                make_curve(rundir, name)
            if not a.skip_trees:
                ckpt = find_ckpt(rundir)
                if ckpt:
                    export_tree(rundir, ckpt, name)
                else:
                    print("  [tree] no checkpoint — skipped")
    else:
        print(f"no runs dir at {a.runs_dir}")

    if not a.skip_frontier:
        csv_path = os.path.join(HPC, "results.csv")
        if os.path.isfile(csv_path):
            make_frontier(csv_path, os.path.join(HPC, "frontier.png"))
        else:
            print("[frontier] run collect_results.py first (no results.csv)")


if __name__ == "__main__":
    main()
