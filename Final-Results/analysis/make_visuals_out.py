#!/usr/bin/env python3
import argparse, os, re, subprocess, sys, tempfile, warnings
warnings.filterwarnings("ignore")
def _code_root():
    p = os.environ.get("RL2GRID_CODE")
    if p:
        return os.path.abspath(p)
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, os.pardir, "code"),
              os.path.join(here, os.pardir, os.pardir),
              os.path.join(here, os.pardir), here):
        if os.path.isfile(os.path.join(c, "viz_dtpo_tree.py")):
            return os.path.abspath(c)
    raise SystemExit("RL2Grid code root not found; set RL2GRID_CODE")

CODE = _code_root()
HPC = os.path.join(CODE, "hpc")
PY = os.environ.get("RL2GRID_PY", sys.executable)

def find_ckpt(rundir):
    cdir = os.path.join(rundir, "checkpoint")
    if not os.path.isdir(cdir): return None
    tars = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".tar")]
    if not tars: return None
    finals = [t for t in tars if os.path.basename(t).startswith("final_")]
    return max(finals or tars, key=os.path.getmtime)

def winner_iter(rundir):
    log = os.path.join(rundir, "honest_eval.log")
    if not os.path.isfile(log): return None
    m = None
    for line in open(log, errors="ignore"):
        hit = re.search(r"WINNER iter=(\d+)", line)
        if hit: m = int(hit.group(1))
    return m

def export_tree(rundir, outdir, ckpt):
    import torch as th
    rec = th.load(ckpt, map_location="cpu", weights_only=False)
    tree = rec.get("tree"); wit = winner_iter(rundir); cands = rec.get("candidates") or []
    if wit is not None and cands:
        for c in cands:
            if c.get("iter") == wit: tree = c["tree"]; break
    if tree is None: return "no-tree"
    src = ckpt
    if tree is not rec.get("tree") or "action_map" not in rec:
        tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
        th.save({"tree": tree, "action_map": rec.get("action_kept")}, tmp.name); src = tmp.name
    okc = 0
    for extra, base in ((["--landscape"], "tree_landscape"), ([], "tree")):
        r = subprocess.run([PY, os.path.join(CODE, "viz_dtpo_tree.py"), "--ckpt", src,
                            "--out", os.path.join(outdir, base)] + extra,
                           capture_output=True, text=True)
        if r.returncode == 0: okc += 1
    if src != ckpt: os.unlink(src)
    return f"trees:{okc}/2" + (f" winner_iter={wit}" if wit is not None else " (last tree)")

def make_curve(rundir, outdir, name):
    log = os.path.join(rundir, "train.log")
    if not os.path.isfile(log): return "no-log"
    with open(log, errors="ignore") as f:
        if "[DTPO] it=" not in f.read(): return "not-dtpo"
    r = subprocess.run([PY, os.path.join(CODE, "plot_dtpo_curve.py"), "--log", log,
                        "--out", os.path.join(outdir, "curve.png"), "--title", name],
                       capture_output=True, text=True)
    return "curve:ok" if r.returncode == 0 else "curve:FAIL"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--runs-dir", default=os.path.join(HPC, "runs"))
    a = ap.parse_args()
    runs = a.runs_dir
    names = [a.only] if a.only else sorted(os.listdir(runs))
    n_ok = n_tree = n_curve = 0
    for i, name in enumerate(names, 1):
        rundir = os.path.join(runs, name)
        if not os.path.isdir(rundir): continue
        outdir = os.path.join(a.outroot, name); os.makedirs(outdir, exist_ok=True)
        c = make_curve(rundir, outdir, name)
        ck = find_ckpt(rundir)
        t = export_tree(rundir, outdir, ck) if ck else "no-ckpt"
        if c == "curve:ok": n_curve += 1
        if t.startswith("trees:2"): n_tree += 1
        n_ok += 1
        print(f"[{i:3d}/{len(names)}] {name:44s} {c:10s} {t}", flush=True)
    print(f"\nDONE runs={n_ok} curves={n_curve} full-tree-pairs={n_tree}")

if __name__ == "__main__":
    main()
