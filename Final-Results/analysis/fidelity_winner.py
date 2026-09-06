#!/usr/bin/env python3
"""Tree-vs-oracle fidelity for the policy a run actually delivered.

Reads the winning candidate from the run's honest_eval.log and scores that tree,
rather than the final training tree that fidelity_eval.py reconstructs. Rollout,
scoring and metric are reused unchanged from fidelity_eval.py. Read-only on the
code/ tree.

Usage: fidelity_winner.py <run-dir> <oracle-ckpt> [episodes] [collect-seed]
"""
import sys, os, re, glob, time, warnings
warnings.filterwarnings("ignore")
def _code_root():
    p = os.environ.get("RL2GRID_CODE")
    if p:
        return os.path.abspath(p)
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, os.pardir, "code"),
              os.path.join(here, os.pardir, os.pardir),
              os.path.join(here, os.pardir), here):
        if os.path.isfile(os.path.join(c, "env", "eval.py")):
            return os.path.abspath(c)
    raise SystemExit("RL2Grid code root not found; set RL2GRID_CODE")

CODE = _code_root()
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, torch as th
from env.eval import Evaluator
from eval_ckpt import _build_model, _VecSpec
from common.metrics import fidelity
from fidelity_eval import _collect, _oracle_score_fn
from honest_eval_any import _tree_policy, _tree_stats

def run(run_dir, oracle_ckpt, episodes, collect_seed):
    dev = th.device("cpu")
    ck = glob.glob(os.path.join(run_dir, "checkpoint", "*.tar"))[0]
    log = os.path.join(run_dir, "honest_eval.log")
    r = th.load(ck, map_location=dev, weights_only=False)
    a = r["args"]
    a.cuda = a.track = False
    t0 = time.time()
    ev = Evaluator(a, None, dev)
    vec = _VecSpec(ev.env)

    m = re.search(r"WINNER iter=(\d+)", open(log).read())
    if m:
        kept = r.get("action_kept")
        kept_arr = None if kept is None else np.asarray(kept, dtype=np.int64)
        c = [c for c in (r.get("candidates") or []) if c.get("iter") == int(m.group(1))]
        if not c:
            raise ValueError(f"winner iter={m.group(1)} not among stored candidates")
        tree_policy, picked = _tree_policy(c[0]["tree"], ev, kept_arr, dev), f"iter={m.group(1)}"
    else:
        tree_policy, picked = _build_model(r, vec, a, dev), "single"

    orun = th.load(oracle_ckpt, map_location=dev, weights_only=False)
    score_fn = _oracle_score_fn(_build_model(orun, vec, orun["args"], dev),
                                orun["args"].alg, dev)
    leaves, depth = _tree_stats(tree_policy)
    out = {}
    for driver in ("oracle", "tree"):
        ta, oa, gaps, surv = _collect(ev.env, ev.max_steps, driver, tree_policy,
                                      score_fn, episodes, collect_seed, dev)
        f = fidelity(ta, oa, weights=gaps)
        out[driver] = (f["fidelity"], f.get("weighted_fidelity", float("nan")), f["n"], surv)
    o, t = out["oracle"], out["tree"]
    print(f"FIDELITY run={os.path.basename(run_dir)} seed={a.seed} picked={picked} "
          f"leaves={leaves} depth={depth} episodes={episodes} collect_seed={collect_seed} "
          f"orac_fid={o[0]:.4f} orac_wfid={o[1]:.4f} orac_n={o[2]} orac_surv={o[3]:.4f} "
          f"tree_fid={t[0]:.4f} tree_wfid={t[1]:.4f} tree_n={t[2]} tree_surv={t[3]:.4f} "
          f"secs={time.time()-t0:.0f}", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else 10,
        int(sys.argv[4]) if len(sys.argv) > 4 else 0)
