#!/usr/bin/env python3
"""Score a PPO oracle at several --seed values with the observation normaliser
warmed on a training-pool burn-in set and then frozen.

Usage: oracle_honest_eval.py <env-id> <seeds csv> [burnin]
"""
import sys, os, time, warnings
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

def _ckpt(name):
    for d in ("checkpoint", "checkpoints"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"checkpoint not found in checkpoint/ or checkpoints/: {name}")

from env.eval import Evaluator
from eval_ckpt import _VecSpec, _build_model

CK = {
 "bus14":   ("final_PPO_bus14_T_0_0__I__1775940444_3936.tar", 4, 80),
 "bus5":    ("final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar", 2, 10),
 "bus36-M": ("final_PPO_bus36-M_T_100_0__I__1784927195_14574.tar", 4, 80),
}

def run(ckpt, tag, seeds, burnin, total, holdout):
    dev = th.device("cpu")
    for sd in seeds:
        r = th.load(ckpt, map_location=dev, weights_only=False)
        a = r["args"]
        a.chronic_holdout, a.eval_seed = holdout, 12345
        a.eval_norm_reset, a.chronics_mode = "per-chronic", "multifolder"
        a.use_heuristic, a.cuda, a.track = False, False, False
        a.seed = sd
        t0 = time.time()
        ev = Evaluator(a, None, dev)
        ids = ev.fixed_chronic_ids(min(total, ev.n_chronics()))
        sel, rep = ids[0::2], ids[1::2]
        m = _build_model(r, _VecSpec(ev.env), a, dev)
        nw = ev._norm_wrapper()
        assert nw is not None

        warm = ev.gate_chronic_ids(burnin)
        nw.reset_stats(); nw.training = True
        ev.evaluate_fixed(m, warm, reset_norm=False)
        nw.training = False

        s = ev.evaluate_fixed(m, sel, reset_norm=False)["survival"]
        R = np.asarray(ev.evaluate_fixed(m, rep, reset_norm=False)["per_chronic"])
        print(f"ORACLE env={tag} seed={sd} holdout={holdout} burnin={len(warm)} "
              f"select={100*s:.2f} report={100*R.mean():.2f} min={100*R.min():.2f} "
              f"max={100*R.max():.2f} n={len(R)} frac100={100*(R>=0.999).mean():.1f} "
              f"secs={time.time()-t0:.0f}", flush=True)

if __name__ == "__main__":
    tag = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(",")]
    burnin = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ck, hold, tot = CK[tag]; ck = _ckpt(ck)
    run(ck, tag, seeds, burnin, tot, hold)
