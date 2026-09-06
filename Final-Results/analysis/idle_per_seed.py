#!/usr/bin/env python3
"""Score the do-nothing policy at several --seed values, so the baseline is
graded under the same chronics permutation as the policies it is quoted
against.

Usage: idle_per_seed.py <env-id> <seeds csv> [total] [holdout]
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
from env.config import get_env_args
from env.eval import Evaluator

class Idle:
    def get_eval_action(self, x): return th.tensor(0)

def run(env_id, seeds, total=80, holdout=4):
    for sd in seeds:
        sys.argv = ["x", "--env-id", env_id, "--difficulty", "0",
                    "--action-type", "topology", "--chronic-holdout", str(holdout),
                    "--eval-seed", "12345", "--eval-norm-reset", "per-chronic",
                    "--chronics-mode", "multifolder",
                    "--use-heuristic", "False", "--heuristic-type", "reconnect"]
        args = get_env_args()
        args.seed, args.alg, args.cuda, args.track = sd, "DTPO", False, False
        t0 = time.time()
        ev = Evaluator(args, None, th.device("cpu"))
        n = ev.n_chronics()
        ids = ev.fixed_chronic_ids(min(total, n))
        sel, rep = ids[0::2], ids[1::2]
        s = ev.evaluate_fixed(Idle(), sel)["survival"]
        R = np.asarray(ev.evaluate_fixed(Idle(), rep)["per_chronic"])
        print(f"IDLE env={env_id} seed={sd} select={100*s:.2f} report={100*R.mean():.2f} "
              f"min={100*R.min():.2f} max={100*R.max():.2f} n={len(R)} "
              f"distinct={len(set(np.round(R,6).tolist()))} secs={time.time()-t0:.0f}", flush=True)

if __name__ == "__main__":
    env = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]
    tot = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    hold = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    run(env, seeds, tot, hold)
