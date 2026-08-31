#!/usr/bin/env python3
"""
Score the do-nothing (always action 0) policy 
Usage:
  python eval_idle.py --env-id bus14 --total 80
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch as th

from env.config import get_env_args
from env.eval import Evaluator


class Idle:
    def get_eval_action(self, x):
        return th.tensor(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--total", type=int, default=80)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--holdout", type=int, default=4)
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--use-heuristic", default="False", choices=["True", "False"])
    p.add_argument("--heuristic-type", default="reconnect", choices=["idle", "reconnect"])
    cli = p.parse_args()

    sys.argv = ["eval_idle", "--env-id", cli.env_id, "--difficulty", str(cli.difficulty),
                "--action-type", "topology", "--chronic-holdout", str(cli.holdout),
                "--eval-seed", str(cli.eval_seed), "--eval-norm-reset", "per-chronic",
                "--chronics-mode", "multifolder",
                "--use-heuristic", cli.use_heuristic, "--heuristic-type", cli.heuristic_type]
    args = get_env_args()
    args.seed, args.alg, args.cuda, args.track = 100, "DTPO", False, False

    ev = Evaluator(args, None, th.device("cpu"))
    n = ev.n_chronics()
    m = Idle()

    for pool, ids in (("held", ev.fixed_chronic_ids(min(cli.total, n))),
                      ("train", ev.gate_chronic_ids(min(cli.total, n)))):
        sel, rep = ids[0::2], ids[1::2]
        s = ev.evaluate_fixed(m, sel)["survival"]
        R = np.asarray(ev.evaluate_fixed(m, rep)["per_chronic"])
        print(f"RESULT_IDLE env={cli.env_id} pool={pool} heur={cli.use_heuristic} "
              f"n_chronics={n} select={100*s:.2f} report={100*R.mean():.2f} "
              f"report_min={100*R.min():.2f} report_max={100*R.max():.2f} "
              f"frac100={100*(R >= 0.999).mean():.1f} n_report={len(R)} "
              f"distinct={len(set(np.round(R, 6).tolist()))}")


if __name__ == "__main__":
    main()
