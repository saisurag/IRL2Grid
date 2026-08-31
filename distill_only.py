"""
Distill-only baseline (no RL fine-tuning)

Usage:
  python distill_only.py --env-id bus14 --oracle checkpoint/final_PPO_bus14_...tar \
      --tree-max-leaf-nodes 16 --temperature 0.5 --seed 100 --out checkpoint/
"""
import argparse
import os
from time import time

import numpy as np
import torch as th

from alg.distill import load_oracle_score_fn, distill
from common.action_reduction import ActionReducer
from common.utils import set_random_seed, str2bool
from env.config import get_env_args
from env.eval import Evaluator

class _VecSpec:
    """Minimal vec-env shim over the Evaluator's single env (as in eval_ckpt)."""
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


def main():
    p = argparse.ArgumentParser(description="Distill-only baseline -> DTPO-format ckpt.")
    p.add_argument("--oracle", required=True, help="Oracle checkpoint (.tar path or run name)")
    p.add_argument("--oracle-type", type=str, default="auto", choices=["auto", "dqn", "ppo"])
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--distill-seeds", type=int, default=5, help="Seeded rollout episodes for the dataset")
    p.add_argument("--steps-cap", type=int, default=20000, help="Distillation dataset size cap")
    p.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for soft targets")
    p.add_argument("--tree-max-leaf-nodes", type=int, default=16)
    p.add_argument("--tree-max-depth", type=int, default=None)
    p.add_argument("--tree-min-samples-leaf", type=int, default=1)
    p.add_argument("--curate", type=str2bool, default=False, help="Curate the action set before distilling")
    p.add_argument("--curate-strategy", type=str, default="coverage+criticality")
    p.add_argument("--coverage", type=float, default=0.999)
    p.add_argument("--dagger-rounds", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    cli, _ = p.parse_known_args()

    args = argparse.Namespace(**vars(cli), **vars(get_env_args()))
    args.alg = "DTPO"                    # so downstream loaders build a tree policy
    args.cuda = False
    set_random_seed(args.seed)

    device = th.device(cli.device)
    evaluator = Evaluator(args, None, device)
    vec = _VecSpec(evaluator.env)
    n_actions = int(evaluator.env.action_space.n)

    score_fn = load_oracle_score_fn(cli.oracle, vec, device, cli.oracle_type)
    seeds = list(range(args.seed, args.seed + cli.distill_seeds))

    reducer = None
    if cli.curate:
        reducer = ActionReducer.from_oracle(
            score_fn, evaluator.env, evaluator.max_steps, seeds=seeds,
            strategy=cli.curate_strategy, coverage=cli.coverage)
        reducer.log()

    policy, dinfo = distill(
        score_fn, evaluator.env, seeds, n_actions=n_actions, device=device,
        mode="soft", temperature=cli.temperature, action_reducer=reducer,
        max_leaf_nodes=cli.tree_max_leaf_nodes, max_depth=cli.tree_max_depth,
        min_samples_leaf=cli.tree_min_samples_leaf, steps_cap=cli.steps_cap,
        dagger_rounds=cli.dagger_rounds)
    print(f"[DISTILL-ONLY] {dinfo}")

    run_name = (f"DISTILL_{args.env_id}_T_{args.seed}_{args.difficulty}_"
                f"L{cli.tree_max_leaf_nodes}_{int(time())}_{np.random.randint(0, 50000)}")
    os.makedirs("checkpoint", exist_ok=True)

    record = {
        "args": args,
        "tree": policy.tree,
        "global_step": 0,
        "wb_run_name": "",
        "last_iter": 0,
        "best_score": None,
        "action_kept": (reducer.kept if reducer is not None else None),
        "candidates": [{"tree": policy.tree, "iter": 0, "step": 0, "cheap_survival": 0.0}],
        "distill_info": dinfo,
    }
    
    path = f"checkpoint/{run_name}.tar"
    th.save(record, path)
    print(f"[DISTILL-ONLY] saved {path} "
          f"(leaves={policy.tree.get_n_leaves()}, curated={'yes' if reducer is not None else 'no'})")


if __name__ == "__main__":
    main()
