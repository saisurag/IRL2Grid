"""
survival-vs-#leaves interpretability frontier.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.eval import Evaluator
from alg.dtpo.agent import RegressionTreePolicy
from alg.distill import load_oracle_score_fn, collect_oracle_dataset, distill_from_dataset
from common.metrics import rollout_metrics, tree_interpretability, pareto_frontier, format_frontier
from eval_ckpt import _VecSpec


def _point(method, surv_metrics, interp, extra=None):
    p = dict(method=method, survival_mean=surv_metrics["survival_mean"],
             survival_std=surv_metrics["survival_std"],
             mean_max_rho=surv_metrics.get("mean_max_rho"),
             n_leaves=interp["n_leaves"], max_depth=interp["max_depth"],
             n_features=interp["n_features_used"], mean_path=round(interp["mean_path_length"], 3))
    if extra:
        p.update(extra)
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True, help="oracle checkpoint name (defines the grid + env args)")
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--distill-seeds", type=int, default=5)
    p.add_argument("--distill-steps-cap", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-seed0", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-viper", type=int, default=4, help="cap VIPER checkpoints ingested as the strong-distillation baseline")
    p.add_argument("--out", default=None)
    cli = p.parse_args()
    device = th.device("cpu")
    eval_seeds = list(range(cli.eval_seed0, cli.eval_seed0 + cli.eval_episodes))

    # Build the grid's eval env from the oracle's arguments
    orec = th.load(cli.oracle if cli.oracle.endswith(".tar") else f"checkpoint/{cli.oracle}.tar",
                   map_location=device, weights_only=False)
    oargs = orec["args"]
    grid = oargs.env_id
    ev = Evaluator(oargs, None, device)
    n_actions = int(ev.env.action_space.n)
    score_fn = load_oracle_score_fn(cli.oracle, _VecSpec(ev.env), device)
    print(f"[frontier] grid={grid} n_actions={n_actions} oracle={cli.oracle} eval_seeds={eval_seeds}", flush=True)

    points = []

    # Oracle ceiling
    from eval_ckpt import _build_model
    omodel = _build_model(orec, _VecSpec(ev.env), oargs, device)
    om = rollout_metrics(omodel, ev.env, ev.max_steps, eval_seeds, device="cpu")
    points.append(dict(method="oracle", survival_mean=om["survival_mean"], survival_std=om["survival_std"],
                       mean_max_rho=om.get("mean_max_rho"), n_leaves=None))
    print(f"  oracle           survival={om['survival_mean']*100:6.2f}%  rho={om.get('mean_max_rho')}", flush=True)

    # Distillation sweep
    ds_seeds = list(range(cli.eval_seed0 + 100, cli.eval_seed0 + 100 + cli.distill_seeds))
    O, S, A, W = collect_oracle_dataset(score_fn, ev.env, ds_seeds, steps_cap=cli.distill_steps_cap)
    print(f"  [distill dataset] {len(O)} samples", flush=True)
    for b in list(cli.budgets) + [None]:
        policy, info, tree = distill_from_dataset(O, S, A, W, n_actions=n_actions, device=device,
                                                  mode="soft", temperature=cli.temperature,
                                                  max_leaf_nodes=b)
        interp = tree_interpretability(tree)
        m = rollout_metrics(policy, ev.env, ev.max_steps, eval_seeds, device="cpu")
        tag = f"distill@{b}" if b is not None else "distill@free"
        points.append(_point(tag, m, interp, extra=dict(fidelity=round(info["fidelity"], 4),
                                                         weighted_fidelity=round(info["weighted_fidelity"], 4))))
        print(f"  {tag:14s} leaves={interp['n_leaves']:3d} depth={interp['max_depth']:2d} "
              f"feat={interp['n_features_used']:3d} survival={m['survival_mean']*100:6.2f}%  "
              f"wfid={info['weighted_fidelity']:.3f}", flush=True)

    # Tree-policy checkpoints
    from eval_ckpt import _build_model
    viper_files = sorted(glob.glob(f"checkpoint/*VIPER_{grid}*.tar"))[: cli.max_viper]
    ckpt_files = sorted(glob.glob(f"checkpoint/DTPO_{grid}*.tar")) + viper_files
    for f in ckpt_files:
        try:
            r = th.load(f, map_location=device, weights_only=False)
            a = r.get("args")
            tree = r.get("tree")
            if tree is None:
                continue
            pol = _build_model(r, _VecSpec(ev.env), a, device)   # DecisionTreePolicy / RegressionTreePolicy
            interp = tree_interpretability(tree)
            m = rollout_metrics(pol, ev.env, ev.max_steps, eval_seeds, device="cpu")
            alg = a.alg.upper()
            if alg == "VIPER":
                tag = "viper"
            else:  # DTPO
                tag = "hybrid" if getattr(a, "warmstart_oracle", "") else "dtpo-scratch"
            points.append(_point(f"{tag}@{interp['n_leaves']}", m, interp,
                                 extra=dict(ckpt=os.path.basename(f), best_score=r.get("best_score"))))
            print(f"  {tag:12s}@{interp['n_leaves']:<3d} survival={m['survival_mean']*100:6.2f}%  "
                  f"({os.path.basename(f)})", flush=True)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}", flush=True)

    # Pareto frontier over tree-based policie
    tree_pts = [p for p in points if p.get("n_leaves") is not None]
    front = pareto_frontier(tree_pts, x="n_leaves", y="survival_mean")
    front_ids = {id(p) for p in front}

    out = cli.out or f"experiments/results/frontier_{grid}.json"
    json.dump(dict(grid=grid, oracle=cli.oracle, eval_seeds=eval_seeds, points=points,
                   pareto=[p["method"] for p in front]), open(out, "w"), indent=2)

    print(f"\n=== FRONTIER ({grid}) — survival vs #leaves ===")
    print(f"  {'':1} {'method':>16} {'leaves':>7} {'survival':>9} {'depth':>6} {'feat':>5}")
    for pt in sorted(tree_pts, key=lambda d: (d["n_leaves"], -d["survival_mean"])):
        mark = "*" if id(pt) in front_ids else " "
        print(f"  {mark} {pt['method']:>16} {pt['n_leaves']:>7} {pt['survival_mean']*100:>8.2f}% "
              f"{pt['max_depth']:>6} {pt['n_features']:>5}")
    print(f"  oracle ceiling = {om['survival_mean']*100:.2f}%   (* = Pareto-efficient)")
    print(f"[frontier] wrote {out}")


if __name__ == "__main__":
    main()
