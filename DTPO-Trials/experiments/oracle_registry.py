"""
Oracle readiness.

Evaluate every candidate RL teacher checkpoint per grid based on survival rate.
Pick the strongest per grid and write an oracle registry recording the survival CEILING per gri
Flags grids whose best teacher is too weak (e.g. bus118-M).

Candidates are discovered from filenames, then only candidates are loaded.
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
from common.metrics import rollout_metrics
from eval_ckpt import _build_model, _VecSpec

CANDIDATE_ALGS = {"DQN", "PPO", "LAGRPPO"}

def parse_name(path):
    """Best-effort (alg, env_id) from a checkpoint filename without loading it."""
    base = os.path.basename(path)[:-4]
    if base.startswith("final_"):
        base = base[6:]
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    return parts[0].upper(), parts[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", nargs="+", default=["bus5", "bus14", "bus36-M", "bus118-M"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--adequate", type=float, default=0.20, help="survival below this flags 'needs stronger oracle'")
    p.add_argument("--out", default="experiments/results/oracle_registry.json")
    cli = p.parse_args()
    seeds = list(range(cli.seed0, cli.seed0 + cli.episodes))
    grids = set(cli.grids)
    device = th.device("cpu")

    # discover candidates by filename
    cands = []
    for f in sorted(glob.glob("checkpoint/*.tar")):
        alg, env = parse_name(f)
        if alg in CANDIDATE_ALGS and env in grids:
            cands.append((f, alg, env))
    print(f"[oracle] {len(cands)} candidate teachers across {sorted(grids)}; seeds={seeds}\n")

    results = []
    for f, alg, env in cands:
        name = os.path.basename(f)
        ev = None
        try:
            run = th.load(f, map_location=device, weights_only=False)
            a = run["args"]
            ev = Evaluator(a, None, device)
            model = _build_model(run, _VecSpec(ev.env), a, device)
            m = rollout_metrics(model, ev.env, ev.max_steps, seeds, device="cpu")
            row = dict(grid=env, alg=alg, ckpt=name,
                       survival=m["survival_mean"], std=m["survival_std"],
                       per_ep=m["survival_per_ep"], mean_max_rho=m.get("mean_max_rho"))
            results.append(row)
            print(f"  {env:9s} {alg:7s} surv={m['survival_mean']*100:6.2f}% ± {m['survival_std']*100:5.2f}"
                  f"  rho_max={m.get('mean_max_rho')}  {name}", flush=True)
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                if ev is not None:
                    ev.env.close()
            except Exception:
                pass

    # registry: best teacher per grid
    registry = {}
    for g in cli.grids:
        rows = [r for r in results if r["grid"] == g]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["survival"])
        registry[g] = dict(best_oracle=best["ckpt"], oracle_type=best["alg"],
                           survival_ceiling=best["survival"], survival_std=best["std"],
                           mean_max_rho=best["mean_max_rho"],
                           adequate=bool(best["survival"] >= cli.adequate),
                           n_candidates=len(rows))

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    json.dump(dict(seeds=seeds, adequate_threshold=cli.adequate,
                   registry=registry, all=results), open(cli.out, "w"), indent=2)

    print("\n=== ORACLE REGISTRY (best teacher per grid; warm-start source for the hybrid) ===")
    for g in cli.grids:
        v = registry.get(g)
        if not v:
            print(f"  {g:9s} -> (no candidate found)")
            continue
        flag = "" if v["adequate"] else "   <-- WEAK: train a stronger oracle"
        print(f"  {g:9s} -> {v['oracle_type']:6s} ceiling={v['survival_ceiling']*100:6.2f}% "
              f"(of {v['n_candidates']} cand)  {v['best_oracle']}{flag}")
    print(f"\n[oracle] wrote {cli.out}")


if __name__ == "__main__":
    main()
