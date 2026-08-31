"""
Unified honest evaluation for any RL2Grid checkpoint.
Usage:
  python honest_eval_any.py --ckpt <run.tar> --total 80
"""
import argparse
import os
import numpy as np
import torch as th

from env.eval import Evaluator, CMDPEvaluator
from alg.dtpo.agent import RegressionTreePolicy
from eval_ckpt import _VecSpec, _build_model


def _tree_policy(tree, ev, kept_arr, device):
    if kept_arr is not None:
        return RegressionTreePolicy(len(kept_arr), device, tree=tree, action_map=kept_arr)
    return RegressionTreePolicy(int(ev.env.action_space.n), device, tree=tree)


def _tree_stats(model):
    """(n_leaves, depth) if the policy is tree-backed, else (None, None)."""
    tree = getattr(model, "tree", None)
    if tree is None:
        return None, None
    try:
        return int(tree.get_n_leaves()), int(tree.get_depth())
    except Exception:
        return None, None


def main():
    p = argparse.ArgumentParser(description="Unified honest eval (fixed chronics + select/report split).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--total", type=int, default=80,
                   help="Total fixed chronics (evenly spaced over all; capped at n_chronics).")
    p.add_argument("--topk", type=int, default=10,
                   help="For multi-candidate ckpts: candidates (by cheap score) ranked on SELECT.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chronic-holdout", type=int, default=None)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--eval-norm-reset", type=str, default=None, choices=["per-chronic", "once"])
    p.add_argument("--chronics-mode", type=str, default=None, choices=["multifolder", "cache"])
    p.add_argument("--use-heuristic", type=str, default=None, choices=["True", "False"])
    p.add_argument("--eval-pool", type=str, default="held", choices=["held", "train"])
    cli = p.parse_args()
    device = th.device(cli.device)

    run = th.load(cli.ckpt, map_location=device, weights_only=False)
    args = run["args"]
    alg = str(args.alg).upper()
    if cli.use_heuristic is not None:
        _uh = (cli.use_heuristic == "True")
        if bool(getattr(args, "use_heuristic", False)) != _uh:
            print(f"[honest-any] OVERRIDE use_heuristic "
                  f"{getattr(args, 'use_heuristic', None)} -> {_uh} "
                  f"(checkpoint was trained with heuristic="
                  f"{getattr(args, 'use_heuristic', None)}:"
                  f"{getattr(args, 'heuristic_type', None)})")
        args.use_heuristic = _uh
    default_values_new = dict(chronic_holdout=4, eval_seed=12345,
                        eval_norm_reset="per-chronic", chronics_mode="multifolder")
    for _k, _d in default_values_new.items():
        _cli = getattr(cli, _k)
        if _cli is not None:
            setattr(args, _k, _cli)
        elif getattr(args, _k, None) is None:
            setattr(args, _k, _d)
    print(f"[honest-any] PROTOCOL v2: pool={cli.eval_pool} holdout={args.chronic_holdout} "
          f"eval_seed={args.eval_seed} norm_reset={args.eval_norm_reset} "
          f"chronics={args.chronics_mode}")

    ev = (CMDPEvaluator if getattr(args, "constraints_type", 0) != 0 else Evaluator)(args, None, device)
    n_chronics = ev.n_chronics()
    ids = (ev.fixed_chronic_ids(min(cli.total, n_chronics)) if cli.eval_pool == "held"
           else ev.gate_chronic_ids(min(cli.total, n_chronics)))
    select_ids, report_ids = ids[0::2], ids[1::2]

    print(f"[honest-any] ckpt={cli.ckpt}")
    print(f"[honest-any] alg={alg} env={args.env_id} difficulty={getattr(args, 'difficulty', '?')} "
          f"n_chronics={n_chronics} using={len(ids)} (select={len(select_ids)} / report={len(report_ids)})")

    kept = run.get("action_kept")
    kept_arr = None if kept is None else np.asarray(kept, dtype=np.int64)
    cands = run.get("candidates") or []

    if alg == "DTPO" and cands:
        # rank stored candidate trees on SELECT, report the winner on REPORT
        cands = sorted(cands, key=lambda c: (c.get("cheap_survival") or 0.0), reverse=True)[:max(1, cli.topk)]
        ranked = []
        for c in cands:
            pol = _tree_policy(c["tree"], ev, kept_arr, device)
            r = ev.evaluate_fixed(pol, select_ids)
            ranked.append((r["survival"], c))
            print(f"[honest-any] cand iter={c.get('iter')} cheap={(c.get('cheap_survival') or 0)*100:6.2f}% "
                  f"-> SELECT={r['survival']*100:6.2f}%")
        ranked.sort(key=lambda t: t[0], reverse=True)
        best_sel, best_c = ranked[0]
        model = _tree_policy(best_c["tree"], ev, kept_arr, device)
        picked = f"iter={best_c.get('iter')}"
    else:
        # single stored policy
        model = _build_model(run, _VecSpec(ev.env), args, device)
        r = ev.evaluate_fixed(model, select_ids)
        best_sel, picked = r["survival"], "single"
        print(f"[honest-any] single policy -> SELECT={best_sel*100:6.2f}%")

    rep = np.asarray(ev.evaluate_fixed(model, report_ids)["per_chronic"])
    n_leaves, depth = _tree_stats(model)

    print(f"\n[honest-any] WINNER {picked}"
          + (f" leaves={n_leaves} depth={depth}" if n_leaves is not None else " (non-tree policy)"))
    print(f"[honest-any]   select-set survival = {best_sel*100:.2f}%  ({len(select_ids)} chronics)")
    print(f"[honest-any]   HELD-OUT REPORT survival = {rep.mean()*100:.2f}%  ({len(report_ids)} chronics)  "
          f"[min {rep.min()*100:.0f}% / max {rep.max()*100:.0f}% / frac@100% {(rep >= 0.999).mean()*100:.0f}%]")

    _cdir = os.path.dirname(os.path.abspath(cli.ckpt))
    run_dir = os.path.basename(os.path.dirname(_cdir) if os.path.basename(_cdir) == 'checkpoint'
                               else _cdir)
    print(f"RESULT alg={alg} run={run_dir} env={args.env_id} seed={getattr(args, 'seed', '?')} "
          f"difficulty={getattr(args, 'difficulty', '?')} leaves={n_leaves if n_leaves is not None else 'NA'} "
          f"depth={depth if depth is not None else 'NA'} select={best_sel*100:.2f} report={rep.mean()*100:.2f} "
          f"report_min={rep.min()*100:.2f} report_max={rep.max()*100:.2f} frac100={(rep >= 0.999).mean()*100:.1f} "
          f"n_report={len(report_ids)}")


if __name__ == "__main__":
    main()
