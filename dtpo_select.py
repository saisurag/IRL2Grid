"""
Honest final tree-selection for DTPO checkpoints.
Usage:
    python dtpo_select.py --ckpt checkpoint/DTPO_bus36-M_...tar --episodes 100 --topk 5
"""
import argparse

import numpy as np
import torch as th

from env.eval import Evaluator, CMDPEvaluator
from alg.dtpo.agent import RegressionTreePolicy


def _str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "t")


def main():
    p = argparse.ArgumentParser(description="Honest final tree-selection for a DTPO checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to a DTPO .tar checkpoint with stored candidates.")
    p.add_argument("--episodes", type=int, default=100, help="Fresh-eval episodes per candidate.")
    p.add_argument("--topk", type=int, default=5, help="How many top candidates (by cheap score) to deep-eval.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--write", type=_str2bool, default=True, help="Write the selected tree back into the checkpoint.")
    cli = p.parse_args()
    device = th.device(cli.device)

    run = th.load(cli.ckpt, map_location=device, weights_only=False)
    if "args" not in run or getattr(run["args"], "alg", "").upper() != "DTPO":
        raise SystemExit("Not a DTPO checkpoint.")
    args = run["args"]

    cands = run.get("candidates") or []
    if not cands:
        if run.get("tree") is None:
            raise SystemExit("Checkpoint has neither candidates nor a tree to evaluate.")
        print("[select] no stored candidates; evaluating the single stored tree.")
        cands = [{"tree": run["tree"], "iter": run.get("last_iter"), "cheap_survival": run.get("best_score") or 0.0}]

    cands = sorted(cands, key=lambda c: (c.get("cheap_survival") or 0.0), reverse=True)[:max(1, cli.topk)]

    kept = run.get("action_kept")
    kept_arr = None if kept is None else np.asarray(kept, dtype=np.int64)

    print(f"[select] ckpt={cli.ckpt}")
    print(f"[select] env={args.env_id} candidates={len(cands)} episodes={cli.episodes} "
          f"curated={'yes' if kept_arr is not None else 'no'}")

    results, best = [], None
    for c in cands:
        evaluator = (CMDPEvaluator if getattr(args, "constraints_type", 0) != 0 else Evaluator)(args, None, device)
        n_act = int(evaluator.env.action_space.n)
        if kept_arr is not None:
            pol = RegressionTreePolicy(len(kept_arr), device, tree=c["tree"], action_map=kept_arr)
        else:
            pol = RegressionTreePolicy(n_act, device, tree=c["tree"])
        surv = float(evaluator.evaluate(0, pol, eval_ep=cli.episodes)["survival"])
        results.append({"iter": c.get("iter"), "cheap": float(c.get("cheap_survival") or 0.0), "honest": surv, "leaves": int(c["tree"].get_n_leaves())})
        print(f"[select] iter={c.get('iter')} cheap={(c.get('cheap_survival') or 0)*100:6.2f}% "
              f"-> {cli.episodes}ep={surv*100:6.2f}%  leaves={c['tree'].get_n_leaves()}")
        if best is None or surv > best["honest"]:
            best = {"tree": c["tree"], "iter": c.get("iter"), "honest": surv,
                    "leaves": int(c["tree"].get_n_leaves())}
        del evaluator

    print(f"[select] HONEST BEST = {best['honest']*100:.2f}% over {cli.episodes} ep "
          f"(iter={best['iter']}, leaves={best['leaves']})")

    if cli.write:
        run["tree"] = best["tree"]
        run["best_score"] = best["honest"]
        run["selection"] = {"episodes": cli.episodes, "topk": cli.topk, "results": results, "best_iter": best["iter"], "best_survival": best["honest"]}
        th.save(run, cli.ckpt)
        print(f"[select] wrote honestly-selected tree back into {cli.ckpt}")


if __name__ == "__main__":
    main()
