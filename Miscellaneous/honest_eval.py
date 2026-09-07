"""
RL2Grid Evaluator samples chronics randomly per reset(), and grids have many chronics of differing difficulty.
Fix via deterministic chronics via env.set_id(cid) -> every candidate is scored on the exact same chronics
Select and report split -> candidates are ranked on the select chronics, then the winner is re-scored on a disjoint held-out report set. 
Usage:
  python honest_eval.py --ckpt dtpo_testing_4/bus14_dagger_seed0/DTPO_bus14_...tar --total 80
"""
import argparse
import numpy as np
import torch as th

from env.eval import Evaluator, CMDPEvaluator
from alg.dtpo.agent import RegressionTreePolicy


def _run_chronic(ev, pol, cid):
    """Pin chronic `cid`, roll the deterministic tree policy to episode end, return survival = nb_time_step / max_steps."""
    ev.env.init_env.set_id(int(cid))
    obs, info = ev.env.reset()
    while True:
        a = pol.get_eval_action(th.tensor(obs, dtype=th.float)).detach().cpu().numpy()
        obs, _, _, _, info = ev.env.step(a)
        if "episode" in info:
            return ev.env.init_env.nb_time_step / ev.max_steps


def _make_pol(tree, ev, kept_arr, device):
    if kept_arr is not None:
        return RegressionTreePolicy(len(kept_arr), device, tree=tree, action_map=kept_arr)
    return RegressionTreePolicy(int(ev.env.action_space.n), device, tree=tree)


def main():
    p = argparse.ArgumentParser(description="Corrected honest eval (fixed chronics + select/report split).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--total", type=int, default=80, help="Total fixed chronics to use (evenly spaced over all). Capped at n_chronics.")
    p.add_argument("--topk", type=int, default=10, help="Candidates (by cheap score) to rank on the SELECT set.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--write", type=lambda s: str(s).lower() in ("1", "true", "yes", "y"), default=False, help="Write the genuinely-best tree back into the checkpoint.")
    cli = p.parse_args()
    device = th.device(cli.device)

    run = th.load(cli.ckpt, map_location=device, weights_only=False)
    args = run["args"]
    cands = run.get("candidates") or []
    if not cands:
        if run.get("tree") is None:
            raise SystemExit("No candidates and no tree.")
        cands = [{"tree": run["tree"], "iter": run.get("last_iter"), "cheap_survival": run.get("best_score") or 0.0}]
    cands = sorted(cands, key=lambda c: (c.get("cheap_survival") or 0.0), reverse=True)[:max(1, cli.topk)]

    kept = run.get("action_kept")
    kept_arr = None if kept is None else np.asarray(kept, dtype=np.int64)

    ev = (CMDPEvaluator if getattr(args, "constraints_type", 0) != 0 else Evaluator)(args, None, device)
    n_chronics = len(ev.env.init_env.chronics_handler.real_data.subpaths)
    total = min(cli.total, n_chronics)
    ids = np.unique(np.linspace(0, n_chronics - 1, total).astype(int))
    select_ids, report_ids = ids[0::2], ids[1::2]

    print(f"[honest] ckpt={cli.ckpt}")
    print(f"[honest] env={args.env_id} n_chronics={n_chronics} using={len(ids)} "
          f"(select={len(select_ids)} / report={len(report_ids)} held-out) candidates={len(cands)} curated={'yes' if kept_arr is not None else 'no'}")

    # rank candidates on the K_select chronics
    ranked = []
    for c in cands:
        pol = _make_pol(c["tree"], ev, kept_arr, device)
        s = np.array([_run_chronic(ev, pol, cid) for cid in select_ids])
        ranked.append((float(s.mean()), c))
        print(f"[honest] iter={c.get('iter')} cheap={(c.get('cheap_survival') or 0)*100:6.2f}% " f"-> SELECT({len(select_ids)})={s.mean()*100:6.2f}%")
    ranked.sort(key=lambda t: t[0], reverse=True)
    best_sel, best_c = ranked[0]

    # re-score the winner on the disjoint held-out K_report chronics
    pol = _make_pol(best_c["tree"], ev, kept_arr, device)
    rep = np.array([_run_chronic(ev, pol, cid) for cid in report_ids])
    print(f"\n[honest] WINNER iter={best_c.get('iter')} leaves={best_c['tree'].get_n_leaves()}")
    print(f"[honest]   select-set survival = {best_sel*100:.2f}%  ({len(select_ids)} chronics)")
    print(f"[honest]   HELD-OUT REPORT survival = {rep.mean()*100:.2f}%  ({len(report_ids)} chronics)" f"[min {rep.min()*100:.0f}% / max {rep.max()*100:.0f}% / frac@100% {(rep>=0.999).mean()*100:.0f}%]")

    if cli.write:
        run["tree"] = best_c["tree"]
        run["best_score"] = float(rep.mean())
        run["honest_eval"] = {"total": int(total), "select": best_sel, "report": float(rep.mean()), "report_per_chronic": rep.tolist(), "best_iter": best_c.get("iter")}
        th.save(run, cli.ckpt)
        print(f"[honest] wrote winner + honest_eval back into {cli.ckpt}")


if __name__ == "__main__":
    main()
