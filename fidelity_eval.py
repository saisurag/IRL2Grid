#!/usr/bin/env python
"""
Tree-vs-oracle fidelity for a DTPO/hybrid/VIPER checkpoint.
"""

import argparse, json
import numpy as np
import torch as th

from env.eval import Evaluator, CMDPEvaluator
from eval_ckpt import _build_model, _VecSpec
from common.action_reduction import dqn_score_fn, ppo_score_fn
from common.metrics import fidelity

def _oracle_score_fn(model, alg, device):
    """Per-action score vector (DQN Q-values / PPO logits); argmax=action, top1-top2=criticality."""
    alg = alg.upper()
    if alg == "DQN":
        return dqn_score_fn(model, device)
    if alg in ("PPO", "SREINFORCE", "DAGGER", "LAGRPPO"):
        return ppo_score_fn(model, device)
    raise ValueError(f"Unsupported oracle alg for scoring: {alg!r}")

def _tree_action(policy, obs, device):
    a = policy.get_eval_action(th.tensor(obs, dtype=th.float).to(device)).detach().cpu().numpy()
    return int(np.asarray(a).ravel()[0])

def _collect(env, max_steps, driver, tree_policy, score_fn, episodes, seed, device):
    """Roll out `episodes` driven by oracle or tree"""
    tree_a, orac_a, gaps, survivals = [], [], [], []
    obs, info = env.reset(seed=int(seed))
    n_ep = 0
    while n_ep < episodes:
        v = np.asarray(score_fn(obs)).ravel()
        order = np.argsort(v)[::-1]
        o_act = int(order[0])
        gap = float(v[order[0]] - v[order[1]]) if v.size > 1 else 0.0
        t_act = _tree_action(tree_policy, obs, device)

        tree_a.append(t_act); orac_a.append(o_act); gaps.append(gap)

        step_a = np.int64(o_act) if driver == "oracle" else np.int64(t_act)
        obs, _, _, _, info = env.step(step_a)
        if "episode" in info:
            survivals.append(env.init_env.nb_time_step / max_steps)
            obs, _ = env.reset()
            n_ep += 1
    return (np.array(tree_a), np.array(orac_a), np.array(gaps),
            float(np.mean(survivals)) if survivals else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree-ckpt", required=True)
    p.add_argument("--oracle-ckpt", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    cli = p.parse_args()
    device = th.device(cli.device)

    tree_run = th.load(cli.tree_ckpt, map_location=device, weights_only=False)
    orac_run = th.load(cli.oracle_ckpt, map_location=device, weights_only=False)
    targs, oargs = tree_run["args"], orac_run["args"]

    # Build the eval env from the TREE's args (same grid the oracle was trained on).
    eval_cls = CMDPEvaluator if getattr(targs, "constraints_type", 0) != 0 else Evaluator
    evaluator = eval_cls(targs, None, device)
    vec = _VecSpec(evaluator.env)

    tree_policy = _build_model(tree_run, vec, targs, device)
    oracle_model = _build_model(orac_run, vec, oargs, device)
    score_fn = _oracle_score_fn(oracle_model, oargs.alg, device)

    out = {"tree_ckpt": cli.tree_ckpt, "oracle_ckpt": cli.oracle_ckpt,
           "grid": targs.env_id, "oracle_alg": oargs.alg.upper(),
           "episodes": cli.episodes, "seed": cli.seed}

    for driver in ("oracle", "tree"):
        ta, oa, gaps, surv = _collect(evaluator.env, evaluator.max_steps, driver,
                                      tree_policy, score_fn, cli.episodes, cli.seed, device)
        fid = fidelity(ta, oa, weights=gaps)
        out[f"on_{driver}_states"] = {
            "fidelity": round(fid["fidelity"], 4),
            "weighted_fidelity": round(fid.get("weighted_fidelity", float("nan")), 4),
            "n_states": fid["n"],
            "driver_survival": round(surv, 4),
        }
        print(f"[fidelity] {targs.env_id} on {driver}-states: "
              f"fid={fid['fidelity']:.3f} wfid={fid.get('weighted_fidelity', float('nan')):.3f} "
              f"n={fid['n']} driver_surv={surv:.3f}")

    if cli.out:
        with open(cli.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[fidelity] wrote {cli.out}")


if __name__ == "__main__":
    main()
