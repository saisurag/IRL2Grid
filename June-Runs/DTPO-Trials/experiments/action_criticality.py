"""
Oracle action-criticality study.

Rolls out a trained DQN oracle greedily and measures, for a topology grid:
  (A) how many distinct actions the oracle actually uses + their frequency
  (B) the survival cost of restricting the controller to a curated action subset

Reuses the repo's Evaluator to build the (heuristic-wrapped) eval env, and the DQN QNetwork as oracle.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch as th

# allow running from anywhere, put the repo root in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.eval import Evaluator
from alg.dqn.agent import QNetwork


class _VecSpec:
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.num_envs = 1


def build(ckpt, device):
    run = th.load(ckpt, map_location=device, weights_only=False)
    args = run["args"]
    assert args.action_type == "topology", "criticality study is for discrete topology grids"
    evaluator = Evaluator(args, None, device)
    net = QNetwork(_VecSpec(evaluator.env), args).to(device)
    net.load_state_dict(run["qnet"])
    net.eval()
    n_actions = int(evaluator.env.action_space.n)
    return evaluator, net, n_actions, args


def _q(net, obs, device):
    with th.no_grad():
        return net(th.tensor(obs, dtype=th.float32, device=device)).cpu().numpy()


def run_episode(evaluator, net, device, seed, allowed=None, record=None):
    """Greedy episode from a fixed seed. Returns survival."""
    env, max_steps = evaluator.env, evaluator.max_steps
    allowed = None if allowed is None else np.asarray(sorted(allowed), dtype=np.int64)
    obs, info = env.reset(seed=seed)
    while True:
        q = _q(net, obs, device)
        order = np.argsort(q)[::-1]
        if record is not None:
            freq, gaps = record
            greedy = int(order[0])
            freq[greedy] += 1
            gaps[greedy].append(float(q[order[0]] - q[order[1]]) if len(q) > 1 else 0.0)
        a = int(order[0]) if allowed is None else int(allowed[np.argmax(q[allowed])])
        obs, _, _, _, info = env.step(np.int64(a))
        if "episode" in info:
            return env.init_env.nb_time_step / max_steps


def collect_full(evaluator, net, device, seeds):
    """Unmasked greedy over fixed seeds; aggregate per-action stats + baseline survival."""
    freq, gaps = Counter(), defaultdict(list)
    survivals = [run_episode(evaluator, net, device, s, allowed=None, record=(freq, gaps)) for s in seeds]
    per_action = {a: dict(freq=int(freq[a]), mean_gap=float(np.mean(g)), max_gap=float(np.max(g)), sum_gap=float(np.sum(g))) for a, g in gaps.items()}
    return per_action, survivals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=5, help="seeded episodes per condition (same seeds across all)")
    p.add_argument("--seed0", type=int, default=1000, help="first seed; episodes use seed0..seed0+episodes-1")
    p.add_argument("--ks", type=int, nargs="+", default=None, help="curated set sizes K to test")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    cli = p.parse_args()
    device = th.device(cli.device)
    seeds = list(range(cli.seed0, cli.seed0 + cli.episodes))

    evaluator, net, n_actions, args = build(cli.ckpt, device)
    print(f"[crit] grid={args.env_id} difficulty={args.difficulty} n_actions={n_actions} "
          f"use_heuristic={args.use_heuristic} seeds={seeds} ckpt={cli.ckpt.split('/')[-1]}")

    per_action, base_surv = collect_full(evaluator, net, device, seeds)
    steps = sum(s["freq"] for s in per_action.values())
    used = sorted(per_action, key=lambda a: per_action[a]["freq"], reverse=True)
    base = float(np.mean(base_surv))
    print(f"\n[A] full oracle baseline survival = {base*100:.2f}%  (per-ep: "
          f"{', '.join(f'{x*100:.1f}' for x in base_surv)})   total steps={steps}")
    print(f"    distinct actions used: {len(used)} / {n_actions}")
    print(f"    {'rank':>4} {'act':>4} {'freq':>7} {'freq%':>6} {'mean_gap':>9} {'max_gap':>9} {'sum_gap':>10}")
    for r, a in enumerate(used):
        s = per_action[a]
        print(f"    {r:>4} {a:>4} {s['freq']:>7} {100*s['freq']/steps:>5.1f}% "
              f"{s['mean_gap']:>9.4f} {s['max_gap']:>9.4f} {s['sum_gap']:>10.3f}")

    by_freq = sorted(per_action, key=lambda a: per_action[a]["freq"], reverse=True)
    by_crit = sorted(per_action, key=lambda a: per_action[a]["max_gap"], reverse=True)  # rare-but-decisive kept
    ks = cli.ks or sorted({k for k in [2, 3, 5, 10, len(used)] if 1 <= k <= len(used)})
    print(f"\n[B] survival vs curated-set size K  ({cli.episodes} seeded eps each; full baseline {base*100:.2f}%)")
    print(f"    {'K':>4} | {'by_frequency':>14} | {'by_criticality':>14}")
    results = {}
    for k in ks:
        sf = [run_episode(evaluator, net, device, s, allowed=by_freq[:k]) for s in seeds]
        sc = [run_episode(evaluator, net, device, s, allowed=by_crit[:k]) for s in seeds]
        results[k] = dict(freq=sf, crit=sc)
        print(f"    {k:>4} | {np.mean(sf)*100:>13.2f}% | {np.mean(sc)*100:>13.2f}%")

    if cli.out:
        json.dump({"grid": args.env_id, "ckpt": cli.ckpt.split('/')[-1], "n_actions": n_actions,
                   "seeds": seeds, "baseline_survival": base, "baseline_per_ep": base_surv,
                   "steps": steps, "distinct_used": len(used),
                   "per_action": {int(a): v for a, v in per_action.items()},
                   "by_freq": [int(a) for a in by_freq], "by_crit": [int(a) for a in by_crit],
                   "ablation": {int(k): v for k, v in results.items()}}, open(cli.out, "w"), indent=2)
        print(f"\n[crit] wrote {cli.out}")


if __name__ == "__main__":
    main()
