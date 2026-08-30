"""
Evaluate a trained RL2Grid checkpoint (.tar) by rolling out the greedy policy
in the evaluation environment and printing survival / per-reward returns.

The official RL2Grid repo has no standalone evaluation entrypoint -- evaluation
happens in-training through env.eval.Evaluator. This script reuses that exact
Evaluator so the numbers match training-time evaluation, and rebuilds the policy
for whichever algorithm produced the checkpoint (read from the stored args.alg).

Usage:
    python eval_ckpt.py --ckpt checkpoint/PPO_bus14_T_0_0__I__...tar --episodes 10
"""
import argparse

import numpy as np
import torch as th

from env.eval import Evaluator, CMDPEvaluator


class _VecSpec:
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.num_envs = 1


def _build_model(run, vec, args, device):
    alg = args.alg.upper()
    continuous = args.action_type == "redispatch"

    if alg == "DQN":
        from alg.dqn.agent import QNetwork
        model = QNetwork(vec, args).to(device)
        model.load_state_dict(run["qnet"])
        model.eval()
        return model

    if alg in ("PPO", "SREINFORCE", "DAGGER"):
        from alg.ppo.agent import Agent
        # DAGGER is topology-only (discrete); PPO/SREINFORCE follow the action type.
        model = Agent(vec, args, continuous_actions=(continuous and alg != "DAGGER")).to(device)
        model.actor.load_state_dict(run["actor"])
        model.actor.eval()
        return model

    if alg == "LAGRPPO":
        from alg.lagr_ppo.agent import Agent
        model = Agent(vec, args, continuous_actions=continuous).to(device)
        model.actor.load_state_dict(run["actor"])
        model.actor.eval()
        return model

    if alg == "SAC":
        from alg.sac.agent import Actor
        model = Actor(vec, args, continuous_actions=continuous).to(device)
        model.load_state_dict(run["actor"])
        model.eval()
        return model

    if alg == "TD3":
        from alg.td3.agent import Actor
        model = Actor(vec, args).to(device)
        model.load_state_dict(run["actor"])
        model.eval()
        return model

    if alg == "VIPER":
        from alg.viper.agent import DecisionTreePolicy
        tree = run.get("tree")
        if tree is None:
            if "actor" in run:
                raise ValueError(
                    "VIPER checkpoint stores an 'actor' (old/incompatible VIPER format), not a "
                    "decision tree. Re-run VIPER with the current implementation to get a 'tree'."
                )
            raise ValueError(
                "VIPER checkpoint has no stored tree (best_policy was never set during the run)."
            )
        obs_dim = int(np.prod(vec.single_observation_space.shape))
        n_actions = int(vec.single_action_space.n)
        return DecisionTreePolicy(tree, obs_dim, n_actions, device)

    raise ValueError(f"Unsupported algorithm in checkpoint: {args.alg!r}")


def main():
    p = argparse.ArgumentParser(description="Evaluate an RL2Grid checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to a .tar checkpoint saved by RL2Grid.")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    p.add_argument("--device", type=str, default="cpu", help="Torch device (e.g. cpu, cuda).")
    cli = p.parse_args()

    device = th.device(cli.device)
    run = th.load(cli.ckpt, map_location=device, weights_only=False)

    if "args" not in run or not hasattr(run["args"], "alg"):
        raise ValueError("Checkpoint is missing args/args.alg; cannot determine the algorithm.")
    args = run["args"]

    eval_cls = CMDPEvaluator if getattr(args, "constraints_type", 0) != 0 else Evaluator
    evaluator = eval_cls(args, None, device)

    vec = _VecSpec(evaluator.env)
    model = _build_model(run, vec, args, device)

    print(
        f"[eval_ckpt] alg={args.alg} env={args.env_id} action_type={args.action_type} "
        f"difficulty={getattr(args, 'difficulty', '?')} episodes={cli.episodes} device={device}"
    )
    evaluator.evaluate(0, model, eval_ep=cli.episodes)


if __name__ == "__main__":
    main()
