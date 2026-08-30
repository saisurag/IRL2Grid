import argparse as ap
from multiprocessing import Manager

import numpy as np
import torch as th
import gymnasium as gym

from grid2op.gym_compat import DiscreteActSpace

from alg.ppo.agent import Agent as PPOAgent
from common.logger import Logger
from env.eval import Evaluator
from env.utils import auxiliary_make_env


def build_vector_env(args):
    main_gym_env, main_g2o_env = auxiliary_make_env(args, False)

    with Manager() as manager:
        if args.action_type == "topology":
            print("Initializing the distributed action 'mapper'... (takes a while with big action spaces)")
            shared_action_space = manager.list(main_gym_env.action_space.converter.all_actions)

        def make_vec_subprocess(idx):
            if args.action_type == "topology":
                action_space = DiscreteActSpace(
                    main_g2o_env.action_space,
                    action_list=shared_action_space
                )
                return auxiliary_make_env(
                    args,
                    resume_run=False,
                    idx=idx,
                    action_space=action_space
                )[0]

            return auxiliary_make_env(args, resume_run=False, idx=idx)[0]

        env_fns = [lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)]

        if args.n_envs == 1:
            envs = gym.vector.SyncVectorEnv(env_fns)
        else:
            envs = gym.vector.AsyncVectorEnv(env_fns)

    return envs


def main():
    parser = ap.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--eval-ep", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    args_cli = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() and args_cli.cuda else "cpu")

    run = th.load(args_cli.ckpt, map_location=device, weights_only=False)
    args = run["args"]

    # keep eval simple / stable
    args.n_envs = 1

    envs = build_vector_env(args)
    logger = Logger("eval_run", args)
    evaluator = Evaluator(args, logger, device)

    model = PPOAgent(envs, args, continuous_actions=False).to(device)
    model.actor.load_state_dict(run["actor"])

    if "critic" in run:
        try:
            model.critic.load_state_dict(run["critic"])
        except Exception:
            pass

    model.actor.eval()
    if hasattr(model, "critic"):
        try:
            model.critic.eval()
        except Exception:
            pass

    evaluator.evaluate(0, model, eval_ep=args_cli.eval_ep)


if __name__ == "__main__":
    main()