import os
import copy
from time import time

import numpy as np
import torch as th
from sklearn.tree import DecisionTreeClassifier

from alg.dqn.agent import QNetwork
from .agent import DecisionTreePolicy
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator


def _resolve_checkpoint_path(run_name: str) -> str:
    if run_name.endswith(".tar"):
        return run_name
    return os.path.join("checkpoint", run_name + ".tar")


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        return obs.reshape(1, -1)
    return obs.reshape(obs.shape[0], -1)


class VIPER:
    """
    True VIPER-style policy extraction using:
    - a pretrained DQN oracle
    - DAgger-style dataset aggregation
    - Q-gap-based importance weighting
    - a decision tree student
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        # VIPER is simplest as a fresh run because the aggregate dataset is not checkpointed
        if ckpt.resumed:
            raise NotImplementedError(
                "VIPER resume is not implemented in this version because the aggregate dataset "
                "is not stored in the checkpoint."
            )

        args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # ---- load oracle checkpoint ----
        oracle_ckpt_path = _resolve_checkpoint_path(args.oracle_run_name)
        oracle_record = th.load(oracle_ckpt_path, map_location=device, weights_only=False)

        oracle_args = oracle_record["args"]
        oracle = QNetwork(envs, oracle_args).to(device)
        oracle.load_state_dict(oracle_record["qnet"])
        oracle.eval()

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = int(envs.single_action_space.n)

        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        # Aggregated dataset
        dataset_states = []
        dataset_actions = []
        dataset_weights = []

        current_policy = None
        best_policy = None
        best_score = -np.inf
        global_step = 0

        try:
            for iter_idx in range(1, args.viper_iters + 1):
                obs, _ = envs.reset(seed=args.seed + iter_idx)

                collected = 0
                while collected < args.viper_steps_per_iter:
                    obs_tensor = th.as_tensor(obs, dtype=th.float32, device=device)

                    # Oracle labels + Q-based VIPER weights
                    with th.no_grad():
                        q_values = oracle(obs_tensor)
                        oracle_actions = th.argmax(q_values, dim=1).cpu().numpy()

                        # VIPER-style importance: bigger gap => more critical state
                        q_max = q_values.max(dim=1).values
                        q_min = q_values.min(dim=1).values
                        weights = (q_max - q_min).detach().cpu().numpy()
                        weights = np.maximum(weights, 1e-8)

                    dataset_states.append(_flatten_obs(obs).copy())
                    dataset_actions.append(oracle_actions.copy())
                    dataset_weights.append(weights.copy())

                    # DAgger/VIPER rollout: execute current student if available, otherwise oracle
                    if current_policy is None:
                        actions = oracle_actions
                    else:
                        with th.no_grad():
                            actions = current_policy.get_action(obs_tensor).cpu().numpy()

                    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                    obs = next_obs
                    collected += envs.num_envs
                    global_step += envs.num_envs

                    if (time() - start_time) / 60 >= args.time_limit:
                        break

                X = np.concatenate(dataset_states, axis=0)
                y = np.concatenate(dataset_actions, axis=0)
                w = np.concatenate(dataset_weights, axis=0)

                tree = DecisionTreeClassifier(
                    max_depth=args.tree_max_depth,
                    min_samples_leaf=args.tree_min_samples_leaf,
                    random_state=args.seed + iter_idx
                )
                tree.fit(X, y, sample_weight=w)

                sample_size = len(X)

                current_policy = DecisionTreePolicy(tree, obs_dim, n_actions, device)

                # Proxy model-selection score:
                # weighted imitation accuracy on the aggregated dataset
                train_pred = tree.predict(X)
                score = np.average((train_pred == y).astype(np.float32), weights=w)

                if score > best_score:
                    best_score = score
                    best_policy = DecisionTreePolicy(copy.deepcopy(tree), obs_dim, n_actions, device)

                if iter_idx % args.viper_eval_freq == 0:
                    evaluator.evaluate(global_step, current_policy)
                    if args.verbose:
                        print(
                            f"[VIPER] iter={iter_idx}/{args.viper_iters} "
                            f"dataset={len(X)} resample={sample_size} "
                            f"weighted_acc={score:.6f}"
                        )

                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Save best extracted tree
            tree_to_save = None if best_policy is None else best_policy.tree
            ckpt.set_record(
                args=args,
                tree=tree_to_save,
                global_step=global_step,
                oracle_run_name=args.oracle_run_name,
                wb_run_name="" if not logger else logger.wb_path,
                last_iter=iter_idx if "iter_idx" in locals() else 0,
                best_score=best_score,
            )
            ckpt.save()
            if logger:
                logger.close()
            envs.close()