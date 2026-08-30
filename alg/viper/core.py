import os
import copy
from time import time

import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from alg.dqn.agent import QNetwork
from alg.ppo.agent import Agent as PPOAgent
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


# Oracle adapter for being able to test both DQN and PPO with the same VIPER code.
class _DQNOracle:
    """Q-network oracle. Importance weight l(s) = max_a Q(s,a) - min_a Q(s,a),
    exactly as in Bastani et al. (2018) -- the Q-values are available directly."""

    name = "dqn"

    def __init__(self, envs, oracle_args, state_dict, device):
        self.net = QNetwork(envs, oracle_args).to(device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @th.no_grad()
    def label_and_weight(self, obs_tensor: th.Tensor):
        q_values = self.net(obs_tensor)
        actions = th.argmax(q_values, dim=1).cpu().numpy()
        weights = (q_values.max(dim=1).values - q_values.min(dim=1).values).cpu().numpy()
        return actions, weights


class _PPOOracle:
    """PPO actor oracle. Since no Q-values, approximate importance weight by top-1/top-2 action-probability gap"""
    name = "ppo"

    def __init__(self, envs, oracle_args, state_dict, device):
        agent = PPOAgent(envs, oracle_args, continuous_actions=False).to(device)
        agent.actor.load_state_dict(state_dict)
        agent.eval()
        self.actor = agent.actor

    @th.no_grad()
    def label_and_weight(self, obs_tensor: th.Tensor):
        probs = th.softmax(self.actor(obs_tensor), dim=1)
        actions = th.argmax(probs, dim=1).cpu().numpy()
        top2 = th.topk(probs, k=2, dim=1).values
        weights = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        return actions, weights


def _build_oracle(record: Dict[str, Any], envs, device, requested_type: str):
    """Pick the oracle adapter from --oracle-type, or auto-detect from the checkpoint contents ('qnet' => DQN, 'actor' => PPO)."""
    if requested_type == "auto":
        if "qnet" in record:
            requested_type = "dqn"
        elif "actor" in record:
            requested_type = "ppo"
        else:
            raise ValueError(
                "Could not auto-detect oracle type: checkpoint has neither 'qnet' nor 'actor'. "
                "Pass --oracle-type explicitly."
            )

    oracle_args = record["args"]
    if requested_type == "dqn":
        if "qnet" not in record:
            raise ValueError("--oracle-type dqn but the checkpoint has no 'qnet' weights.")
        return _DQNOracle(envs, oracle_args, record["qnet"], device)
    if requested_type == "ppo":
        if "actor" not in record:
            raise ValueError("--oracle-type ppo but the checkpoint has no 'actor' weights.")
        return _PPOOracle(envs, oracle_args, record["actor"], device)
    raise ValueError(f"Unknown oracle type: {requested_type!r}")


def _new_tree(args, ccp_alpha=None) -> DecisionTreeClassifier:
    """Build a DecisionTreeClassifier honouring every tree-related config flag."""
    return DecisionTreeClassifier(
        max_depth=args.tree_max_depth,
        ccp_alpha=args.tree_ccp_alpha if ccp_alpha is None else float(ccp_alpha),
        min_samples_leaf=args.tree_min_samples_leaf,
        min_samples_split=args.tree_min_samples_split,
        max_leaf_nodes=args.tree_max_leaf_nodes,
        random_state=args.seed,
    )


def _make_fit_dataset(X: np.ndarray, y: np.ndarray, w: np.ndarray, args, rng_seed: int):
    """
    VIPER resampling (Bastani et al., 2018, Algorithm 3).

    Draw points from aggregated dataset with probability proportional to the importance weights.
    Train student on resample uniformly.
    """
    requested = int(args.viper_resample_size)
    if requested <= 0 or requested >= len(X):
        return X, y, w

    probs = np.asarray(w, dtype=np.float64)
    total = probs.sum()
    probs = probs / total if total > 0 else None

    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(X), size=requested, replace=True, p=probs)

    return X[idx], y[idx], np.ones(requested, dtype=np.float64)


def _safe_stratify_labels(y: np.ndarray):
    """Return ``y`` for a stratified split only when every class has >= 2 samples."""
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) > 1 and counts.min() >= 2:
        return y
    return None


def fit_pruned_viper_tree(X, y, sample_weight, args):
    stratify = _safe_stratify_labels(y)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )

    base_tree = _new_tree(args, ccp_alpha=0.0)
    path = base_tree.cost_complexity_pruning_path(X_train, y_train, sample_weight=w_train)

    candidate_alphas = np.unique(path.ccp_alphas)
    candidate_alphas = candidate_alphas[np.isfinite(candidate_alphas)]
    if len(candidate_alphas) == 0:
        candidate_alphas = np.array([0.0])

    # Always include the user's explicit alpha and an unpruned candidate.
    candidate_alphas = np.unique(
        np.concatenate([candidate_alphas, np.array([0.0, args.tree_ccp_alpha], dtype=np.float64)])
    )

    if len(candidate_alphas) > 25:
        quantiles = np.linspace(0.0, 1.0, 25)
        candidate_alphas = np.unique(np.quantile(candidate_alphas, quantiles))

    best_tree = None
    best_adjusted_score = -np.inf
    best_val_acc = -np.inf
    best_alpha = 0.0

    for alpha in candidate_alphas:
        tree = _new_tree(args, ccp_alpha=float(alpha))
        tree.fit(X_train, y_train, sample_weight=w_train)

        val_acc = accuracy_score(y_val, tree.predict(X_val), sample_weight=w_val)
        adjusted_score = val_acc - 0.0001 * tree.get_n_leaves()

        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_val_acc = val_acc
            best_tree = tree
            best_alpha = float(alpha)

    print(
        f"[VIPER PRUNE SWEEP] selected_alpha={best_alpha:.8f} "
        f"val_acc={best_val_acc:.6f} adjusted_score={best_adjusted_score:.6f} "
        f"depth={best_tree.get_depth()} leaves={best_tree.get_n_leaves()} "
        f"nodes={best_tree.tree_.node_count}"
    )

    return best_tree, best_alpha


def _export_tree_dot(tree, obs_dim, n_actions, out_path, max_depth=None) -> None:
    """Export a decision-tree visualisation in Graphviz DOT format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_kwargs = dict(
        out_file=out_path,
        feature_names=[f"obs_{i}" for i in range(obs_dim)],
        class_names=[str(i) for i in range(n_actions)],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    if max_depth is not None:
        export_kwargs["max_depth"] = max_depth
    export_graphviz(tree, **export_kwargs)


class VIPER:
    """
    VIPER policy extraction (Bastani et al., 2018,
    "Verifiable Reinforcement Learning via Policy Extraction"):
      - a pretrained DQN *or* PPO oracle (set --oracle-type),
      - DAgger-style dataset aggregation with the *student* rolled out,
      - importance weighting l(s) = max_a Q(s,a) - min_a Q(s,a) for a DQN oracle
        (probability-gap proxy for a PPO oracle),
      - a decision-tree student fit on a weight-proportional resample,
      - the returned policy is the iterate with the best evaluated survival.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
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
        oracle = _build_oracle(oracle_record, envs, device, args.oracle_type)

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = int(envs.single_action_space.n)

        tree_dir = os.path.join("viper_trees", run_name)
        if args.viper_export_trees:
            os.makedirs(tree_dir, exist_ok=True)

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
        iter_idx = 0

        print(f"[VIPER] oracle={oracle.name} (from {oracle_ckpt_path})")

        try:
            for iter_idx in range(1, args.viper_iters + 1):
                obs, _ = envs.reset(seed=args.seed + iter_idx)

                collected = 0
                while collected < args.viper_steps_per_iter:
                    obs_tensor = th.as_tensor(obs, dtype=th.float32, device=device)

                    # Oracle greedy labels + VIPER importance weights.
                    oracle_actions, weights = oracle.label_and_weight(obs_tensor)
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

                X_fit, y_fit, w_fit = _make_fit_dataset(X, y, w, args, rng_seed=args.seed + iter_idx)

                selected_alpha = args.tree_ccp_alpha
                if args.viper_prune_sweep:
                    tree, selected_alpha = fit_pruned_viper_tree(X_fit, y_fit, w_fit, args)
                else:
                    tree = _new_tree(args)
                    tree.fit(X_fit, y_fit, sample_weight=w_fit)

                current_policy = DecisionTreePolicy(tree, obs_dim, n_actions, device)

                # Weighted imitation accuracy on all aggregated data (diagnostic only).
                train_pred = tree.predict(X)
                weighted_acc = np.average((train_pred == y).astype(np.float32), weights=w)

                print(
                    f"[VIPER TREE] depth={tree.get_depth()} leaves={tree.get_n_leaves()} "
                    f"nodes={tree.tree_.node_count} ccp_alpha={selected_alpha:.8f} "
                    f"fit_samples={len(X_fit)} dataset={len(X)}"
                )

                if iter_idx % args.viper_eval_freq == 0:
                    eval_result = evaluator.evaluate(global_step, current_policy)
                    survival = float(eval_result["survival"])

                    if args.viper_export_trees:
                        tree_path = os.path.join(
                            tree_dir,
                            f"viper_iter_{iter_idx:03d}_step_{global_step}_acc_{weighted_acc:.6f}.dot",
                        )
                        _export_tree_dot(tree, obs_dim, n_actions, tree_path)

                    if survival > best_score:
                        best_score = survival
                        best_policy = DecisionTreePolicy(copy.deepcopy(tree), obs_dim, n_actions, device)
                        if args.viper_export_trees:
                            _export_tree_dot(tree, obs_dim, n_actions, os.path.join(tree_dir, "best_tree.dot"))

                    if args.verbose:
                        print(
                            f"[VIPER] iter={iter_idx}/{args.viper_iters} "
                            f"dataset={len(X)} fit_samples={len(X_fit)} "
                            f"weighted_acc={weighted_acc:.6f} "
                            f"survival={survival*100:.3f}% "
                            f"best_survival={best_score*100:.3f}%"
                        )

                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Save best extracted tree (selected by evaluated survival).
            tree_to_save = None if best_policy is None else best_policy.tree
            ckpt.set_record(
                args=args,
                tree=tree_to_save,
                global_step=global_step,
                oracle_run_name=args.oracle_run_name,
                wb_run_name="" if not logger else logger.wb_path,
                last_iter=iter_idx,
                best_score=best_score,
            )
            ckpt.save()
            if logger:
                logger.close()
            envs.close()