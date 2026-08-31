# Soft-target policy distillation into hard CART trees

import os

import numpy as np
import torch as th
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from alg.dtpo.agent import RegressionTreePolicy
from alg.viper.agent import DecisionTreePolicy


def _resolve_ckpt(name: str) -> str:
    return name if name.endswith(".tar") else os.path.join("checkpoint", name + ".tar")


def load_oracle_score_fn(ckpt_name, vec, device, oracle_type="auto"):
    """
    Load an oracle checkpoint and return a `score_fn(obs) -> vector` whose argmax is the chosen action and top1-top2 gap is the criticality (Q-values for a DQN oracle, actor logits for PPO).
    """
    from alg.dqn.agent import QNetwork
    from alg.ppo.agent import Agent as PPOAgent
    from common.action_reduction import dqn_score_fn, ppo_score_fn

    rec = th.load(_resolve_ckpt(ckpt_name), map_location=device, weights_only=False)
    oa = rec["args"]
    t = oracle_type
    if t == "auto":
        t = "dqn" if "qnet" in rec else "ppo" if "actor" in rec else None
        if t is None:
            raise ValueError("Cannot auto-detect oracle type (no 'qnet'/'actor' in checkpoint).")
    if t == "dqn":
        net = QNetwork(vec, oa).to(device)
        net.load_state_dict(rec["qnet"])
        net.eval()
        return dqn_score_fn(net, device)
    agent = PPOAgent(vec, oa, continuous_actions=False).to(device)
    agent.actor.load_state_dict(rec["actor"])
    agent.eval()
    return ppo_score_fn(agent, device)


def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def collect_oracle_dataset(score_fn, env, seeds, *, student=None, action_reducer=None, steps_cap=None):
    """
    DAgger-style aggregation over fixed seeds. Roll out the `student` if given, else the oracle; at every visited state record the oracle's score vector, its argmax action, and criticality = top1-top2 gap.
    """
    O, S, A, W = [], [], [], []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        while True:
            v = np.asarray(score_fn(obs)).ravel()
            if action_reducer is not None:
                v = action_reducer.restrict(v)
            order = np.argsort(v)[::-1]
            a = int(order[0])
            gap = float(v[order[0]] - v[order[1]]) if v.size > 1 else 0.0
            O.append(np.asarray(obs, dtype=np.float32))
            S.append(v.astype(np.float64))
            A.append(a)
            W.append(max(gap, 1e-8))

            # execute student (curated idx -> full env action) or the oracle
            if student is not None:
                ca = int(student.get_eval_action(th.as_tensor(obs, dtype=th.float32)).item())
                env_a = action_reducer.to_full(ca) if action_reducer is not None else ca
            else:
                env_a = action_reducer.to_full(a) if action_reducer is not None else a
            obs, _, _, _, info = env.step(np.int64(env_a))

            if "episode" in info:
                break
            if steps_cap and len(O) >= steps_cap:
                return np.asarray(O), np.asarray(S), np.asarray(A), np.asarray(W)
    return np.asarray(O), np.asarray(S), np.asarray(A), np.asarray(W)


def distill_from_dataset(obs, scores, actions, weights, *, n_actions, device="cpu",
                         mode="soft", temperature=1.0, max_leaf_nodes=16, max_depth=None,
                         min_samples_leaf=1, ccp_alpha=0.0, seed=0):
    """
    Fit a CART tree to a pre-collected (obs, oracle-scores, actions, weights) dataset and wrap it as a policy.
    """
    tree_kw = dict(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                   min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, random_state=seed)
    weights = np.asarray(weights, dtype=np.float64)

    if mode == "soft":
        Y = softmax(scores, temperature=temperature)            # [N, n_actions] target distributions
        tree = DecisionTreeRegressor(**tree_kw).fit(obs, Y, sample_weight=weights)
        policy = RegressionTreePolicy(n_actions, th.device(device), tree=tree)
        pred = np.argmax(np.atleast_2d(tree.predict(obs)), axis=1)
    elif mode == "hard":
        tree = DecisionTreeClassifier(**tree_kw).fit(obs, actions, sample_weight=weights)
        policy = DecisionTreePolicy(tree, obs.shape[1], n_actions, th.device(device))
        pred = tree.predict(obs)
    else:
        raise ValueError(f"unknown distillation mode: {mode!r}")

    match = (np.asarray(pred) == np.asarray(actions)).astype(np.float64)
    info = dict(mode=mode, n_samples=int(len(obs)), n_actions=int(n_actions),
                n_leaves=int(tree.get_n_leaves()), max_depth=int(tree.get_depth()),
                fidelity=float(match.mean()),
                weighted_fidelity=float((weights * match).sum() / weights.sum()))
    return policy, info, tree


def distill(score_fn, env, seeds, *, n_actions, device="cpu", mode="soft",
            temperature=1.0, action_reducer=None, student=None, steps_cap=None,
            dagger_rounds=0, dagger_steps_cap=None, **tree_kw):
    """
    End-to-end: collect an oracle dataset over `seeds`, then distil a tree. Returns (policy, info). The policy operates in curated action space when `action_reducer` is given (map to env actions with action_reducer.to_full).
    """
    O, S, A, W = collect_oracle_dataset(score_fn, env, seeds, student=student,
                                        action_reducer=action_reducer, steps_cap=steps_cap)
    k = action_reducer.n_curated if action_reducer is not None else int(n_actions)
    policy, info, _ = distill_from_dataset(O, S, A, W, n_actions=k, device=device,
                                           mode=mode, temperature=temperature, **tree_kw)

    round_sizes = [int(len(O))]                                # [oracle-only, +round1, ...]
    for d in range(int(dagger_rounds)):
        # roll out the current student; collect_oracle_dataset labels with the oracle
        o, s, a, w = collect_oracle_dataset(score_fn, env, seeds, student=policy,
                                            action_reducer=action_reducer,
                                            steps_cap=(dagger_steps_cap or steps_cap))
        if len(o):
            O = np.concatenate([O, o]); S = np.concatenate([S, s])
            A = np.concatenate([A, a]); W = np.concatenate([W, w])
        policy, info, _ = distill_from_dataset(O, S, A, W, n_actions=k, device=device,
                                               mode=mode, temperature=temperature, **tree_kw)
        round_sizes.append(int(len(O)))

    info["curated"] = action_reducer is not None
    info["dagger_rounds"] = int(dagger_rounds)
    if dagger_rounds:
        info["dagger_dataset_sizes"] = round_sizes
    return policy, info
