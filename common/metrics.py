# Evaluation metrics for interpretable power-grid controllers.
import numpy as np
import torch as th


# Interpretability — structural metrics of a fitted decision tree
def _as_sklearn_tree(tree):
    """Accept a sklearn DecisionTreeClassifier or a wrapper exposing `.tree`."""
    return getattr(tree, "tree", tree)


def _node_depths(t):
    """Depth of every node (root = 0) for a sklearn `tree_`."""
    depth = np.zeros(t.node_count, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        if t.children_left[node] != t.children_right[node]:   # internal
            stack.append((t.children_left[node], d + 1))
            stack.append((t.children_right[node], d + 1))
    return depth


def tree_interpretability(tree):
    """Structural interpretability metrics of a fitted sklearn decision tree."""
    clf = _as_sklearn_tree(tree)
    t = clf.tree_
    is_leaf = t.children_left == -1
    n_leaves = int(is_leaf.sum())
    n_decision = int((~is_leaf).sum())
    depths = _node_depths(t)

    # sample-weighted mean leaf depth = expected #conditions checked per decision
    leaf_samples = t.n_node_samples[is_leaf].astype(np.float64)
    leaf_depths = depths[is_leaf].astype(np.float64)
    total = leaf_samples.sum()
    mean_path = float((leaf_depths * leaf_samples).sum() / total) if total else 0.0

    feats_used = np.unique(t.feature[t.feature >= 0])
    # distinct predicted actions across leaves
    classes = getattr(clf, "classes_", None)
    leaf_pred = np.argmax(t.value[is_leaf].reshape(n_leaves, -1), axis=1)
    if classes is not None:
        leaf_pred = np.asarray(classes)[leaf_pred]
    n_actions_used = int(len(np.unique(leaf_pred)))

    return dict(
        n_leaves=n_leaves,
        n_decision_nodes=n_decision,
        n_nodes=int(t.node_count),
        max_depth=int(depths.max()),
        mean_path_length=mean_path,
        n_features_used=int(len(feats_used)),
        features_used=feats_used.tolist(),
        n_actions_used=n_actions_used,
    )


def action_distribution(actions, n_actions=None):
    """Coverage and (normalised) entropy of an action sequence."""
    actions = np.asarray(actions).ravel()
    vals, counts = np.unique(actions, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    k = n_actions or len(vals)
    norm_ent = float(ent / np.log(k)) if k > 1 else 0.0
    return dict(n_distinct=int(len(vals)), entropy=ent, norm_entropy=norm_ent,
                most_common=int(vals[np.argmax(counts)]), top_freq=float(p.max()))


# Fidelity to the oracle (plain + criticality-weighted)
def fidelity(student_actions, oracle_actions, weights=None):
    """
    Fraction of states where the student matches the oracle.
    """
    s = np.asarray(student_actions).ravel()
    o = np.asarray(oracle_actions).ravel()
    match = (s == o).astype(np.float64)
    out = dict(fidelity=float(match.mean()), n=int(len(match)))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        out["weighted_fidelity"] = float((w * match).sum() / w.sum()) if w.sum() else float("nan")
    return out


def action_flip_rate(predict_fn, states, sigma=0.05, n_perturb=5, rng_seed=0,
                     relative=True):
    """
    Fraction of (state, perturbation) pairs whose predicted action differs from the action on the clean state.
    """
    X = np.asarray(states, dtype=np.float64)
    base = np.asarray(predict_fn(X)).ravel()
    scale = (X.std(axis=0) if relative else np.ones(X.shape[1])) * sigma
    rng = np.random.default_rng(rng_seed)
    flips = 0
    for _ in range(n_perturb):
        Xp = X + rng.normal(0.0, 1.0, X.shape) * scale
        flips += (np.asarray(predict_fn(Xp)).ravel() != base).sum()
    return float(flips / (len(base) * n_perturb))


def _grid_obs(env):
    """Best-effort access to the underlying grid2op observation (for rho)."""
    ie = getattr(env, "init_env", None)
    if ie is None:
        return None
    o = getattr(ie, "current_obs", None)
    if o is not None:
        return o
    try:
        return ie.get_obs()
    except Exception:
        return None


def rollout_metrics(policy, env, max_steps, seeds, action_map=None, device="cpu"):
    """
    Roll `policy` (exposes get_eval_action) over fixed `seeds` and collect survival + safety sub-metrics.
    """
    device = th.device(device)
    survivals, max_rhos, overload_fracs, costs = [], [], [], []
    actions_taken = []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        ep_rho_max, ep_over, ep_len, ep_cost = 0.0, 0, 0, 0.0
        while True:
            a = policy.get_eval_action(th.as_tensor(obs, dtype=th.float32, device=device))
            a = int(np.asarray(a.detach().cpu().numpy() if th.is_tensor(a) else a).ravel()[0])
            actions_taken.append(a)
            env_a = action_map(a) if action_map is not None else a
            obs, _, _, _, info = env.step(np.int64(env_a))
            if "cost" in info:                                   # CMDP constraint cost
                ep_cost += float(info["cost"])
            gobs = _grid_obs(env)
            rho = getattr(gobs, "rho", None)
            if rho is not None and len(rho):
                ep_rho_max = max(ep_rho_max, float(np.max(rho)))
                ep_over += int(np.any(np.asarray(rho) >= 1.0))
            ep_len += 1
            if "episode" in info:
                survivals.append(env.init_env.nb_time_step / max_steps)
                max_rhos.append(ep_rho_max)
                overload_fracs.append(ep_over / max(ep_len, 1))
                costs.append(ep_cost)
                break
    survivals = np.asarray(survivals, dtype=np.float64)
    out = dict(
        survival_mean=float(survivals.mean()),
        survival_std=float(survivals.std()),
        survival_per_ep=survivals.tolist(),
        n_episodes=len(seeds),
    )
    if any(r > 0 for r in max_rhos):
        out["mean_max_rho"] = float(np.mean(max_rhos))          # closeness to thermal limit (1.0 = at limit)
        out["overload_step_frac"] = float(np.mean(overload_fracs))
    if any(c != 0 for c in costs):
        out["mean_constraint_cost"] = float(np.mean(costs))     # CMDP cost return (constrained runs)
    out["action_distribution"] = action_distribution(actions_taken)
    return out


def pareto_frontier(points, x="n_leaves", y="survival_mean", minimize_x=True, maximize_y=True):
    """
    Return the Pareto-efficient subset of `points` (list of dicts) for the
    trade-off between an interpretability scalar `x` (smaller better) and a
    performance scalar `y` (larger better), sorted by `x`.
    """
    pts = sorted(points, key=lambda d: (d[x], -d[y] if maximize_y else d[y]))
    frontier, best_y = [], -np.inf if maximize_y else np.inf
    for d in pts:
        better = d[y] > best_y if maximize_y else d[y] < best_y
        if better:
            frontier.append(d)
            best_y = d[y]
    return frontier


def format_frontier(points, x="n_leaves", y="survival_mean", label="method"):
    """Pretty-print a frontier table; marks Pareto-efficient rows with '*'."""
    front = {id(d) for d in pareto_frontier(points, x=x, y=y)}
    lines = [f"{'':1} {label:>24} {x:>10} {y:>12}"]
    for d in sorted(points, key=lambda d: d[x]):
        mark = "*" if id(d) in front else " "
        lines.append(f"{mark} {str(d.get(label, '')):>24} {d[x]:>10} {d[y]:>12.4f}")
    lines.append("(* = Pareto-efficient)")
    return "\n".join(lines)
