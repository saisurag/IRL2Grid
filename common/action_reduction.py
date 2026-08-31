"""
Action-support reduction for interpretable topology controllers.
"""

from collections import Counter, defaultdict

import json

import numpy as np
import torch as th


# Oracle scoring helpers (argmax = action, top1-top2 gap = criticality)
def dqn_score_fn(qnet, device):
    """Q-values from a DQN QNetwork; gap is the true Q-advantage."""
    def f(obs):
        with th.no_grad():
            return qnet(th.as_tensor(obs, dtype=th.float32, device=device)).cpu().numpy()
    return f


def ppo_score_fn(agent, device):
    """Logits from a PPO actor; gap is the (pre-softmax) confidence margin."""
    def f(obs):
        with th.no_grad():
            return agent.actor(th.as_tensor(obs, dtype=th.float32, device=device)).cpu().numpy()
    return f


# Statistics collection + curation
def collect_action_stats(score_fn, env, max_steps, seeds):
    """
    Greedy rollout of the oracle over fixed `seeds`; record per-action frequency
    and Q-advantage gap (top1 - top2 of the score) for every decision step.

    Returns (stats, survivals) where stats[a] = {freq, mean_gap, max_gap, sum_gap}.
    """
    freq, gaps, survivals = Counter(), defaultdict(list), []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        while True:
            v = np.asarray(score_fn(obs)).ravel()
            order = np.argsort(v)[::-1]
            a = int(order[0])
            freq[a] += 1
            gaps[a].append(float(v[order[0]] - v[order[1]]) if v.size > 1 else 0.0)
            obs, _, _, _, info = env.step(np.int64(a))
            if "episode" in info:
                survivals.append(env.init_env.nb_time_step / max_steps)
                break
    stats = {int(a): dict(freq=int(freq[a]),
                          mean_gap=float(np.mean(g)),
                          max_gap=float(np.max(g)),
                          sum_gap=float(np.sum(g)))
             for a, g in gaps.items()}
    return stats, survivals


def curate(stats, n_full, *, strategy="coverage+criticality", coverage=0.999,
           idle_action=0, crit_quantile=0.9, max_actions=None, min_actions=1):
    """
    Decide which full-action indices to retain, given per-action `stats`.

    Returns a sorted list of retained full-action indices.
    """
    used = [a for a in stats if stats[a]["freq"] > 0]
    total = sum(stats[a]["freq"] for a in used) or 1

    if strategy == "full":
        return list(range(int(n_full)))
    if strategy == "all_used":
        kept = set(used)
    else:
        kept = set()
        if strategy in ("frequency", "coverage+criticality"):
            cum = 0
            for a in sorted(used, key=lambda a: stats[a]["freq"], reverse=True):
                kept.add(a)
                cum += stats[a]["freq"]
                if cum / total >= coverage:
                    break
        if strategy in ("criticality", "coverage+criticality"):
            maxgaps = np.array([stats[a]["max_gap"] for a in used], dtype=np.float64)
            thr = float(np.quantile(maxgaps, crit_quantile)) if maxgaps.size else 0.0
            crit = {a for a in used if stats[a]["max_gap"] >= thr}
            kept = crit if strategy == "criticality" else (kept | crit)
        if strategy == "topk":
            k = max_actions or len(used)
            kept = set(sorted(used, key=lambda a: stats[a]["freq"], reverse=True)[:k])

    # always retain idle (do-nothing)
    if idle_action is not None and 0 <= idle_action < n_full:
        kept.add(int(idle_action))

    # cap size, but never evict idle or the most critical actions
    if max_actions and len(kept) > max_actions:
        prio = sorted(kept, key=lambda a: (a == idle_action, stats.get(a, {}).get("max_gap", 0.0)),
                      reverse=True)
        kept = set(prio[:max_actions])
        if idle_action is not None and 0 <= idle_action < n_full:
            kept.add(int(idle_action))

    # floor on size (pad with most frequent)
    if len(kept) < min_actions:
        for a in sorted(used, key=lambda a: stats[a]["freq"], reverse=True):
            kept.add(a)
            if len(kept) >= min_actions:
                break

    return sorted(int(a) for a in kept)


# ActionReducer: the curated set + index mapping + diagnostics
class ActionReducer:
    """
    Holds a curated subset of a discrete action space and the mapping between the
    student's compact action indices (0..n_curated-1) and the env's full indices.
    """

    def __init__(self, kept, n_full, stats=None, strategy=None, survivals=None, params=None):
        self.kept = sorted(int(a) for a in kept)
        self.n_full = int(n_full)
        self.stats = {int(a): v for a, v in (stats or {}).items()}
        self.strategy = strategy
        self.survivals = list(survivals) if survivals is not None else None
        self.params = params or {}
        self.curated_to_full = np.asarray(self.kept, dtype=np.int64)
        self.full_to_curated = {a: i for i, a in enumerate(self.kept)}

    @property
    def n_curated(self):
        return len(self.kept)

    @property
    def dropped(self):
        return [a for a in range(self.n_full) if a not in self.full_to_curated]

    def to_full(self, curated_idx):
        return int(self.curated_to_full[int(curated_idx)])

    def to_curated(self, full_idx):
        return self.full_to_curated.get(int(full_idx))

    def restrict(self, full_vec):
        """full-width scores (..., n_full) -> curated (..., n_curated)."""
        return np.asarray(full_vec)[..., self.curated_to_full]

    def project_actions(self, full_actions):
        """Map env actions to curated indices; dropped actions -> -1."""
        return np.array([self.full_to_curated.get(int(a), -1) for a in np.ravel(full_actions)])

    def coverage(self):
        """Fraction of observed oracle decisions covered by the kept set."""
        if not self.stats:
            return None
        total = sum(s["freq"] for s in self.stats.values()) or 1
        return sum(self.stats.get(a, {}).get("freq", 0) for a in self.kept) / total

    def risky_drops(self, top=5):
        """Dropped actions ranked by criticality (max Q-gap) -- the safety check."""
        d = [(a, self.stats[a]["max_gap"]) for a in self.dropped if a in self.stats]
        return sorted(d, key=lambda x: x[1], reverse=True)[:top]

    def summary(self):
        return dict(strategy=self.strategy, n_full=self.n_full, n_curated=self.n_curated,
                    kept=self.kept, coverage=self.coverage(), risky_drops=self.risky_drops())

    def log(self):
        cov = self.coverage()
        print(f"[ActionReducer] strategy={self.strategy} kept {self.n_curated}/{self.n_full} actions "
              f"(coverage={cov*100:.3f}%)" if cov is not None
              else f"[ActionReducer] strategy={self.strategy} kept {self.n_curated}/{self.n_full} actions")
        print(f"    kept indices: {self.kept}")
        risky = self.risky_drops()
        if risky:
            print(f"    dropped actions by criticality (max_gap): "
                  + ", ".join(f"a{a}:{g:.2f}" for a, g in risky))

    def save(self, path):
        json.dump(dict(kept=self.kept, n_full=self.n_full, strategy=self.strategy,
                       params=self.params, survivals=self.survivals,
                       stats=self.stats), open(path, "w"), indent=2)
        return path

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        stats = {int(a): v for a, v in d.get("stats", {}).items()}
        return cls(d["kept"], d["n_full"], stats=stats, strategy=d.get("strategy"),
                   survivals=d.get("survivals"), params=d.get("params"))

    @classmethod
    def from_oracle(cls, score_fn, env, max_steps, seeds, *, n_full=None, **curate_kw):
        stats, survivals = collect_action_stats(score_fn, env, max_steps, seeds)
        n_full = int(n_full if n_full is not None else env.action_space.n)
        kept = curate(stats, n_full, **curate_kw)
        return cls(kept, n_full, stats=stats, strategy=curate_kw.get("strategy", "coverage+criticality"),
                   survivals=survivals, params=dict(seeds=list(map(int, seeds)), **curate_kw))
