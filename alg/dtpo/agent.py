import numpy as np
import torch as th
import torch.nn as nn

from common.utils import Linear, th_act_fns


class Critic(nn.Module):
    """NN value function used only to improve DTPO's optimization (GAE). It is not part of the interpretable policy"""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_str, act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]
        cl = args.critic_layers
        layers = [Linear(obs_dim, cl[0], act_str), act_fn]
        for i in range(1, len(cl)):
            layers += [Linear(cl[i - 1], cl[i], act_str), act_fn]
        layers.append(Linear(cl[-1], 1, "linear"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).squeeze(-1)


class RegressionTreePolicy:
    """
    DTPO policy: a (multi-output) regression tree mapping observations to action probabilities. 
    `tree=None` is the DTPO initialization -- a single max-entropy leaf (uniform over actions). Stochastic during training (sample).
    Deterministic at evaluation (argmax leaf).
    """

    def __init__(self, n_actions: int, device, tree=None, action_map=None):
        self.n_actions = int(n_actions)
        self.device = device
        self.tree = tree
        self.action_map = None if action_map is None else np.asarray(action_map, dtype=np.int64)

    def proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.tree is None:
            return np.full((X.shape[0], self.n_actions), 1.0 / self.n_actions, dtype=np.float64)
        p = np.asarray(self.tree.predict(X), dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, self.n_actions)
        p = np.clip(p, 1e-8, None)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, X, rng) -> np.ndarray:
        """Vectorized categorical sampling from the per-row action distribution."""
        p = self.proba(X)
        c = np.cumsum(p, axis=1)
        r = rng.random((p.shape[0], 1))
        return (r < c).argmax(axis=1)

    def action_proba(self, X, actions) -> np.ndarray:
        p = self.proba(X)
        return p[np.arange(len(p)), np.asarray(actions, dtype=np.int64)]

    def get_eval_action(self, x: th.Tensor) -> th.Tensor:
        """Deterministic (argmax-leaf) action for one observation. Returns the full env action when a curated action_map is set."""
        X = x.detach().cpu().numpy() if th.is_tensor(x) else np.asarray(x)
        a = int(np.argmax(self.proba(X.reshape(1, -1))[0]))
        if self.action_map is not None:
            a = int(self.action_map[a])
        return th.tensor(a, dtype=th.long)
