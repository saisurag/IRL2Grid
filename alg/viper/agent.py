import numpy as np
import torch as th
import torch.nn as nn


class DecisionTreePolicy(nn.Module):

    def __init__(self, tree, obs_dim: int, n_actions: int, device: th.device):
        super().__init__()
        self.tree = tree
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

    def _to_numpy_2d(self, x):
        if isinstance(x, th.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(x.shape[0], -1)

        return x

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_np = self._to_numpy_2d(x)
        pred = self.tree.predict(x_np)

        logits = np.zeros((len(pred), self.n_actions), dtype=np.float32)
        logits[np.arange(len(pred)), pred] = 1.0

        if isinstance(x, th.Tensor):
            return th.tensor(logits, dtype=th.float32, device=x.device)
        return th.tensor(logits, dtype=th.float32, device=self.device)

    def get_action(self, x: th.Tensor) -> th.Tensor:
        x_np = self._to_numpy_2d(x)
        pred = self.tree.predict(x_np)
        if isinstance(x, th.Tensor):
            return th.as_tensor(pred, dtype=th.long, device=x.device)
        return th.as_tensor(pred, dtype=th.long, device=self.device)

    def get_eval_action(self, x: th.Tensor) -> th.Tensor:
        x_np = self._to_numpy_2d(x)
        pred = self.tree.predict(x_np)
        device = x.device if isinstance(x, th.Tensor) else self.device
        return th.tensor(pred[0], dtype=th.long, device=device)