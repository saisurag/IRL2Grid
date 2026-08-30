import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_env(env_id: str = "bus14", action_type: str = "topology", seed: int = 0):
    """
    Replace this with your RL2Grid environment constructor.

    Expected interface:
      env.reset() -> obs OR (obs, info)
      env.step(action) -> obs, reward, done, info
    """
    raise NotImplementedError(
        "Edit make_env() to return your RL2Grid environment."
    )


def load_oracle(oracle_type: str, oracle_ckpt: str, device: str = "cpu"):
    """
    Replace this with your PPO/DQN checkpoint loader.

    Return any object you want. It will be passed to oracle_predict().
    """
    raise NotImplementedError(
        "Edit load_oracle() to load your PPO or DQN checkpoint."
    )


def oracle_predict(oracle, obs: np.ndarray) -> int:
    """
    Replace this with your oracle action-selection function.

    Input:
      obs: flat numpy array
    Output:
      integer discrete action
    """
    raise NotImplementedError(
        "Edit oracle_predict() so it returns an integer action."
    )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs) -> np.ndarray:
    """
    Robustly flatten most observation formats into 1D float32 numpy.
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(obs, "to_vect"):
        obs = obs.to_vect()
    obs = np.asarray(obs, dtype=np.float32)
    return obs.reshape(-1)


def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env, action: int):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    elif len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    else:
        raise ValueError("Unexpected env.step(...) return format.")


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    done: bool


def collect_oracle_dataset(
    oracle,
    env_id: str,
    action_type: str,
    seed: int,
    episodes: int,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out oracle in env and collect (obs, action, reward).
    """
    env = make_env(env_id=env_id, action_type=action_type, seed=seed)

    obs_list = []
    act_list = []
    rew_list = []

    for ep in range(episodes):
        obs = safe_reset(env)
        obs = flatten_obs(obs)

        for _ in range(max_steps):
            action = oracle_predict(oracle, obs)
            next_obs, reward, done, _ = safe_step(env, int(action))
            next_obs = flatten_obs(next_obs)

            obs_list.append(obs.copy())
            act_list.append(int(action))
            rew_list.append(float(reward))

            obs = next_obs
            if done:
                break

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.int64)
    rew_arr = np.asarray(rew_list, dtype=np.float32)
    return obs_arr, act_arr, rew_arr


class PredicateEncoder:
    def __init__(
        self,
        n_quantiles: int = 3,
        max_features: int = 64,
        feature_indices: Optional[List[int]] = None,
    ):
        self.n_quantiles = n_quantiles
        self.max_features = max_features
        self.feature_indices = feature_indices

        self.selected_idx = None
        self.thresholds = None
        self.names = None

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape

        # Pick feature subset with highest variance if too many features.
        if self.feature_indices is not None:
            idx = np.array(self.feature_indices, dtype=np.int64)
        else:
            variances = np.var(X, axis=0)
            topk = min(self.max_features, n_features)
            idx = np.argsort(-variances)[:topk]

        Xs = X[:, idx]

        qs = np.linspace(0.25, 0.75, self.n_quantiles)
        thresholds = []
        names = []

        for local_j, feat_idx in enumerate(idx):
            feat_vals = X[:, feat_idx]
            feat_thresholds = np.quantile(feat_vals, qs).astype(np.float32)

            # deduplicate near-identical thresholds
            uniq = []
            for t in feat_thresholds:
                if len(uniq) == 0 or abs(float(t) - float(uniq[-1])) > 1e-6:
                    uniq.append(float(t))
            feat_thresholds = np.array(uniq, dtype=np.float32)

            thresholds.append(feat_thresholds)

            feat_names = []
            for t in feat_thresholds:
                feat_names.append(f"x[{feat_idx}] > {t:.5f}")
                feat_names.append(f"x[{feat_idx}] <= {t:.5f}")
            names.extend(feat_names)

        self.selected_idx = idx
        self.thresholds = thresholds
        self.names = names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_idx is not None and self.thresholds is not None

        X = np.asarray(X, dtype=np.float32)
        Xs = X[:, self.selected_idx]

        preds = []
        for local_j in range(len(self.selected_idx)):
            col = Xs[:, local_j]
            ts = self.thresholds[local_j]
            for t in ts:
                preds.append((col > t).astype(np.float32))
                preds.append((col <= t).astype(np.float32))

        P = np.stack(preds, axis=1) if len(preds) > 0 else np.zeros((len(X), 0), dtype=np.float32)
        return P

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        data = {
            "selected_idx": self.selected_idx.tolist(),
            "thresholds": [arr.tolist() for arr in self.thresholds],
            "names": self.names,
            "n_quantiles": self.n_quantiles,
            "max_features": self.max_features,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        enc = cls(
            n_quantiles=data["n_quantiles"],
            max_features=data["max_features"],
        )
        enc.selected_idx = np.array(data["selected_idx"], dtype=np.int64)
        enc.thresholds = [np.array(x, dtype=np.float32) for x in data["thresholds"]]
        enc.names = data["names"]
        return enc



class RulePolicy(nn.Module):
    def __init__(self, n_predicates: int, n_actions: int):
        super().__init__()
        self.linear = nn.Linear(n_predicates, n_actions)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.linear(p)

    def predict(self, p_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            p = torch.tensor(p_np, dtype=torch.float32)
            logits = self.forward(p)
            return torch.argmax(logits, dim=1).cpu().numpy()


def split_dataset(X, y, val_frac=0.1, seed=0):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def train_rule_policy(
    P_train: np.ndarray,
    y_train: np.ndarray,
    P_val: np.ndarray,
    y_val: np.ndarray,
    n_actions: int,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    l1_coef: float = 1e-5,
    device: str = "cpu",
):
    model = RulePolicy(P_train.shape[1], n_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0

    Xtr = torch.tensor(P_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xva = torch.tensor(P_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)

    n = len(Xtr)
    steps_per_epoch = math.ceil(n / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0

        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]

            logits = model(xb)
            loss = ce(logits, yb)

            # sparse rules
            l1 = 0.0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + l1_coef * l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_pred = torch.argmax(val_logits, dim=1)
            val_acc = (val_pred == yva).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[train] epoch={epoch:03d} "
            f"loss={total_loss / steps_per_epoch:.6f} "
            f"val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return model, best_val_acc


def export_rules(model: RulePolicy, encoder: PredicateEncoder, out_txt: str, top_k: int = 12):
    W = model.linear.weight.detach().cpu().numpy()
    b = model.linear.bias.detach().cpu().numpy()
    names = encoder.names

    lines = []
    n_actions = W.shape[0]

    for a in range(n_actions):
        weights = W[a]
        top_pos = np.argsort(-weights)[:top_k]
        top_neg = np.argsort(weights)[:top_k]

        lines.append(f"=== ACTION {a} ===")
        lines.append(f"bias = {b[a]:.6f}")
        lines.append("Top positive rules:")
        for idx in top_pos:
            lines.append(f"  + {weights[idx]: .6f} * [{names[idx]}]")
        lines.append("Top negative rules:")
        for idx in top_neg:
            lines.append(f"  - {abs(weights[idx]): .6f} * [{names[idx]}]")
        lines.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"[saved] rules -> {out_txt}")


def evaluate_rule_policy(
    model: RulePolicy,
    encoder: PredicateEncoder,
    env_id: str,
    action_type: str,
    seed: int,
    episodes: int,
    max_steps: int,
    device: str = "cpu",
):
    env = make_env(env_id=env_id, action_type=action_type, seed=seed)
    returns = []
    lengths = []

    model.eval()

    for ep in range(episodes):
        obs = safe_reset(env)
        obs = flatten_obs(obs)

        total_r = 0.0
        t = 0

        for t in range(max_steps):
            P = encoder.transform(obs[None, :])
            p = torch.tensor(P, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(p)
                action = int(torch.argmax(logits, dim=1).item())

            obs, reward, done, _ = safe_step(env, action)
            obs = flatten_obs(obs)

            total_r += float(reward)
            if done:
                break

        returns.append(total_r)
        lengths.append(t + 1)

    print(
        f"[eval] episodes={episodes} "
        f"return_mean={np.mean(returns):.4f} "
        f"return_std={np.std(returns):.4f} "
        f"len_mean={np.mean(lengths):.2f}"
    )
    return np.array(returns), np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")

    parser.add_argument("--oracle-type", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--oracle-ckpt", type=str, default="")

    parser.add_argument("--env-id", type=str, default="bus14")
    parser.add_argument("--action-type", type=str, default="topology")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--collect-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--n-quantiles", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-coef", type=float, default=1e-5)

    parser.add_argument("--save-dir", type=str, default="nudge_runs")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    encoder_json = os.path.join(args.save_dir, "predicate_encoder.json")
    model_pt = os.path.join(args.save_dir, "rule_policy.pt")
    rules_txt = os.path.join(args.save_dir, "rules.txt")

    if args.mode == "train":
        oracle = load_oracle(args.oracle_type, args.oracle_ckpt, device=args.device)

        X, y, r = collect_oracle_dataset(
            oracle=oracle,
            env_id=args.env_id,
            action_type=args.action_type,
            seed=args.seed,
            episodes=args.collect_episodes,
            max_steps=args.max_steps,
        )

        print(f"[data] obs={X.shape} actions={y.shape} rewards={r.shape}")
        print(f"[data] unique actions={np.unique(y)}")

        encoder = PredicateEncoder(
            n_quantiles=args.n_quantiles,
            max_features=args.max_features,
        )
        P = encoder.fit_transform(X)

        Xtr, ytr, Xva, yva = split_dataset(P, y, val_frac=0.1, seed=args.seed)
        n_actions = int(np.max(y) + 1)

        model, best_val_acc = train_rule_policy(
            P_train=Xtr,
            y_train=ytr,
            P_val=Xva,
            y_val=yva,
            n_actions=n_actions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l1_coef=args.l1_coef,
            device=args.device,
        )

        torch.save(model.state_dict(), model_pt)
        encoder.save(encoder_json)
        export_rules(model, encoder, rules_txt, top_k=12)

        print(f"[saved] model -> {model_pt}")
        print(f"[saved] encoder -> {encoder_json}")
        print(f"[done] best_val_acc={best_val_acc:.4f}")

        evaluate_rule_policy(
            model=model,
            encoder=encoder,
            env_id=args.env_id,
            action_type=args.action_type,
            seed=args.seed + 1,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            device=args.device,
        )

    else:
        encoder = PredicateEncoder.load(encoder_json)

        # Need action count; infer from saved weight shape.
        state = torch.load(model_pt, map_location=args.device)
        n_actions = state["linear.weight"].shape[0]
        n_predicates = state["linear.weight"].shape[1]

        model = RulePolicy(n_predicates=n_predicates, n_actions=n_actions).to(args.device)
        model.load_state_dict(state)

        evaluate_rule_policy(
            model=model,
            encoder=encoder,
            env_id=args.env_id,
            action_type=args.action_type,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            device=args.device,
        )


if __name__ == "__main__":
    main()