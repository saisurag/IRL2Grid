from collections import deque

from common.imports import *
from common.logger import Logger
from .utils import auxiliary_make_env

class Evaluator:
    """Evaluator class for evaluating a reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device) -> None:
        """Initialize the Evaluator with the given arguments, logger, and device.

        Args:
            args: Arguments containing environment configuration.
            logger: Logger for storing evaluation metrics.
            device: Device to run the model on.
        """
 
        self.args = args
        self.env = auxiliary_make_env(args, eval_env=True)[0]  # Initialize synchronized vector environment
        self.max_steps = self.env.init_env.chronics_handler.max_episode_duration()  # Get max episode duration

        self.logger = logger  # Logger for evaluation metrics
        self.device = device  # Device for model inference

        # Fix n° rewards based on the env specs (for simplifying logging ops)
        # The reward returned by the env is an increasing survival reward
        self.reward_tags = ['Redispatch Reward', 'Line Margin Reward', 'Overload Reward']   
        if args.action_type == 'topology': self.reward_tags += ['Topology Reward']      
        if args.n1_reward: self.reward_tags += ['N1 Reward']
        self.use_heuristic = args.use_heuristic
        if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))
            
    def evaluate(self, glob_step: int, model: object, eval_ep: int = 3) -> dict:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))

        obs, info = self.env.reset()

        if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        while len(ep_survivals) < eval_ep:
            action = model.get_eval_action(th.tensor(obs, dtype=th.float).to(self.device)).detach().cpu().numpy()
            next_obs, _, _, _, info = self.env.step(action)

            obs = next_obs

            ep_rewards += list(info['rewards'].values())
            # Record rewards for plotting purposes
            if "episode" in info:   # Denote end of an episode
                ep_survivals.append(self.env.init_env.nb_time_step/self.max_steps)
                ep_returns.append(ep_rewards)
                obs, _ = self.env.reset()
                ep_rewards = np.zeros(len(self.reward_tags))

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, self.reward_tags)

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")

        return {
            "survival": avg_survival,
            "return": avg_return,
        }


    def _norm_wrapper(self):
        """Return the NormalizeObservation wrapper on the eval env (or None)."""
        e = self.env
        while e is not None:
            if hasattr(e, "reset_stats") and hasattr(e, "obs_rms"):
                return e
            e = getattr(e, "env", None)
        return None

    def n_chronics(self) -> int:
        try:
            return len(self.env.init_env.chronics_handler.real_data.subpaths)
        except Exception:
            return 0

    def held_out_ids(self) -> np.ndarray:
        n = self.n_chronics()
        k = int(getattr(self.args, "chronic_holdout", 0) or 0)
        if n <= 0:
            return np.arange(0)
        ids = np.arange(n)
        return ids[ids % k == k - 1] if k > 1 else ids

    def gate_chronic_ids(self, total: int) -> np.ndarray:
        n = self.n_chronics()
        k = int(getattr(self.args, "chronic_holdout", 0) or 0)
        if n <= 0:
            return np.arange(max(1, total))
        ids = np.arange(n)
        pool = ids[ids % k != k - 1] if k > 1 else ids
        total = min(total, len(pool))
        return np.unique(pool[np.unique(np.linspace(0, len(pool) - 1, total).astype(int))])

    def fixed_chronic_ids(self, total: int) -> np.ndarray:
        pool = self.held_out_ids()
        if len(pool) == 0:
            return np.arange(max(1, total))
        total = min(total, len(pool))
        return np.unique(pool[np.unique(np.linspace(0, len(pool) - 1, total).astype(int))])

    def evaluate_fixed(self, model: object, chronic_ids, reset_norm: bool = True, disc_gamma: float = None) -> dict:

        nw = self._norm_wrapper()
        per_chronic_reset = (getattr(self.args, "eval_norm_reset", "per-chronic") == "per-chronic")
        if nw is not None and reset_norm and not per_chronic_reset:
            nw.reset_stats()

        survs = []
        disc_returns = []
        for cid in np.asarray(chronic_ids).tolist():
            if nw is not None and reset_norm and per_chronic_reset:
                nw.reset_stats()
            _es = int(getattr(self.args, "eval_seed", 12345))
            try:
                self.env.init_env.seed(_es + int(cid))
            except Exception:
                pass

            self.env.init_env.set_id(int(cid))
            try:
                obs, info = self.env.reset(options={"time serie id": int(cid)})
            except TypeError:
                obs, info = self.env.reset()

            g, disc, t = (disc_gamma or 0.0), 0.0, 0
            while True:
                action = model.get_eval_action(
                    th.tensor(obs, dtype=th.float).to(self.device)).detach().cpu().numpy()
                obs, rew, _, _, info = self.env.step(action)
                if disc_gamma is not None:
                    comp = float(rew)
                    rw = info.get("rewards") or {}
                    comp += 0.5 * float(rw.get("redispatchReward", 0.0))
                    comp += 1.0 * float(rw.get("overloadReward", 0.0))
                    disc += (g ** t) * comp; t += 1
                if "episode" in info:
                    survs.append(self.env.init_env.nb_time_step / self.max_steps)
                    if disc_gamma is not None: disc_returns.append(disc)
                    break
        survs = np.asarray(survs, dtype=np.float64)
        out = {
            "survival": float(survs.mean()) if len(survs) else 0.0,
            "survival_std": float(survs.std()) if len(survs) else 0.0,
            "per_chronic": survs.tolist(),
        }
        if disc_gamma is not None:
            dr = np.asarray(disc_returns, dtype=np.float64)
            out["disc_return"] = float(dr.mean()) if len(dr) else 0.0
            out["disc_return_per_chronic"] = dr.tolist()
        return out


class CMDPEvaluator(Evaluator):
    """Evaluator class for evaluating a constrained reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device) -> None:
        super().__init__(args, logger, device)

    def evaluate(self, glob_step: int, model: object, eval_ep: int = 3) -> dict:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_cost_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store cost returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))
        ep_costs = 0

        obs, info = self.env.reset()

        if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        while len(ep_survivals) < eval_ep:
            action = model.get_eval_action(th.tensor(obs, dtype=th.float).to(self.device)).detach().numpy()
            next_obs, _, _, _, info = self.env.step(action)

            obs = next_obs
            ep_rewards += list(info['rewards'].values())
            ep_costs += info['cost']

            # Record rewards and cost for plotting purposes
            if "episode" in info:   # Denote end of an episode
                ep_survivals.append(self.env.init_env.nb_time_step/self.max_steps)
                ep_returns.append(ep_rewards)
                ep_cost_returns.append(ep_costs)

                obs, _ = self.env.reset()
                ep_rewards = np.zeros(len(self.reward_tags))
                ep_costs = 0

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]
        avg_cost_return = [sum(ep_cost_returns)/eval_ep]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, avg_cost_return, self.reward_tags)

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}, cost_return={avg_cost_return}")

        return {
            "survival": avg_survival,
            "return": avg_return,
            "cost_return": avg_cost_return,
        }
