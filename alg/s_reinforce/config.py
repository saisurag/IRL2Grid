from common.imports import *
from common.utils import str2bool


def get_alg_args() -> Namespace:
    """Parse command-line arguments for S-REINFORCE.

    S-REINFORCE extends PPO with a Symbolic Regressor (SR) component that is
    fitted periodically on the NN's action-probability outputs and used via
    importance sampling to modulate the policy gradient advantages.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    # ── Rollout ────────────────────────────────────────────────────────────────
    parser.add_argument("--total-timesteps", type=int, default=25000000,
                        help="Total timesteps for the experiment")
    parser.add_argument("--n-steps", type=int, default=20000,
                        help="Steps per policy rollout")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Total timesteps between deterministic evals")

    # ── Network ────────────────────────────────────────────────────────────────
    parser.add_argument('--actor-layers', nargs='+', type=int, default=[256, 256],
                        help='Actor network size')
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[256, 256],
                        help='Critic network size')
    parser.add_argument('--actor-act-fn', type=str, default='tanh',
                        help='Actor activation function')
    parser.add_argument('--critic-act-fn', type=str, default='tanh',
                        help='Critic activation function')
    parser.add_argument("--actor-lr", type=float, default=3e-4,
                        help="Learning rate for the actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Learning rate for the critic")
    parser.add_argument("--anneal-lr", type=str2bool, default=True,
                        help="Toggles learning rate annealing")

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    parser.add_argument("--gamma", type=float, default=.9,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=.95,
                        help="Lambda for the generalised advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=80,
                        help="Number of update epochs")
    parser.add_argument("--n-minibatches", type=int, default=4,
                        help="Number of minibatches")
    parser.add_argument("--max-grad-norm", type=float, default=10,
                        help="Maximum norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Target KL divergence threshold")
    parser.add_argument("--norm-adv", type=str2bool, default=True,
                        help="Toggles advantage normalisation")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="Surrogate clip coefficient")
    parser.add_argument("--clip-vfloss", type=str2bool, default=True,
                        help="Toggles clip for value function loss")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")

    # ── Lagrangian (kept for API parity with LagrPPO) ─────────────────────────
    parser.add_argument("--cost-threshold", type=float, default=15.0,
                        help="Cost threshold")
    parser.add_argument("--lag-mul", type=float, default=0.0,
                        help="Initial value for the Lagrangian multiplier")
    parser.add_argument("--lag-lr", type=float, default=0.05,
                        help="Learning rate for the Lagrangian multiplier")

    # ── S-REINFORCE specific ──────────────────────────────────────────────────
    parser.add_argument("--sr-interval", type=int, default=10,
                        help="Refit symbolic regressor every N rollouts. "
                             "Higher = cheaper but staler symbolic policy.")
    parser.add_argument("--sr-iterations", type=int, default=20,
                        help="PySR niterations per SR fit. "
                             "Higher = better symbolic fit but slower.")
    parser.add_argument("--sr-maxsize", type=int, default=12,
                        help="Maximum complexity (node count) of SR expressions.")
    parser.add_argument("--sr-max-obs-features", type=int, default=20,
                        help="Maximum number of observation features fed to SR. "
                             "Features are selected by variance; keeps SR tractable "
                             "on high-dimensional grid observations.")
    parser.add_argument("--is-clip-low", type=float, default=0.5,
                        help="Lower clip bound for importance-sampling weights.")
    parser.add_argument("--is-clip-high", type=float, default=2.0,
                        help="Upper clip bound for importance-sampling weights.")

    return parser.parse_known_args()[0]
