from common.imports import *
from common.utils import str2bool

def get_env_args() -> Namespace:
    """
    Parse and return the command-line arguments for configuring the environment.

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    # Settings
    parser.add_argument("--env-id", type=str, default="bus14", help="ID of the grid2op environment")
    parser.add_argument("--n-envs", type=int, default=5, help="Number of parallel envs to run")
    parser.add_argument("--action-type", type=str, default="topology", choices=["topology", "redispatch"], help="Type of environment: topology (discrete) or redispatch (continuous)")
    parser.add_argument("--difficulty", type=int, default=0, help="Higher difficulty means bigger action spaces")
    parser.add_argument("--n1-reward", type=str2bool, default=False, help="Toggles N1 contintency analysis as an additional reward")

    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")

    # Scenarios
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to environment configuration file")

    # Normalization
    parser.add_argument("--norm-obs", type=str2bool, default=True, help="Toggle normalize observations")
    parser.add_argument("--use-heuristic", type=str2bool, default=False, help="Toggles heuristics for base operations")
    parser.add_argument("--heuristic-type", type=str, default="idle", choices=["idle", "reconnect"], help="Select the type of heuristic to use: idle or reconnect")

    parser.add_argument("--optimize-mem", type=str2bool, default=False, help="Whether to load data chunks upon resets (True), or the whole dataset once (False)")

    parser.add_argument("--chronics-mode", type=str, default="multifolder",
                        choices=["multifolder", "cache"],
                        help="multifolder: set_id works, chunked reads (correct). "
                             "cache: legacy MultifolderWithCache -- ONLY the first scenario is ever used.")
    parser.add_argument("--chronic-holdout", type=int, default=4,
                        help="Reserve every K-th chronic for eval and hide it from training. 0 = no split.")
    parser.add_argument("--eval-seed", type=int, default=12345,
                        help="Base seed for exogenous events during evaluation; episode seed = "
                             "eval_seed + chronic_id. Shared by all methods, so comparisons stay paired.")
    parser.add_argument("--eval-norm-reset", type=str, default="per-chronic",
                        choices=["per-chronic", "once"],
                        help="per-chronic: reset obs-normaliser before every eval episode (order-independent). "
                             "once: legacy, reset once per eval pass (scores depend on episode order).")
    parser.add_argument("--constraints-type", type=int, default=0, choices=[0, 1, 2], help="Select the type of constraints to use: no constraints (0), failure constraints (1), overloads constraints (2)")

    # Parse the arguments
    params, _ = parser.parse_known_args()

    return params
