from common.imports import *

def get_alg_args() -> Namespace:
    parser = ap.ArgumentParser()

    # Oracle
    parser.add_argument(
        "--oracle-run-name",
        type=str,
        required=True,
        help="Checkpoint name of the trained DQN oracle, with or without '.tar'."
    )

    # VIPER loop
    parser.add_argument("--viper-iters", type=int, default=15, help="Number of VIPER iterations")
    parser.add_argument(
        "--viper-steps-per-iter",
        type=int,
        default=5000,
        help="Number of env interaction steps collected per VIPER iteration"
    )
    parser.add_argument(
        "--viper-resample-size",
        type=int,
        default=50000,
        help="Number of weighted samples drawn from the aggregated dataset to fit the tree"
    )
    parser.add_argument(
        "--viper-eval-freq",
        type=int,
        default=1,
        help="Evaluate every N VIPER iterations"
    )

    # Student tree
    parser.add_argument("--tree-max-depth", type=int, default=8, help="Decision tree max depth")
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=1,
        help="Decision tree min_samples_leaf"
    )

    return parser.parse_known_args()[0]