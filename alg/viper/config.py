from common.imports import *
from common.utils import str2bool


def get_alg_args() -> Namespace:
    parser = ap.ArgumentParser()

    # Oracle
    parser.add_argument(
        "--oracle-run-name",
        type=str,
        required=True,
        help="Checkpoint name of the trained PPO/DQN oracle, with or without '.tar'.",
    )
    parser.add_argument(
        "--oracle-type",
        type=str,
        default="auto",
        choices=["auto", "dqn", "ppo"],
        help="Oracle policy type. 'auto' detects it from the checkpoint ('qnet' => DQN, 'actor' => PPO).",
    )

    # VIPER loop
    parser.add_argument("--viper-iters", type=int, default=15, help="Number of VIPER iterations")
    parser.add_argument(
        "--viper-steps-per-iter",
        type=int,
        default=5000,
        help="Number of env interaction steps collected per VIPER iteration",
    )
    parser.add_argument(
        "--viper-resample-size",
        type=int,
        default=50000,
        help="Number of weighted samples drawn from the aggregated dataset to fit the tree",
    )
    parser.add_argument(
        "--viper-eval-freq",
        type=int,
        default=1,
        help="Evaluate every N VIPER iterations",
    )

    # Student tree / pruning
    parser.add_argument("--tree-max-depth", type=int, default=8, help="Decision tree max_depth")
    parser.add_argument(
        "--tree-ccp-alpha",
        type=float,
        default=0.0,
        help="Cost-complexity pruning alpha for DecisionTreeClassifier",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf for the VIPER decision tree",
    )
    parser.add_argument(
        "--tree-min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal tree node",
    )
    parser.add_argument(
        "--tree-max-leaf-nodes",
        type=int,
        default=None,
        help="Maximum number of leaf nodes in the VIPER tree",
    )
    parser.add_argument(
        "--viper-prune-sweep",
        action="store_true",
        help="Try multiple cost-complexity pruning alpha values and select the best tree",
    )
    parser.add_argument(
        "--viper-export-trees",
        type=str2bool,
        default=True,
        help="Export each evaluated tree (and the best tree) as Graphviz .dot under viper_trees/<run_name>/.",
    )

    return parser.parse_known_args()[0]
