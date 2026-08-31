from common.imports import *
from common.utils import str2bool

def get_alg_args() -> Namespace:
    parser = ap.ArgumentParser(allow_abbrev=False)

    # DAgger loop
    parser.add_argument("--dagger-iters", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=2000)   # per iter (per env)
    parser.add_argument("--beta-init", type=float, default=1.0)      # start: mostly expert
    parser.add_argument("--beta-final", type=float, default=0.0)     # end: student only
    parser.add_argument("--beta-decay", type=float, default=0.95)    # exp decay per iter

    # Expert PPO checkpoint (the .tar saved by RL2Grid)
    parser.add_argument("--expert-ckpt", type=str, required=True)

    # Supervised (behaviour cloning) training
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--dataset-max", type=int, default=500_000)  # cap memory
    parser.add_argument("--eval-every", type=int, default=5)

    # Network (reuse PPO Agent architecture args)
    parser.add_argument('--actor-layers', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[256, 256])  # unused but PPO Agent expects it
    parser.add_argument('--actor-act-fn', type=str, default='tanh')
    parser.add_argument('--critic-act-fn', type=str, default='tanh')  # unused

    # Checkpoint / logging
    parser.add_argument("--checkpoint-every", type=int, default=5)
    # parser.add_argument("--verbose", type=str2bool, default=True)

    return parser.parse_known_args()[0]