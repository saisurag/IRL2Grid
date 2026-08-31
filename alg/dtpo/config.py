from common.imports import *
from common.utils import str2bool


def get_alg_args() -> Namespace:
    parser = ap.ArgumentParser()

    parser.add_argument("--dtpo-iters", type=int, default=1500, help="Number of DTPO iterations (N)")
    parser.add_argument("--dtpo-batch", type=int, default=10000,
                        help="Env steps collected per iteration (T). Large batches matter: the tree has no memory between iterations.")
    parser.add_argument("--dtpo-eta", type=float, default=1.0, help="Policy-gradient step size on the logits")
    parser.add_argument("--dtpo-gamma", type=float, default=0.99, help="Discount for GAE")
    parser.add_argument("--dtpo-gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--dtpo-norm-adv", type=str2bool, default=True, help="Normalize advantages to zero mean / unit var")
    parser.add_argument("--dtpo-eval-every", type=int, default=10, help="Evaluate the deterministic policy every N iterations")
    parser.add_argument("--dtpo-eval-ep", type=int, default=3, help="[legacy random-chronic eval] Episodes per in-training eval. Only used if --dtpo-eval-total <= 0. Default 3 is noisy.")
    parser.add_argument("--dtpo-eval-total", type=int, default=20, help="Deterministic in-loop eval: evaluate on this many FIXED, evenly-spaced chronics (capped at n_chronics) with cold-reset normalization each pass. Makes the in-training survival reproducible and consistent with honest_eval, so the running-best curve is monotone. <=0 falls back to the legacy random-chronic eval.")

    parser.add_argument("--dtpo-select-by", type=str, default="survival", choices=["survival","return"], help="Criterion for ranking stored candidate trees. 'survival' (default) uses deterministic survival on C_gate; 'return' uses the discounted TRAINING-composite return, the criterion the reference DTPO implementation uses.")
    parser.add_argument("--dtpo-rerank-topk", type=int, default=5, help="How many top candidate trees (by cheap in-training eval) to re-rank at the end. 0 disables re-ranking.")
    parser.add_argument("--dtpo-rerank-ep", type=int, default=50, help="Episodes used to honestly re-rank candidate trees at the end of training.")

    # Student regression tree (interpretability budget)
    parser.add_argument("--tree-max-leaf-nodes", type=int, default=16, help="Leaf cap for the policy tree")
    parser.add_argument("--tree-max-depth", type=int, default=None, help="Optional max depth for the policy tree")
    parser.add_argument("--tree-min-samples-leaf", type=int, default=1, help="Min samples per leaf")

    # Hybrid warm-start: if --warmstart-oracle is set, DTPO is seeded
    # with a soft-distilled tree from that oracle instead of the max-entropy leaf.
    parser.add_argument("--warmstart-oracle", type=str, default="",
                        help="Oracle checkpoint name/path to warm-start from (empty = from-scratch DTPO).")
    parser.add_argument("--warmstart-oracle-type", type=str, default="auto", choices=["auto", "dqn", "ppo"])
    parser.add_argument("--warmstart-seeds", type=int, default=5, help="Seeded episodes for the distillation rollout")
    parser.add_argument("--warmstart-steps-cap", type=int, default=20000, help="Cap on distillation dataset size")
    parser.add_argument("--warmstart-temperature", type=float, default=1.0, help="Softmax temperature for soft targets")
    parser.add_argument("--warmstart-curate", type=str2bool, default=False, help="Curate the action set before distilling")
    parser.add_argument("--warmstart-curate-strategy", type=str, default="coverage+criticality")
    parser.add_argument("--warmstart-coverage", type=float, default=0.999)

    parser.add_argument("--warmstart-dagger-rounds", type=int, default=0,
                        help="Extra DAgger aggregation rounds during warm-start (0 = off).")
    parser.add_argument("--warmstart-dagger-steps-cap", type=int, default=10000,
                        help="Cap on states collected per DAgger round.")

    parser.add_argument("--dtpo-ppo-clip", type=float, default=0.0,
                        help="PPO clip epsilon for the target update. 0 = legacy unclipped single-step.")
    parser.add_argument("--dtpo-policy-updates", type=int, default=1,
                        help="Inner grad-step+refit candidates per iteration (needs --dtpo-ppo-clip>0 to matter).")
    parser.add_argument("--dtpo-anneal-lr", type=str2bool, default=False,
                        help="Linearly anneal eta -> 0 over iterations (reference behavior).")

    # Structure-preserving updates (extension to reduce iteration-to-iteration instability). 
    # The paper's Algorithm 1 refits a brand-new regression tree from scratch every iteration.
    # A small shift in targets can flip which feature/threshold a node splits on, producing a structurally different tree (and a big jump in behavior/survival)
    # When --dtpo-structure-refit-every K is > 1, only every Kth iteration does a full CART refit; on the other iterations we keep the CURRENT tree's split structure fixed and only overwrite each visited leaf's value with this batch's target mean.
    parser.add_argument("--dtpo-structure-refit-every", type=int, default=1,
                        help="Full CART structural refit every K iterations (1 = every iteration, "
                             "the paper-faithful default). K>1 does leaf-value-only updates on the "
                             "other iterations, keeping the tree's splits fixed.")

    # Survival-gated acceptance (extension). 
    # A candidate is deployed only if its fixed-chronic survival >= current candidate's, otherwise we revert back to original/current.

    parser.add_argument("--dtpo-survival-gate", type=str2bool, default=False,
                        help="Gate rollout-policy acceptance on the deterministic eval survival "
                             "(revert to the previous (current) tree if the candidate scores lower). "
                             "Use with --dtpo-eval-every 1.")
    parser.add_argument("--dtpo-gate-patience", type=int, default=0,
                        help="If >0: after this many consecutive gate rejections, swap the rollout policy to the current candidate to escape a stall. " \
                             "The acceptance bar is NOT lowered. 0 = strict gate.")

    parser.add_argument("--dtpo-gate-tol", type=float, default=0.0,
                        help="accept a candidate whose survival is >= incumbent - tolerance.")
    parser.add_argument("--dtpo-flush-replay-on-revert", type=str2bool, default=True,
                        help="clear the replay buffer when the gate reverts, so the rejected branch's off-policy transitions do not drive the next fit.")
    parser.add_argument("--dtpo-replay-weight-fit", type=str2bool, default=True,
                        help="pass the clipped importance ratio to CART as sample_weight, so stale replayed samples influence SPLIT SELECTION less, not just the leaf targets. No effect when M=1.")

    # Replay aggregation (extension).
    # Aggregating the last M batches (with clipped importance ratios pi_cur/pi_behavior correcting for the older batches' off-policy data) gives the tree fit M-times the data.
    # Stabilises split choices. M=1 reproduces the legacy on-policy behavior exactly. Only wired into the unclipped (--dtpo-ppo-clip 0) branch.

    parser.add_argument("--dtpo-replay-iters", type=int, default=1,
                        help="Fit the tree on the last M batches of experience (importance-weighted). "
                             "1 = current batch only (paper-faithful).")
    parser.add_argument("--dtpo-replay-ratio-clip", type=float, default=2.0,
                        help="Clip on the importance ratios pi_cur/pi_behavior used for replayed "
                             "batches (only applied when --dtpo-replay-iters > 1).")

    # Value critic (an NN, used only to improve optimization; not part of the interpretable policy)
    parser.add_argument("--critic-layers", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--critic-act-fn", type=str, default="tanh")
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-epochs", type=int, default=4, help="Critic update epochs per iteration (E)")
    parser.add_argument("--critic-batch", type=int, default=64, help="Critic minibatch size (B)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Critic gradient clipping")

    return parser.parse_known_args()[0]
