#!/usr/bin/env python3
"""
Generate the experiment lists for the RL2Grid interpretable-policy empirical study.

Methods
  distill            soft-CART distillation only (no RL)          [needs oracle]
  dagger             original DAgger (MLP student, BC baseline)   [needs oracle]
  viper              original VIPER (hard-label tree)             [needs oracle]
  dtpo               original DTPO, from scratch (Vos & Verwer)
  dtpo_ws            DTPO + oracle warm-start (hybrid, no stabilization)
  dtpo_full          ENHANCED: warm-start + survival gate + replay 4 + K=5
  dtpo_gate          ablation: warm-start + gate only
  dtpo_replay        ablation: warm-start + replay only
  dtpo_scratch_full  scratch + gate + replay (stabilization without a teacher)

Usage example:
  python gen_experiments.py --envs bus5 --seeds 100 101 102
"""
import argparse
import os

ORACLES = {
    "bus5":    "__ROOT__/checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar",
    "bus14":   "__ROOT__/checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar",
    "bus36-M": "__ROOT__/checkpoint/final_PPO_bus36-M_T_100_0__I__1784927195_14574.tar",
    "bus118-M": None,
}

EVAL_TOTAL  = {"bus5": 20, "bus14": 40, "bus36-M": 40}
HONEST_TOTAL = {"bus5": 20, "bus14": 80, "bus36-M": 80}

DTPO_BASE = ("--alg DTPO --action-type topology --n-envs 3 "
             "--dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 "
             "--dtpo-gae-lambda 0.95 --dtpo-norm-adv True --dtpo-eval-every 1 "
             "--dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False "
             "--critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 "
             "--critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 "
             "--checkpoint True --track False --time-limit 1300 --verbose True")
WARMSTART = "--warmstart-oracle {oracle} --warmstart-temperature 0.5 --warmstart-dagger-rounds 0"
STABILIZE = "--dtpo-survival-gate True --dtpo-replay-iters 4"


def dtpo_line(env, seed, leaves, *, warm, gate, replay_iters, K, diff=0, patience=0):
    parts = [DTPO_BASE,
             f"--env-id {env} --difficulty {diff} --seed {seed}",
             f"--dtpo-eval-total {EVAL_TOTAL[env]}",
             f"--tree-max-leaf-nodes {leaves}",
             f"--dtpo-structure-refit-every {K}",
             f"--dtpo-replay-iters {replay_iters}",
             f"--dtpo-survival-gate {'True' if gate else 'False'}"]
    if patience:
        parts.append(f"--dtpo-gate-patience {patience}")
    if warm:
        parts.append(WARMSTART.format(oracle=ORACLES[env]))
    return " ".join(parts)


def viper_line(env, seed, leaves, diff=0):
    return (f"--alg VIPER --env-id {env} --action-type topology --difficulty {diff} "
            f"--n-envs 3 --oracle-run-name {ORACLES[env]} "
            f"--viper-iters 15 --viper-steps-per-iter 5000 "
            f"--tree-max-leaf-nodes {leaves} --tree-max-depth 12 "
            f"--viper-export-trees False "
            f"--checkpoint True --track False --time-limit 1300 --seed {seed} --verbose True")


def dagger_line(env, seed, diff=0):
    return (f"--alg DAGGER --env-id {env} --action-type topology --difficulty {diff} "
            f"--n-envs 3 --expert-ckpt {ORACLES[env]} "
            f"--checkpoint True --track False --time-limit 1300 --seed {seed} --verbose True")


def distill_line(env, seed, leaves, diff=0):
    return (f"--env-id {env} --difficulty {diff} --oracle {ORACLES[env]} "
            f"--tree-max-leaf-nodes {leaves} --temperature 0.5 --seed {seed}")


def daggertree_line(env, seed, leaves, diff=0):
    # DAgger with a interpretable tree student, 5 rounds of DAgger
    return distill_line(env, seed, leaves, diff) + " --dagger-rounds 5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["bus5", "bus14"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    ap.add_argument("--tiers", nargs="+", default=["core", "ablation", "frontier", "hparam", "stress", "daggertree"])
    ap.add_argument("--leaves-core", type=int, default=16)
    ap.add_argument("--leaves-frontier", nargs="+", type=int, default=[8, 32, 64])
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "experiments"))
    ap.add_argument("--suffix", default="", help="Appended to the output filename, e.g. "
                    "--suffix _daggertree -> experiments_bus5_daggertree.txt (avoids "
                    "clobbering a list whose array is still queued/running)")
    ap.add_argument("--no-mlp-dagger", action="store_true",
                    help="Skip the MLP-student DAgger baseline (core tier)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    counts = {}

    for env in a.envs:
        lines = []
        L = a.leaves_core
        has_oracle = ORACLES.get(env) is not None
        ht = HONEST_TOTAL[env]

        def add(name, script, args):
            lines.append(f"{name}|{script}|{ht}|{args}")

        for s in a.seeds:
            if "core" in a.tiers:
                if has_oracle:
                    add(f"{env}_distill_L{L}_s{s}", "distill_only.py", distill_line(env, s, L))
                    if not a.no_mlp_dagger:
                        add(f"{env}_dagger_s{s}", "main.py", dagger_line(env, s))
                    add(f"{env}_viper_L{L}_s{s}", "main.py", viper_line(env, s, L))
                    add(f"{env}_dtpo-ws_L{L}_s{s}", "main.py",
                        dtpo_line(env, s, L, warm=True, gate=False, replay_iters=1, K=1))
                    add(f"{env}_dtpo-full_L{L}_s{s}", "main.py",
                        dtpo_line(env, s, L, warm=True, gate=True, replay_iters=4, K=5))
                add(f"{env}_dtpo_L{L}_s{s}", "main.py",
                    dtpo_line(env, s, L, warm=False, gate=False, replay_iters=1, K=1))

            if "ablation" in a.tiers:
                if has_oracle:
                    add(f"{env}_dtpo-gate_L{L}_s{s}", "main.py",
                        dtpo_line(env, s, L, warm=True, gate=True, replay_iters=1, K=5))
                    add(f"{env}_dtpo-replay_L{L}_s{s}", "main.py",
                        dtpo_line(env, s, L, warm=True, gate=False, replay_iters=4, K=5))
                add(f"{env}_dtpo-scratchfull_L{L}_s{s}", "main.py",
                    dtpo_line(env, s, L, warm=False, gate=True, replay_iters=4, K=5))

            if "frontier" in a.tiers and has_oracle:
                for lf in a.leaves_frontier:
                    add(f"{env}_distill_L{lf}_s{s}", "distill_only.py", distill_line(env, s, lf))
                    add(f"{env}_viper_L{lf}_s{s}", "main.py", viper_line(env, s, lf))
                    add(f"{env}_dtpo-full_L{lf}_s{s}", "main.py",
                        dtpo_line(env, s, lf, warm=True, gate=True, replay_iters=4, K=5))

            if "hparam" in a.tiers and has_oracle:
                for m in (2, 8):
                    add(f"{env}_dtpo-full-M{m}_L{L}_s{s}", "main.py",
                        dtpo_line(env, s, L, warm=True, gate=True, replay_iters=m, K=5))
                add(f"{env}_dtpo-full-pat3_L{L}_s{s}", "main.py",
                    dtpo_line(env, s, L, warm=True, gate=True, replay_iters=4, K=5, patience=3))

            if "daggertree" in a.tiers and has_oracle:
                for lf in [L] + list(a.leaves_frontier):
                    add(f"{env}_daggertree_L{lf}_s{s}", "distill_only.py",
                        daggertree_line(env, s, lf))

            if "stress" in a.tiers and env == "bus14":
                add(f"{env}-d1_dtpo_L{L}_s{s}", "main.py",
                    dtpo_line(env, s, L, warm=False, gate=False, replay_iters=1, K=1, diff=1))
                add(f"{env}-d1_dtpo-scratchfull_L{L}_s{s}", "main.py",
                    dtpo_line(env, s, L, warm=False, gate=True, replay_iters=4, K=5, diff=1))

        out = os.path.join(a.out_dir, f"experiments_{env}{a.suffix}.txt")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        counts[env] = len(lines)
        print(f"{out}: {len(lines)} jobs")

    print(f"\nTotal: {sum(counts.values())} jobs. Submit each list with:")
    for env in counts:
        print(f"  ./submit_array.sh experiments/experiments_{env}.txt")


if __name__ == "__main__":
    main()
