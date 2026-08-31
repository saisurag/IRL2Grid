#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_5
ORACLE=checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar
cd "$ROOT"
export PYTHONUNBUFFERED=1

COMMON="--alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
  --dtpo-iters 60 --dtpo-eval-every 5 --dtpo-batch 20000 --dtpo-eval-total 20 \
  --tree-max-leaf-nodes 16 --dtpo-eta 0.4 --dtpo-ppo-clip 0.2 \
  --dtpo-policy-updates 10 --dtpo-anneal-lr True --verbose True"

run () {
  local name=$1 extra=$2 seed=$3
  local out=$DIR/$name; mkdir -p "$out"
  echo "=== START $name (seed=$seed) $(date +%T) ==="
  $PY main.py $COMMON --seed "$seed" $extra > "$out/train.log" 2>&1
  echo "=== train exit=$? $name $(date +%T) ==="
  local ck; ck=$(ls -t checkpoint/DTPO_bus5_*.tar | head -1); cp "$ck" "$out/ckpt.tar"
  $PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "$name" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree_landscape" --landscape >> "$out/train.log" 2>&1
  echo "--- honest_eval ($name) ---"
  $PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1 | grep -E "HELD-OUT|WINNER|select-set"
}

echo "########## dtpo_testing_5 bus5 STEADY v2 (unbuffered) $(date +%T) ##########"
run bus5_scratch_seed0 ""                            0 &
run bus5_scratch_seed1 ""                            1 &
run bus5_hybrid_seed0  "--warmstart-oracle $ORACLE"  0 &
wait
echo "===== dtpo_testing_5 COMPLETE $(date +%T) ====="
