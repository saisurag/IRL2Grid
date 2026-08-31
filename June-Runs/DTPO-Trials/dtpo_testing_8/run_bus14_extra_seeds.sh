#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_8
ORACLE_BUS14="checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/gatereplay_sweep_RESULTS.txt

run_bus14 () {   # seed
  local seed=$1
  local out=$DIR/bus14_gatereplay_K5_seed${seed}
  mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START bus14 K=5 seed=$seed $(date +%T) ==="
  $PY main.py --alg DTPO --env-id bus14 --action-type topology --n-envs 3 \
    --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
    --dtpo-norm-adv True --dtpo-eval-every 1 --dtpo-eval-total 40 \
    --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
    --dtpo-structure-refit-every 5 \
    --dtpo-survival-gate True --dtpo-replay-iters 4 \
    --tree-max-leaf-nodes 16 \
    --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
    --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
    --warmstart-oracle $ORACLE_BUS14 --warmstart-temperature 0.5 --warmstart-dagger-rounds 0 \
    --seed "$seed" --verbose True > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus14_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "bus14 K=5 seed=$seed honest80=NA (no ckpt)" >> "$RES"
    echo "=== FAIL bus14 K=5 seed=$seed $(date +%T) ==="
    return
  fi
  cp "$ck" "$out/ckpt.tar"
  $PY honest_eval.py --ckpt "$out/ckpt.tar" --total 80 > "$out/honest_eval_total80.log" 2>&1
  echo "bus14 K=5 seed=$seed honest80=$(grep 'HELD-OUT REPORT' "$out/honest_eval_total80.log" | grep -oE 'survival = [0-9.]+%')" >> "$RES"
  echo "=== DONE bus14 K=5 seed=$seed $(date +%T) ==="
}

run_bus14 101 &
run_bus14 102 &
wait
echo "===== BUS14 EXTRA SEEDS COMPLETE $(date +%T) ====="
