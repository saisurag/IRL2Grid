#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ORACLE=final_PPO_bus14_T_0_0__I__1775940444_3936
cd /home/sai/RL2Grid_full

echo "=== eval batch done (or timed out); starting bus14 multi-seed ==="

run_one () {
  local folder="$1" seed="$2" mode="$3"          # mode: scratch|hybrid
  local before="/tmp/before_$(basename "$folder").txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  local extra=""
  [ "$mode" = "hybrid" ] && extra="--warmstart-oracle $ORACLE"
  echo "=== START $folder (seed=$seed mode=$mode) ==="
  PYTHONUNBUFFERED=1 timeout 7200 $PY main.py --alg DTPO --env-id bus14 --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters 150 --dtpo-batch 6000 --tree-max-leaf-nodes 16 $extra \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $folder ==="
  bash dtpo_testing/finalize_run.sh "$folder" "$before" "bus14 $mode seed$seed (16 leaves)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $folder"
}

run_one dtpo_testing/bus14_scratch_seed1 1 scratch
run_one dtpo_testing/bus14_hybrid_seed1  1 hybrid
run_one dtpo_testing/bus14_scratch_seed2 2 scratch
run_one dtpo_testing/bus14_hybrid_seed2  2 hybrid
echo "===== BUS14 MULTISEED SWEEP COMPLETE ====="
