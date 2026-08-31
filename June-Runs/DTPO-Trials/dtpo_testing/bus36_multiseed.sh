#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ORACLE=DQN_bus36-M_T_0_0_H_I__1777671868_9169
cd /home/sai/RL2Grid_full

run_one () {
  local folder="$1" seed="$2" mode="$3"          # mode: scratch|hybrid|curated
  local before="/tmp/before_$(basename "$folder").txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  local extra=""
  [ "$mode" = "hybrid" ]  && extra="--warmstart-oracle $ORACLE"
  [ "$mode" = "curated" ] && extra="--warmstart-oracle $ORACLE --warmstart-curate True --warmstart-curate-strategy coverage+criticality --warmstart-coverage 0.999"
  echo "=== START $folder (seed=$seed mode=$mode) ==="
  PYTHONUNBUFFERED=1 timeout 18000 $PY main.py --alg DTPO --env-id bus36-M --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters 200 --dtpo-batch 8000 --tree-max-leaf-nodes 32 $extra \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $folder ==="
  bash dtpo_testing/finalize_run.sh "$folder" "$before" "bus36-M $mode seed$seed (32 leaves)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $folder"
}

# scratch + hybrid first (both seeds), then curated (both seeds)
run_one dtpo_testing/bus36M_scratch_seed1 1 scratch
run_one dtpo_testing/bus36M_hybrid_seed1  1 hybrid
run_one dtpo_testing/bus36M_scratch_seed2 2 scratch
run_one dtpo_testing/bus36M_hybrid_seed2  2 hybrid
run_one dtpo_testing/bus36M_curated_seed1 1 curated
run_one dtpo_testing/bus36M_curated_seed2 2 curated
echo "===== BUS36-M MULTISEED SWEEP COMPLETE ====="
