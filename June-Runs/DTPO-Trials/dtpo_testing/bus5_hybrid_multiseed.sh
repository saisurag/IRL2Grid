#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
O5=final_PPO_bus5_T_0_0__I__1776174124_47043_45000000
O5_CKPT=checkpoint/$O5.tar
BUS14_LOG=/home/sai/RL2Grid_full/dtpo_testing/bus14_multiseed_driver.log
cd /home/sai/RL2Grid_full

for i in $(seq 1 270); do
  grep -q "BUS14 MULTISEED SWEEP COMPLETE" "$BUS14_LOG" 2>/dev/null && break
  sleep 60
done
echo "=== bus14 sweep done (or timed out); starting bus5 hybrid multi-seed ==="

run_one () {
  local folder="$1" seed="$2"
  local before="/tmp/before_$(basename "$folder").txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  echo "=== START $folder (seed=$seed mode=hybrid) ==="
  PYTHONUNBUFFERED=1 timeout 7200 $PY main.py --alg DTPO --env-id bus5 --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters 80 --dtpo-batch 2000 --tree-max-leaf-nodes 16 \
      --warmstart-oracle $O5 \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $folder ==="
  bash dtpo_testing/finalize_run.sh "$folder" "$before" "bus5 hybrid seed$seed (16 leaves)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $folder"
  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY eval_ckpt.py --ckpt "$ck" --episodes 20 > "$folder/eval_ckpt.txt" 2>&1
    echo "  eval_ckpt: $(grep -oE 'survival=[0-9.]+%' "$folder/eval_ckpt.txt" | tail -1)"
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "$O5_CKPT" --episodes 10 --seed 0 \
        --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]"
  fi
}

run_one dtpo_testing/bus5_hybrid_seed1 1
run_one dtpo_testing/bus5_hybrid_seed2 2
echo "===== BUS5 HYBRID MULTISEED + METRICS COMPLETE ====="
