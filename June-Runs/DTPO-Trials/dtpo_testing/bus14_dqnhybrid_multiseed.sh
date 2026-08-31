#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
PPO_ORACLE=checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar
cd /home/sai/RL2Grid_full

run_one () {
  local folder="$1" seed="$2" oracle_name="$3"
  local before="/tmp/before_$(basename "$folder").txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  echo "=== START $folder (seed=$seed warmstart=$oracle_name) ==="
  PYTHONUNBUFFERED=1 timeout 7200 $PY main.py --alg DTPO --env-id bus14 --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters 150 --dtpo-batch 6000 --tree-max-leaf-nodes 16 \
      --warmstart-oracle "$oracle_name" \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $folder ==="
  bash dtpo_testing/finalize_run.sh "$folder" "$before" "bus14 DQN-hybrid($oracle_name) seed$seed" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $folder"
  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY eval_ckpt.py --ckpt "$ck" --episodes 20 > "$folder/eval_ckpt.txt" 2>&1
    echo "  eval_ckpt: $(grep -oE 'survival=[0-9.]+%' "$folder/eval_ckpt.txt" | tail -1)"
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "$PPO_ORACLE" --episodes 10 --seed 0 \
        --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]"
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "checkpoint/$oracle_name.tar" --episodes 10 --seed 0 \
        --out "$folder/fidelity_vs_teacher.json" 2>&1 | grep -E "\[fidelity\]" | sed 's/^/  (vs DQN teacher) /'
  fi
}

for s in 0 1 2; do run_one dtpo_testing/bus14_dqn225_seed$s $s DQN_bus14_T_0_0_H_I__1777591692_5026; done

for s in 0 1 2; do run_one dtpo_testing/bus14_dqn83_seed$s $s final_DQN_bus14_T_0_0__I__1776196348_3835; done
echo "===== BUS14 DQN-HYBRID SWEEP COMPLETE ====="
