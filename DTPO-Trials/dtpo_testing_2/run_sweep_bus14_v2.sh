#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ROOT=/home/sai/RL2Grid_full
OUT=$ROOT/dtpo_testing_2
SELECT_EP=100
cd "$ROOT"

BUS14_ORACLE=final_PPO_bus14_T_0_0__I__1775940444_3936

run_one () {
  local folder="$1" env="$2" seed="$3" mode="$4" oracle="$5" iters="$6" batch="$7" leaves="$8" tlimit="$9" evalep="${10}"
  mkdir -p "$folder"
  local tag=$(basename "$folder")
  local before="/tmp/before_v2_$tag.txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  local extra=""
  [ "$mode" = "hybrid" ] && extra="--warmstart-oracle $oracle"
  echo "=== START $tag (env=$env seed=$seed mode=$mode leaves=$leaves batch=$batch iters=$iters evalep=$evalep) $(date +%H:%M:%S) ==="
  PYTHONUNBUFFERED=1 timeout "$tlimit" $PY main.py --alg DTPO --env-id "$env" --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters "$iters" --dtpo-batch "$batch" --tree-max-leaf-nodes "$leaves" \
      --dtpo-eval-ep "$evalep" --dtpo-rerank-topk 5 $extra \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $tag $(date +%H:%M:%S) ==="

  ls checkpoint/*.tar 2>/dev/null | sort > "/tmp/after_v2_$tag.txt"
  local NEW=$(comm -13 "$before" "/tmp/after_v2_$tag.txt" | head -1)
  if [ -z "$NEW" ]; then echo "!! no checkpoint produced for $tag"; return; fi
  echo "  [select] honest selection over $SELECT_EP ep on $(basename "$NEW")"
  PYTHONUNBUFFERED=1 $PY dtpo_select.py --ckpt "$NEW" --episodes "$SELECT_EP" --topk 5 > "$folder/selection.txt" 2>&1
  grep -E "iter=|HONEST BEST|wrote" "$folder/selection.txt" | sed 's/^/  /'

  bash dtpo_testing/finalize_run.sh "$folder" "$before" "$env $mode seed$seed ($leaves leaves v2)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $tag"

  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "checkpoint/$oracle.tar" \
        --episodes 10 --seed 0 --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]" || echo "  (fidelity skipped)"
  fi
}

echo "########## BUS14 v2 (iters=250 batch=20000 leaves=64 eval_ep=20) $(date +%H:%M:%S) ##########"
for s in 0 1 2; do
  run_one "$OUT/bus14_scratch_seed$s" bus14 $s scratch "$BUS14_ORACLE" 250 20000 64 21600 20
  run_one "$OUT/bus14_hybrid_seed$s"  bus14 $s hybrid  "$BUS14_ORACLE" 250 20000 64 21600 20
done

echo "===== BUS14 v2 SWEEP COMPLETE $(date +%H:%M:%S) ====="
