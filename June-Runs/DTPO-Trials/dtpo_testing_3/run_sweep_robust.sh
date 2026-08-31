#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ROOT=/home/sai/RL2Grid_full
OUT=$ROOT/dtpo_testing_3
SELECT_EP=100
SELECT_TOPK=10
cd "$ROOT"

ETA=0.5
MIN_SAMPLES_LEAF=50
EVAL_EP=20
RERANK_TOPK=15
EVAL_EVERY=10

BUS36_ORACLE=DQN_bus36-M_T_0_0_H_I__1777671868_9169
BUS14_ORACLE=final_PPO_bus14_T_0_0__I__1775940444_3936 

run_one () {
  local folder="$1" env="$2" seed="$3" oracle="$4" iters="$5" batch="$6" leaves="$7" tlimit="$8"
  mkdir -p "$folder"
  local tag=$(basename "$folder")
  local before="/tmp/before_t3_$tag.txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"

  echo "=== START $tag (env=$env seed=$seed mode=hybrid leaves=$leaves batch=$batch iters=$iters eta=$ETA msl=$MIN_SAMPLES_LEAF evalep=$EVAL_EP) $(date +%H:%M:%S) ==="
  PYTHONUNBUFFERED=1 timeout "$tlimit" $PY main.py --alg DTPO --env-id "$env" --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters "$iters" --dtpo-batch "$batch" --tree-max-leaf-nodes "$leaves" \
      --tree-min-samples-leaf "$MIN_SAMPLES_LEAF" --dtpo-eta "$ETA" \
      --dtpo-eval-every "$EVAL_EVERY" --dtpo-eval-ep "$EVAL_EP" --dtpo-rerank-topk "$RERANK_TOPK" \
      --warmstart-oracle "$oracle" \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $tag $(date +%H:%M:%S) ==="

  ls checkpoint/*.tar 2>/dev/null | sort > "/tmp/after_t3_$tag.txt"
  local NEW=$(comm -13 "$before" "/tmp/after_t3_$tag.txt" | head -1)
  if [ -z "$NEW" ]; then echo "!! no checkpoint produced for $tag"; return; fi
  echo "  [select] honest selection over $SELECT_EP ep (topk=$SELECT_TOPK) on $(basename "$NEW")"
  PYTHONUNBUFFERED=1 $PY dtpo_select.py --ckpt "$NEW" --episodes "$SELECT_EP" --topk "$SELECT_TOPK" \
      > "$folder/selection.txt" 2>&1
  grep -E "iter=|HONEST BEST|wrote" "$folder/selection.txt" | sed 's/^/  /'

  bash dtpo_testing/finalize_run.sh "$folder" "$before" "$env hybrid seed$seed ($leaves leaves, robust)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $tag"

  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "checkpoint/$oracle.tar" \
        --episodes 10 --seed 0 --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]" || echo "  (fidelity skipped)"
  fi
  echo "=== DONE $tag $(date +%H:%M:%S) ==="
}

echo "########## dtpo_testing_3 ROBUSTNESS SWEEP (no curation) $(date +%H:%M:%S) ##########"

echo "########## BUS36-M  hybrid<-DQN  seed 0  (iters=200 batch=12000 leaves=32) ##########"
run_one "$OUT/bus36M_hybrid_seed0" bus36-M 0 "$BUS36_ORACLE" 200 12000 32 28800

echo "########## BUS14    hybrid<-PPO  seeds 0,1  (iters=150 batch=9000 leaves=16) ##########"
for s in 0 1; do
  run_one "$OUT/bus14_hybrid_seed$s" bus14 $s "$BUS14_ORACLE" 150 9000 16 14400
done

echo
echo "===== dtpo_testing_3 SWEEP COMPLETE $(date +%H:%M:%S) ====="
echo "--- honest results ---"
for f in "$OUT"/*/selection.txt; do
  printf "%-32s " "$(basename "$(dirname "$f")")"
  grep -E "HONEST BEST" "$f" 2>/dev/null | sed 's/.*HONEST BEST = //' || echo "(no result)"
done
