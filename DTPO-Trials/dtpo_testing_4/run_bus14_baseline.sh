#!/bin/bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ROOT=/home/sai/RL2Grid_full
OUT=$ROOT/dtpo_testing_4
SELECT_EP=100
SELECT_TOPK=10
cd "$ROOT"

ETA=0.5; MIN_SAMPLES_LEAF=50; EVAL_EP=20; RERANK_TOPK=15; EVAL_EVERY=10
BUS14_ORACLE=final_PPO_bus14_T_0_0__I__1775940444_3936

run_one () {
  local folder="$1" seed="$2" iters="$3" batch="$4" leaves="$5" tlimit="$6"; shift 6
  local extra="$*"
  mkdir -p "$folder"
  local tag=$(basename "$folder")
  local before="/tmp/before_t4_$tag.txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"

  echo "=== START $tag (seed=$seed leaves=$leaves batch=$batch iters=$iters extra: $extra) $(date +%H:%M:%S) ==="
  PYTHONUNBUFFERED=1 timeout "$tlimit" $PY main.py --alg DTPO --env-id bus14 --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters "$iters" --dtpo-batch "$batch" --tree-max-leaf-nodes "$leaves" \
      --tree-min-samples-leaf "$MIN_SAMPLES_LEAF" --dtpo-eta "$ETA" \
      --dtpo-eval-every "$EVAL_EVERY" --dtpo-eval-ep "$EVAL_EP" --dtpo-rerank-topk "$RERANK_TOPK" \
      --warmstart-oracle "$BUS14_ORACLE" --warmstart-oracle-type ppo \
      $extra \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $tag $(date +%H:%M:%S) ==="

  ls checkpoint/*.tar 2>/dev/null | sort > "/tmp/after_t4_$tag.txt"
  local newlist=$(comm -13 "$before" "/tmp/after_t4_$tag.txt")
  local NEW=""
  [ -n "$newlist" ] && NEW=$(ls -t $newlist 2>/dev/null | head -1)
  if [ -z "$NEW" ]; then echo "!! no checkpoint produced for $tag"; return; fi
  echo "  [select] honest selection over $SELECT_EP ep (topk=$SELECT_TOPK) on $(basename "$NEW")"
  PYTHONUNBUFFERED=1 $PY dtpo_select.py --ckpt "$NEW" --episodes "$SELECT_EP" --topk "$SELECT_TOPK" \
      > "$folder/selection.txt" 2>&1
  grep -E "iter=|HONEST BEST|wrote" "$folder/selection.txt" | sed 's/^/  /'

  bash dtpo_testing/finalize_run.sh "$folder" "$before" "bus14 seed$seed ($tag, batch$batch, runA-optimizer)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $tag"

  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "checkpoint/$BUS14_ORACLE.tar" \
        --episodes 10 --seed 0 --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]" || echo "  (fidelity skipped)"
  fi
  echo "=== DONE $tag $(date +%H:%M:%S) ==="
}

echo "########## dtpo_testing_4 bus14 run-A (ppo-clip+anneal, batch 100000, 64 leaves) $(date +%H:%M:%S) ##########"
run_one "$OUT/bus14_baseline_seed0" 0 150 100000 64 172800 \
    --dtpo-ppo-clip 0.2 --dtpo-policy-updates 10 --dtpo-anneal-lr True

echo
echo "===== dtpo_testing_4 bus14 run-A COMPLETE $(date +%H:%M:%S) ====="
echo "--- bus14 baseline floor ~13.5%; batch-20k-alone was null (v2) ---"
grep -E "HONEST BEST" "$OUT/bus14_baseline_seed0/selection.txt" 2>/dev/null | sed 's/.*HONEST BEST = /bus14_baseline_seed0  /' || echo "(no result)"
