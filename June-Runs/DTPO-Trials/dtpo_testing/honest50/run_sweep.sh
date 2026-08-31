#!/bin/bash
# Pipeline per run:
#   1. train DTPO  -> stores top-K candidate TREES in the checkpoint (the policy is
#      still ONE plain decision tree; this is checkpoint selection, not explanation).
#   2. dtpo_select.py -> re-evaluates each candidate with a FRESH Evaluator over 100
#      episodes (reproducible; fixes the stateful-normalization + chronic-sampling
#      bias) and writes the genuinely-best tree back into the checkpoint.
#   3. finalize -> copies the selected ckpt into the run folder + tree SVGs + curve.
#   4. fidelity vs the grid's reference oracle.
# Each run -> its own folder under dtpo_testing/honest50/.

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
ROOT=/home/sai/RL2Grid_full
OUT=$ROOT/dtpo_testing/honest50
SELECT_EP=100
cd "$ROOT"

BUS36_ORACLE=DQN_bus36-M_T_0_0_H_I__1777671868_9169
BUS14_ORACLE=final_PPO_bus14_T_0_0__I__1775940444_3936

run_one () {
  local folder="$1" env="$2" seed="$3" mode="$4" oracle="$5" iters="$6" batch="$7" leaves="$8" tlimit="$9"
  mkdir -p "$folder"
  local tag=$(basename "$folder")
  local before="/tmp/before_$tag.txt"
  ls checkpoint/*.tar 2>/dev/null | sort > "$before"
  local extra=""
  [ "$mode" = "hybrid" ] && extra="--warmstart-oracle $oracle"
  echo "=== START $tag (env=$env seed=$seed mode=$mode) $(date +%H:%M:%S) ==="
  PYTHONUNBUFFERED=1 timeout "$tlimit" $PY main.py --alg DTPO --env-id "$env" --action-type topology --difficulty 0 \
      --n-envs 4 --seed "$seed" --track False --cuda False \
      --dtpo-iters "$iters" --dtpo-batch "$batch" --tree-max-leaf-nodes "$leaves" \
      --dtpo-rerank-topk 5 $extra \
      > "$folder/train.log" 2>&1
  echo "=== train exit=$? for $tag $(date +%H:%M:%S) ==="

  # locate the new checkpoint and run honest selection (fresh eval, SELECT_EP episodes)
  ls checkpoint/*.tar 2>/dev/null | sort > "/tmp/after_$tag.txt"
  local NEW=$(comm -13 "$before" "/tmp/after_$tag.txt" | head -1)
  if [ -z "$NEW" ]; then echo "!! no checkpoint produced for $tag"; return; fi
  echo "  [select] honest selection over $SELECT_EP ep on $(basename "$NEW")"
  PYTHONUNBUFFERED=1 $PY dtpo_select.py --ckpt "$NEW" --episodes "$SELECT_EP" --topk 5 > "$folder/selection.txt" 2>&1
  grep -E "iter=|HONEST BEST|wrote" "$folder/selection.txt" | sed 's/^/  /'

  bash dtpo_testing/finalize_run.sh "$folder" "$before" "$env $mode seed$seed ($leaves leaves)" \
      2>&1 | grep -vE "FutureWarning|warnings.warn" || echo "!! finalize failed for $tag"

  local ck=$(ls "$folder"/DTPO_*.tar 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    PYTHONUNBUFFERED=1 $PY fidelity_eval.py --tree-ckpt "$ck" --oracle-ckpt "checkpoint/$oracle.tar" \
        --episodes 10 --seed 0 --out "$folder/fidelity.json" 2>&1 | grep -E "\[fidelity\]" || echo "  (fidelity skipped)"
  fi
}

echo "########## BUS36-M (iters=200 batch=8000 leaves=32) ##########"
for s in 0 1 2; do
  run_one "$OUT/bus36M_scratch_seed$s" bus36-M $s scratch "$BUS36_ORACLE" 200 8000 32 18000
  run_one "$OUT/bus36M_hybrid_seed$s"  bus36-M $s hybrid  "$BUS36_ORACLE" 200 8000 32 18000
done

echo "########## BUS14 (iters=150 batch=6000 leaves=16) ##########"
for s in 0 1 2; do
  run_one "$OUT/bus14_scratch_seed$s" bus14 $s scratch "$BUS14_ORACLE" 150 6000 16 7200
  run_one "$OUT/bus14_hybrid_seed$s"  bus14 $s hybrid  "$BUS14_ORACLE" 150 6000 16 7200
done

echo "===== HONEST50 SWEEP COMPLETE $(date +%H:%M:%S) ====="
