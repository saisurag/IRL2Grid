#!/usr/bin/env bash
# Hyperparameter search: can bus5 HYBRID (warm-start DTPO from a distilled PPO oracle tree also reach clean honest 100%, the way bus5 scratch DTPO does

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_7
ORACLE="checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/RESULTS.txt
: > "$RES"
MAXJOBS=4

PAPER_COMMON="--alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
  --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
  --dtpo-norm-adv True --dtpo-eval-every 10 --dtpo-eval-total 20 \
  --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
  --tree-max-leaf-nodes 16 \
  --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
  --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
  --warmstart-oracle $ORACLE --verbose True"

declare -A TEMP DAGGER
CFGS="t0.5_d0 t1.0_d0 t2.0_d0 t0.5_d2 t1.0_d2 t2.0_d2"
TEMP[t0.5_d0]=0.5;  DAGGER[t0.5_d0]=0
TEMP[t1.0_d0]=1.0;  DAGGER[t1.0_d0]=0
TEMP[t2.0_d0]=2.0;  DAGGER[t2.0_d0]=0
TEMP[t0.5_d2]=0.5;  DAGGER[t0.5_d2]=2
TEMP[t1.0_d2]=1.0;  DAGGER[t1.0_d2]=2
TEMP[t2.0_d2]=2.0;  DAGGER[t2.0_d2]=2

job () {
  local cfg=$1 seed=$2
  local temp=${TEMP[$cfg]} dagger=${DAGGER[$cfg]}
  local name=${cfg}_seed${seed}
  local out=$DIR/$name; mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START $name $(date +%T) ==="
  $PY main.py $PAPER_COMMON --seed "$seed" \
    --warmstart-temperature "$temp" --warmstart-dagger-rounds "$dagger" \
    > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus5_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "$name cfg=$cfg seed=$seed temp=$temp dagger=$dagger inloop=NA honest=NA" >> "$RES"
    echo "=== FAIL $name (no ckpt) $(date +%T) ==="
    return
  fi
  cp "$ck" "$out/ckpt.tar"
  local inbest; inbest=$(grep -oE "best=[0-9.]+%" "$out/train.log" | tail -1 | tr -d 'best=%')
  local hon; hon=$($PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1)
  echo "$hon" > "$out/honest_eval.log"
  local rep; rep=$(echo "$hon" | grep "HELD-OUT REPORT" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  echo "$name cfg=$cfg seed=$seed temp=$temp dagger=$dagger inloop=${inbest:-NA} honest=${rep:-NA}" >> "$RES"
  $PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "$name" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree" >> "$out/train.log" 2>&1
  echo "=== DONE $name honest=${rep:-NA}% $(date +%T) ==="
}

run_batch () {
  local n=0
  for spec in "$@"; do
    local cfg=${spec%%:*} seed=${spec##*:}
    job "$cfg" "$seed" &
    n=$((n+1))
    if [ "$n" -ge "$MAXJOBS" ]; then wait -n; fi
  done
  wait
}

echo "########## dtpo_testing_7 STAGE 1: 6 configs x seeds(100,101) $(date +%T) ##########"
stage1_specs=()
for cfg in $CFGS; do
  for seed in 100 101; do stage1_specs+=("$cfg:$seed"); done
done
run_batch "${stage1_specs[@]}"
echo "===== STAGE 1 COMPLETE $(date +%T) ====="
cat "$RES"

echo ""
echo "########## Selecting STAGE 2 finalists (top-2 by mean honest survival) ##########"
mapfile -t ranked < <(awk '
  { for (i=1;i<=NF;i++){ if ($i ~ /^cfg=/) cfg=substr($i,5); if ($i ~ /^honest=/) h=substr($i,8) }
    if (h != "NA" && h != "") { sum[cfg]+=h; n[cfg]++ } }
  END { for (c in sum) printf "%.4f %s\n", sum[c]/n[c], c }
' "$RES" | sort -rn)
echo "Stage-1 ranking (mean honest %, config):"
printf '%s\n' "${ranked[@]}"
FINALISTS=("$(echo "${ranked[0]}" | awk '{print $2}')" "$(echo "${ranked[1]}" | awk '{print $2}')")
echo "Finalists: ${FINALISTS[*]}"

echo ""
echo "########## dtpo_testing_7 STAGE 2: finalists x seeds(102,103,104,105) $(date +%T) ##########"
stage2_specs=()
for cfg in "${FINALISTS[@]}"; do
  for seed in 102 103 104 105; do stage2_specs+=("$cfg:$seed"); done
done
run_batch "${stage2_specs[@]}"
echo "===== STAGE 2 COMPLETE $(date +%T) ====="

echo ""
echo "########## FINAL SUMMARY (6 seeds per finalist: 100,101,102,103,104,105) ##########"
for cfg in "${FINALISTS[@]}"; do
  echo "--- $cfg (temp=${TEMP[$cfg]} dagger=${DAGGER[$cfg]}) ---"
  grep "cfg=$cfg " "$RES"
  awk -v c="cfg=$cfg " '
    index($0,c) { for (i=1;i<=NF;i++){ if ($i ~ /^honest=/) h=substr($i,8);
      if (h!="NA" && h!="") { sum+=h; n++; if (h+0>=99) clean++ } } }
    END { if (n>0) printf "  mean honest=%.1f%%  clean-100 rate=%d/%d\n", sum/n, clean+0, n; else print "  no successful runs" }
  ' "$RES"
done
echo "===== SWEEP COMPLETE $(date +%T) ====="
