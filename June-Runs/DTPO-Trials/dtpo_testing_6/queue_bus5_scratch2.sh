#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_6
SEEDS="265 296"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/RESULTS.txt

echo "[queue-scratch2] waiting for bus5 sweep to finish $(date +%T)"
while true; do
  grep -q "SWEEP COMPLETE" "$DIR/sweep.log" 2>/dev/null && break
  if ! ps -eo cmd | grep -q '[p]ython.*main.py --alg DTPO --env-id bus5' && \
       [ "$(grep -cE 'scratch_seed|hybrid_seed' "$RES" 2>/dev/null)" -ge 16 ]; then break; fi
  sleep 60
done
echo "[queue-scratch2] sweep done -> launching extra scratch runs $(date +%T)"

COMMON="--alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
  --dtpo-iters 50 --dtpo-eval-every 5 --dtpo-batch 20000 --dtpo-eval-total 20 \
  --tree-max-leaf-nodes 16 --dtpo-eta 0.4 --dtpo-ppo-clip 0.2 \
  --dtpo-policy-updates 10 --dtpo-anneal-lr True --verbose True"

job () {            # seed
  local seed=$1
  local name=scratch_seed$seed
  local out=$DIR/$name; mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START $name $(date +%T) ==="
  $PY main.py $COMMON --seed "$seed" > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus5_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$name seed=$seed inloop=NA honest=NA (no ckpt)" >> "$RES"; return; }
  cp "$ck" "$out/ckpt.tar"
  local inbest; inbest=$(grep -oE "best=[0-9.]+%" "$out/train.log" | tail -1 | tr -d 'best=%')
  local hon; hon=$($PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1)
  local rep; rep=$(echo "$hon" | grep "HELD-OUT REPORT" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  echo "$name seed=$seed inloop=${inbest:-NA} honest=${rep:-NA}" >> "$RES"
  $PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "$name" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree_landscape" --landscape >> "$out/train.log" 2>&1
  echo "=== DONE $name honest=${rep:-NA}% $(date +%T) ==="
}

echo "########## dtpo_testing_6 EXTRA bus5 scratch (seeds $SEEDS) $(date +%T) ##########"
for s in $SEEDS; do job "$s" & done
wait
echo "===== EXTRA bus5 scratch COMPLETE $(date +%T) ====="
grep -E "scratch_seed(265|296)" "$RES"
