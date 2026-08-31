#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_6
ORACLE=checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar
SEEDS="194 144"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/RESULTS.txt

echo "[queue] waiting for bus5 sweep to finish $(date +%T)"
while true; do
  grep -q "SWEEP COMPLETE" "$DIR/sweep.log" 2>/dev/null && break
  if ! ps -eo cmd | grep -q '[p]ython.*main.py --alg DTPO' && \
       [ "$(grep -cE 'scratch_seed|hybrid_seed' "$RES" 2>/dev/null)" -ge 16 ]; then break; fi
  sleep 60
done
echo "[queue] bus5 sweep done -> launching bus14 hybrid runs $(date +%T)"

COMMON="--alg DTPO --env-id bus14 --action-type topology --n-envs 3 \
  --dtpo-iters 50 --dtpo-eval-every 5 --dtpo-batch 20000 --dtpo-eval-total 20 \
  --tree-max-leaf-nodes 16 --dtpo-eta 0.4 --dtpo-ppo-clip 0.2 \
  --dtpo-policy-updates 10 --dtpo-anneal-lr True --verbose True"

job () {
  local seed=$1
  local name=bus14_hybrid_seed$seed
  local out=$DIR/$name; mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START $name $(date +%T) ==="
  $PY main.py $COMMON --seed "$seed" --warmstart-oracle "$ORACLE" > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus14_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$name seed=$seed inloop=NA honest=NA frac100=NA (no ckpt)" >> "$RES"; return; }
  cp "$ck" "$out/ckpt.tar"
  local inbest; inbest=$(grep -oE "best=[0-9.]+%" "$out/train.log" | tail -1 | tr -d 'best=%')
  local hon; hon=$($PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1)
  local rep; rep=$(echo "$hon" | grep "HELD-OUT REPORT" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  echo "$name seed=$seed inloop=${inbest:-NA} honest=${rep:-NA}" >> "$RES"
  # visualizations
  $PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "$name" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree" >> "$out/train.log" 2>&1
  $PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree_landscape" --landscape >> "$out/train.log" 2>&1
  echo "=== DONE $name honest=${rep:-NA}% $(date +%T) ==="
}

echo "########## dtpo_testing_6 bus14 HYBRID (seeds $SEEDS) $(date +%T) ##########"
for s in $SEEDS; do job "$s" & done
wait
echo "===== bus14 HYBRID COMPLETE $(date +%T) ====="
grep "bus14_hybrid" "$RES"
