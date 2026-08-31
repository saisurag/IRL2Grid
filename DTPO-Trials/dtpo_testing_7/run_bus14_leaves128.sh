#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_7
ORACLE="checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar"
cd "$ROOT"; export PYTHONUNBUFFERED=1

echo "[queue-128leaf] waiting for the 32/64-leaf sweep to finish $(date +%T)"
while [ ! -f "$DIR/bus14_t0.5_d0_seed100_leaves64_evalevery1/honest_eval.log" ] \
      || ! grep -q "HELD-OUT REPORT" "$DIR/bus14_t0.5_d0_seed100_leaves64_evalevery1/honest_eval.log" 2>/dev/null; do
  sleep 30
done
echo "[queue-128leaf] 64-leaf run done -> launching 128-leaf run $(date +%T)"

out=$DIR/bus14_t0.5_d0_seed100_leaves128_evalevery1
marker=$out/.start; : > "$marker"
echo "=== START leaves=128 $(date +%T) ==="
$PY main.py --alg DTPO --env-id bus14 --action-type topology --n-envs 3 \
  --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
  --dtpo-norm-adv True --dtpo-eval-every 1 --dtpo-eval-total 20 \
  --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
  --tree-max-leaf-nodes 128 \
  --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
  --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
  --warmstart-oracle $ORACLE --warmstart-temperature 0.5 --warmstart-dagger-rounds 0 \
  --seed 100 --verbose True > "$out/train.log" 2>&1
ck=$(find checkpoint -name "DTPO_bus14_T_100_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
if [ -z "$ck" ]; then
  echo "=== FAIL leaves=128 (no ckpt) $(date +%T) ==="
  exit 1
fi
cp "$ck" "$out/ckpt.tar"
$PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1 | tee "$out/honest_eval.log"
$PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "bus14_leaves128" >> "$out/train.log" 2>&1
$PY viz_dtpo_tree.py --ckpt "$out/ckpt.tar" --out "$out/tree" >> "$out/train.log" 2>&1
echo "=== DONE leaves=128 $(date +%T) ==="
