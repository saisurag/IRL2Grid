#!/usr/bin/env bash
# dtpo_testing_8: rerun of the dtpo_testing_7 experiments with --dtpo-survival-gate True and --dtpo-replay-iters 4
set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_8
ORACLE_BUS5="checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar"
ORACLE_BUS14="checkpoint/final_PPO_bus14_T_0_0__I__1775940444_3936.tar"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/gatereplay_sweep_RESULTS.txt
: > "$RES"

jumps () {  # file tag
  grep -oE "${2}=[0-9.]+%" "$1" | grep -oE "[0-9.]+" | \
    awk 'NR>1{d=$1-prev; if(d<0)d=-d; if(d>20)c++} {prev=$1} END{print c+0}'
}

run_one () {
  local k=$1 seed=$2
  local out=$DIR/bus5_gatereplay_K${k}_seed${seed}
  mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START bus5 K=$k seed=$seed $(date +%T) ==="
  $PY main.py --alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
    --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
    --dtpo-norm-adv True --dtpo-eval-every 1 --dtpo-eval-total 20 \
    --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
    --dtpo-structure-refit-every "$k" \
    --dtpo-survival-gate True --dtpo-replay-iters 4 \
    --tree-max-leaf-nodes 16 \
    --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
    --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
    --warmstart-oracle $ORACLE_BUS5 --warmstart-temperature 0.5 --warmstart-dagger-rounds 0 \
    --seed "$seed" --verbose True > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus5_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "K=$k seed=$seed honest=NA bigjumps=NA candjumps=NA (no ckpt)" >> "$RES"
    echo "=== FAIL bus5 K=$k seed=$seed $(date +%T) ==="
    return
  fi
  cp "$ck" "$out/ckpt.tar"
  $PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 > "$out/honest_eval.log" 2>&1
  local rep; rep=$(grep "HELD-OUT REPORT" "$out/honest_eval.log" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  local bj cj; bj=$(jumps "$out/train.log" survival); cj=$(jumps "$out/train.log" cand)
  echo "K=$k seed=$seed honest=${rep:-NA} bigjumps=${bj:-NA} candjumps=${cj:-NA}" >> "$RES"
  echo "=== DONE bus5 K=$k seed=$seed honest=${rep:-NA} bigjumps=${bj:-NA} candjumps=${cj:-NA} $(date +%T) ==="
}

run_round () {
  local k=$1; shift
  echo "########## ROUND K=$k: seeds ($*) $(date +%T) ##########"
  for seed in "$@"; do run_one "$k" "$seed" & done
  wait
  echo "===== ROUND K=$k COMPLETE $(date +%T) ====="
}

run_round 1  100 101 102 103
run_round 3  100 101 102 103
run_round 5  100 101 102 103
run_round 10 100 101 102 103

echo ""
echo "########## bus5 SUMMARY per K ##########"
for k in 1 3 5 10; do
  awk -v kk="K=$k " '
    index($0,kk)==1 {
      for (i=1;i<=NF;i++){
        if ($i ~ /^bigjumps=/){v=substr($i,10); if(v!="NA"){bs+=v;bn++}}
        if ($i ~ /^candjumps=/){v=substr($i,11); if(v!="NA"){cs+=v;cn++}}
        if ($i ~ /^honest=/){v=substr($i,8); if(v!="NA"){hs+=v;hn++}}
      }
    }
    END { printf "K='"$k"': mean bigjumps=%.1f candjumps=%.1f honest=%.1f%% (n=%d)\n",
          (bn?bs/bn:-1),(cn?cs/cn:-1),(hn?hs/hn:0),hn }
  ' "$RES"
done
echo "===== BUS5 GATE+REPLAY SWEEP COMPLETE $(date +%T) ====="

# bus14 seed 100, combined
out=$DIR/bus14_gatereplay_K5_seed100
mkdir -p "$out"
marker=$out/.start; : > "$marker"
echo "=== START bus14 K=5 seed=100 $(date +%T) ==="
$PY main.py --alg DTPO --env-id bus14 --action-type topology --n-envs 3 \
  --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
  --dtpo-norm-adv True --dtpo-eval-every 1 --dtpo-eval-total 40 \
  --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
  --dtpo-structure-refit-every 5 \
  --dtpo-survival-gate True --dtpo-replay-iters 4 \
  --tree-max-leaf-nodes 16 \
  --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
  --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
  --warmstart-oracle $ORACLE_BUS14 --warmstart-temperature 0.5 --warmstart-dagger-rounds 0 \
  --seed 100 --verbose True > "$out/train.log" 2>&1
ck=$(find checkpoint -name "DTPO_bus14_T_100_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
if [ -n "$ck" ]; then
  cp "$ck" "$out/ckpt.tar"
  $PY honest_eval.py --ckpt "$out/ckpt.tar" --total 80 > "$out/honest_eval_total80.log" 2>&1
  echo "bus14 K=5 seed=100 honest80=$(grep 'HELD-OUT REPORT' "$out/honest_eval_total80.log" | grep -oE 'survival = [0-9.]+%')" >> "$RES"
else
  echo "bus14 K=5 seed=100 honest80=NA (no ckpt)" >> "$RES"
fi
# old dtpo_testing_7 baseline re-scored
$PY honest_eval.py --ckpt dtpo_testing_7/bus14_t0.5_d0_seed100/ckpt.tar --total 80 \
  > "$DIR/old_bus14_baseline_honest_total80.log" 2>&1
echo "bus14 OLD baseline honest80=$(grep 'HELD-OUT REPORT' "$DIR/old_bus14_baseline_honest_total80.log" | grep -oE 'survival = [0-9.]+%')" >> "$RES"
echo "===== ALL RUNS COMPLETE $(date +%T) ====="
