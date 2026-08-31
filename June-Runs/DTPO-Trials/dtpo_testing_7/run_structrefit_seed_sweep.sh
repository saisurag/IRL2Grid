#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_7
ORACLE="checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar"
cd "$ROOT"; export PYTHONUNBUFFERED=1
RES=$DIR/structrefit_sweep_RESULTS.txt
: > "$RES"

run_one () {
  local k=$1 seed=$2
  local out=$DIR/bus5_structrefit_K${k}_seed${seed}
  mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START K=$k seed=$seed $(date +%T) ==="
  $PY main.py --alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
    --dtpo-iters 150 --dtpo-batch 10000 --dtpo-eta 1.0 --dtpo-gamma 0.99 --dtpo-gae-lambda 0.95 \
    --dtpo-norm-adv True --dtpo-eval-every 1 --dtpo-eval-total 20 \
    --dtpo-ppo-clip 0.0 --dtpo-policy-updates 1 --dtpo-anneal-lr False \
    --dtpo-structure-refit-every "$k" \
    --tree-max-leaf-nodes 16 \
    --critic-layers 64 64 --critic-act-fn tanh --critic-lr 2.5e-4 \
    --critic-epochs 4 --critic-batch 64 --max-grad-norm 0.5 \
    --warmstart-oracle $ORACLE --warmstart-temperature 0.5 --warmstart-dagger-rounds 0 \
    --seed "$seed" --verbose True > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus5_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "K=$k seed=$seed honest=NA bigjumps=NA (no ckpt)" >> "$RES"
    echo "=== FAIL K=$k seed=$seed $(date +%T) ==="
    return
  fi
  cp "$ck" "$out/ckpt.tar"
  $PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 > "$out/honest_eval.log" 2>&1
  local rep; rep=$(grep "HELD-OUT REPORT" "$out/honest_eval.log" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  local bigjumps; bigjumps=$(grep -oE "survival=[0-9.]+%" "$out/train.log" | grep -oE "[0-9.]+" | \
    awk 'NR>1{d=$1-prev; if(d<0)d=-d; if(d>20)c++} {prev=$1} END{print c+0}')
  echo "K=$k seed=$seed honest=${rep:-NA} bigjumps=${bigjumps:-NA}" >> "$RES"
  echo "=== DONE K=$k seed=$seed honest=${rep:-NA} bigjumps=${bigjumps:-NA} $(date +%T) ==="
}

run_round () {
  local k=$1; shift
  echo "########## ROUND K=$k: seeds ($*) $(date +%T) ##########"
  for seed in "$@"; do run_one "$k" "$seed" & done
  wait
  echo "===== ROUND K=$k COMPLETE $(date +%T) ====="
}

run_round 1  101 102 103
run_round 3  100 101 102 103
run_round 5  100 101 102 103
run_round 10 101 102 103


for pre in "1:100" "10:100"; do
  k=${pre%%:*}; seed=${pre##*:}
  out=$DIR/bus5_structrefit_K${k}_seed${seed}
  if [ -f "$out/honest_eval.log" ]; then
    rep=$(grep "HELD-OUT REPORT" "$out/honest_eval.log" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
    bigjumps=$(grep -oE "survival=[0-9.]+%" "$out/train.log" | grep -oE "[0-9.]+" | \
      awk 'NR>1{d=$1-prev; if(d<0)d=-d; if(d>20)c++} {prev=$1} END{print c+0}')
    echo "K=$k seed=$seed honest=${rep:-NA} bigjumps=${bigjumps:-NA} (pre-existing)" >> "$RES"
  fi
done

echo ""
echo "########## SUMMARY: mean bigjump-count and honest survival per K ##########"
for k in 1 3 5 10; do
  awk -v kk="K=$k " '
    index($0,kk)==1 {
      for (i=1;i<=NF;i++){
        if ($i ~ /^bigjumps=/){bj=substr($i,10); if(bj!="NA"){sum+=bj;n++}}
        if ($i ~ /^honest=/){h=substr($i,8); if(h!="NA"){hsum+=h;hn++}}
      }
    }
    END {
      if (n>0) printf "K=%s: mean bigjumps=%.1f (n=%d runs)   mean honest=%.1f%% (n=%d)\n", "'"$k"'", sum/n, n, (hn>0?hsum/hn:0), hn
      else print "K='"$k"': no data"
    }
  ' "$RES"
done
echo "===== STRUCTREFIT SEED SWEEP COMPLETE $(date +%T) ====="
