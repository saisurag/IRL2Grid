#!/usr/bin/env bash

set -u
PY="/home/sai/miniconda3/envs/rl2grid/bin/python -u"
ROOT=/home/sai/RL2Grid_full
DIR=$ROOT/dtpo_testing_6
ORACLE=checkpoint/final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar
MAX=3
cd "$ROOT"; export PYTHONUNBUFFERED=1
mkdir -p "$DIR"
RES=$DIR/RESULTS.txt
: > "$RES"

COMMON="--alg DTPO --env-id bus5 --action-type topology --n-envs 3 \
  --dtpo-iters 50 --dtpo-eval-every 5 --dtpo-batch 20000 --dtpo-eval-total 20 \
  --tree-max-leaf-nodes 16 --dtpo-eta 0.4 --dtpo-ppo-clip 0.2 \
  --dtpo-policy-updates 10 --dtpo-anneal-lr True --verbose True"

job () {
  local name=$1 seed=$2 extra=$3
  local out=$DIR/$name; mkdir -p "$out"
  local marker=$out/.start; : > "$marker"
  echo "=== START $name (seed=$seed) $(date +%T) ==="
  $PY main.py $COMMON --seed "$seed" $extra > "$out/train.log" 2>&1
  local ck; ck=$(find checkpoint -name "DTPO_bus5_T_${seed}_0__*.tar" -newer "$marker" 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$name seed=$seed inloop=NA honest=NA frac100=NA (no ckpt)" >> "$RES"; return; }
  cp "$ck" "$out/ckpt.tar"
  local inbest; inbest=$(grep -oE "best=[0-9.]+%" "$out/train.log" | tail -1 | tr -d 'best=%')
  local hon; hon=$($PY honest_eval.py --ckpt "$out/ckpt.tar" --total 20 2>&1)
  local rep; rep=$(echo "$hon" | grep "HELD-OUT REPORT" | grep -oE "survival = [0-9.]+%" | grep -oE "[0-9.]+")
  local frac; frac=$(echo "$hon" | grep -oE "frac@100% [0-9]+%" | grep -oE "[0-9]+" | head -1)
  echo "$name seed=$seed inloop=${inbest:-NA} honest=${rep:-NA} frac100=${frac:-NA}" >> "$RES"
  $PY plot_dtpo_curve.py --log "$out/train.log" --out "$out/curve.png" --title "$name" >/dev/null 2>&1
  echo "=== DONE $name honest=${rep:-NA}% frac100=${frac:-NA}% $(date +%T) ==="
}

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n; done; }

echo "########## dtpo_testing_6 bus5 SEED SWEEP $(date +%T) ##########"
for s in 0 1 2 3 4 5 6 7;               do throttle; job "scratch_seed$s" "$s" "" & done
for s in 100 101 102 103 104 105 106 107; do throttle; job "hybrid_seed$s" "$s" "--warmstart-oracle $ORACLE" & done
wait

echo "===== SWEEP COMPLETE $(date +%T) ====="
$PY - "$RES" <<'PYEOF'
import sys,re
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
def parse(r):
    d={k:v for k,v in (x.split('=') for x in r if '=' in x)}
    return r[0], d
def num(x):
    try: return float(x)
    except: return None
agg={'scratch':[], 'hybrid':[]}
for r in rows:
    name=r[0]; d={x.split('=')[0]:x.split('=')[1] for x in r if '=' in x}
    grp='scratch' if name.startswith('scratch') else 'hybrid'
    agg[grp].append((name, num(d.get('honest')), num(d.get('frac100'))))
print("\n==== TALLY: clean 100% = honest==100 and frac@100%==100 ====")
for grp in ('scratch','hybrid'):
    items=agg[grp]
    clean=[n for n,h,f in items if h==100.0 and f==100.0]
    high =[n for n,h,f in items if h is not None and h>=99.0]
    print(f"\n[{grp}] n={len(items)}  clean100={len(clean)}/{len(items)}  (>=99%: {len(high)})")
    for n,h,f in sorted(items):
        tag='  <-- CLEAN 100' if (h==100.0 and f==100.0) else ''
        print(f"   {n:20s} honest={h}  frac100={f}{tag}")
PYEOF
cat "$DIR/RESULTS.txt"
