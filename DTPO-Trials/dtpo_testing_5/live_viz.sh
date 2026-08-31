#!/usr/bin/env bash

PY=/home/sai/miniconda3/envs/rl2grid/bin/python
DIR=/home/sai/RL2Grid_full/dtpo_testing_5
cd /home/sai/RL2Grid_full
sleep 120
empty=0
while true; do
  for d in bus5_scratch_seed0 bus5_scratch_seed1 bus5_hybrid_seed0; do
    log=$DIR/$d/train.log
    [ -f "$log" ] && grep -q "DTPO\] it=" "$log" 2>/dev/null && \
      $PY plot_dtpo_curve.py --log "$log" --out "$DIR/$d/curve.png" --title "$d (live)" \
        >/dev/null 2>&1
  done
  if pgrep -f 'main.py --alg DTPO' >/dev/null 2>&1; then empty=0; else empty=$((empty+1)); fi
  [ "$empty" -ge 3 ] && { echo "[live_viz] all runs done $(date +%T)"; break; }
  sleep 180
done
