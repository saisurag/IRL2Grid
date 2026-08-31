#!/bin/bash

ENVID=${1:?usage: launch_oracle_chain.sh <bus36-M|bus118-M> [seed] [n_passes]}
SEED=${2:-0}
NJOBS=${3:-5}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-45000000}

HPCDIR="$(cd "$(dirname "$0")" && pwd)"
case "$ENVID" in
  bus36-M)  SCRIPT="$HPCDIR/pbs_ppo_bus36.sh" ;;
  bus118-M) SCRIPT="$HPCDIR/pbs_ppo_bus118.sh" ;;
  *) echo "unknown env '$ENVID' (expected bus36-M|bus118-M)"; exit 1 ;;
esac

if [ ! -f main.py ]; then
    echo "Run this from the RL2Grid repo root (main.py not found in $(pwd))"
    exit 1
fi

DEP=""
for i in $(seq 1 "$NJOBS"); do
    if [ -z "$DEP" ]; then
        JID=$(qsub -v "SEED=${SEED},TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS},USE_HEURISTIC=${USE_HEURISTIC:-False},HEURISTIC_TYPE=${HEURISTIC_TYPE:-idle}" "$SCRIPT")
    else
        JID=$(qsub -v "SEED=${SEED},TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS},USE_HEURISTIC=${USE_HEURISTIC:-False},HEURISTIC_TYPE=${HEURISTIC_TYPE:-idle}" -W depend=afterany:"$DEP" "$SCRIPT")
    fi
    [ -z "$JID" ] && { echo "qsub failed at pass $i"; exit 1; }
    echo "submitted pass $i/$NJOBS: $JID"
    DEP=$JID
done
