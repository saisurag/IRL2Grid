#!/bin/bash
#PBS -N PPO_bus36M
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe

ENVID=bus36-M
SEED=${SEED:-0}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-45000000}
USE_HEURISTIC=${USE_HEURISTIC:-False}
HEURISTIC_TYPE=${HEURISTIC_TYPE:-idle}

cd "${PBS_O_WORKDIR}" || exit 1

eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate rl2grid

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== $PBS_JOBID on $(hostname) | PPO $ENVID seed=$SEED total=$TOTAL_TIMESTEPS heur=$USE_HEURISTIC:$HEURISTIC_TYPE | $(date) ==="

RUN_ARGS=(--alg PPO --env-id "$ENVID" --action-type topology
          --difficulty 0 --use-heuristic "$USE_HEURISTIC" --heuristic-type "$HEURISTIC_TYPE"
          --n-envs 10 --optimize-mem True
          --total-timesteps "$TOTAL_TIMESTEPS"
          --seed "$SEED"
          --checkpoint True --track False
          --time-limit 2700)

FINAL=$(ls checkpoint/ 2>/dev/null | grep "^final_PPO_${ENVID}_T_${SEED}_" | head -1)
if [ -n "$FINAL" ]; then
    echo "=== Training already complete: checkpoint/$FINAL — nothing to do ==="
    exit 0
fi

CKPT=$(ls -t checkpoint/PPO_"${ENVID}"_T_"${SEED}"_*.tar 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    BK=$(ls -t checkpoint/backup_PPO_"${ENVID}"_T_"${SEED}"_*.tar 2>/dev/null | head -1)
    if [ -n "$BK" ]; then
        BKBASE=$(basename "$BK")
        CKPT="checkpoint/${BKBASE#backup_}"
        echo "=== No live checkpoint; restoring from $BK ==="
        cp "$BK" "$CKPT"
    fi
fi

if [ -n "$CKPT" ]; then
    CKPTBASE=$(basename "$CKPT")
    NAME="${CKPTBASE%.tar}"
    echo "=== Resuming from $NAME ==="
    cp "checkpoint/$CKPTBASE" "checkpoint/backup_${CKPTBASE}"
    python -u main.py "${RUN_ARGS[@]}" --resume-run-name "$NAME"
else
    echo "=== Fresh start (PPO $ENVID seed $SEED) ==="
    python -u main.py "${RUN_ARGS[@]}"
fi

echo "=== Pass finished at $(date) — grep SPS above to size the chain: passes ~= total / SPS / 3600 / 45 ==="
