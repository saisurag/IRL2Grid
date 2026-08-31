#!/bin/bash
#PBS -N rl2grid_array
#PBS -l select=1:ncpus=5:mem=16gb
#PBS -l walltime=23:00:00
#PBS -j oe

: "${LIST:?LIST not set — submit via submit_array.sh}"
: "${ROOT:?ROOT not set — submit via submit_array.sh}"

LINE=$(sed -n "${PBS_ARRAY_INDEX}p" "$LIST")
[ -z "$LINE" ] && { echo "No line ${PBS_ARRAY_INDEX} in $LIST"; exit 1; }

NAME=${LINE%%|*};  REST=${LINE#*|}
SCRIPT=${REST%%|*}; REST=${REST#*|}
HONEST_TOTAL=${REST%%|*}
ARGS=${REST#*|}
ARGS=${ARGS//__ROOT__/$ROOT}

JOBDIR="$ROOT/hpc/runs/$NAME"
mkdir -p "$JOBDIR"
cd "$JOBDIR" || exit 1

eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate rl2grid

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== $PBS_JOBID [$PBS_ARRAY_INDEX] $NAME on $(hostname) | $(date) ==="
echo "=== $SCRIPT $ARGS ==="

if [ -f DONE ]; then echo "=== already DONE — skipping ==="; exit 0; fi

python -u "$ROOT/$SCRIPT" $ARGS > train.log 2>&1
STATUS=$?
echo "=== training exit=$STATUS ==="
tail -5 train.log

CKPT=$(ls -t checkpoint/final_*.tar 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t checkpoint/*.tar 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "=== NO CHECKPOINT — failing ==="
    exit 1
fi

python -u "$ROOT/honest_eval_any.py" --ckpt "$CKPT" --total "$HONEST_TOTAL" > honest_eval.log 2>&1
echo "=== honest eval exit=$? ==="
grep "^RESULT" honest_eval.log

jumps () {
  grep -oE "${1}=[0-9.]+%" train.log | grep -oE "[0-9.]+" | \
    awk 'NR>1{d=$1-prev; if(d<0)d=-d; if(d>20)c++} {prev=$1} END{print c+0}'
}
echo "JUMPS bigjumps=$(jumps survival) candjumps=$(jumps cand)" | tee jumps.txt

grep -q "^RESULT" honest_eval.log && touch DONE
echo "=== $NAME finished $(date) ==="
