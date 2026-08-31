#!/bin/bash

set -e
RUN="$1"; BEFORE="$2"; TITLE="$3"
PY=/home/sai/miniconda3/envs/rl2grid/bin/python
cd /home/sai/RL2Grid_full

ls checkpoint/*.tar 2>/dev/null | sort > /tmp/ckpt_after.txt
# newest-by-MTIME among checkpoints new since BEFORE (NOT alphabetical head -1,
# which picked "DTPO_bus14_..." over "DTPO_bus5_..." since '1'<'5' under
# concurrency). dtpo_select just rewrote this run's own ckpt, so it is newest.
NEWLIST=$(comm -13 "$BEFORE" /tmp/ckpt_after.txt)
NEW=""
[ -n "$NEWLIST" ] && NEW=$(ls -t $NEWLIST 2>/dev/null | head -1)
if [ -z "$NEW" ]; then echo "!! no new checkpoint for $RUN"; exit 1; fi
echo "[finalize] $RUN  <- $NEW"

cp "$NEW" "$RUN/"                         # keep a copy inside the run folder
CK="$RUN/$(basename "$NEW")"

$PY viz_dtpo_tree.py --ckpt "$CK" --out "$RUN/tree" 2>&1 | grep -vE "FutureWarning|warnings.warn"
$PY viz_dtpo_tree.py --ckpt "$CK" --out "$RUN/tree_landscape" --landscape 2>&1 | grep -vE "FutureWarning|warnings.warn"
$PY plot_dtpo_curve.py --log "$RUN/train.log" --out "$RUN/curve.png" --title "$TITLE"
echo "[finalize] $RUN done -> $(ls "$RUN")"
