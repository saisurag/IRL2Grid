#!/bin/bash

# ./submit_array.sh experiments/experiments_bus5.txt
# ./submit_array.sh experiments/experiments_bus14.txt

LIST=${1:?usage: submit_array.sh <experiments_<env>.txt>}
[ -f "$LIST" ] || { echo "list not found: $LIST"; exit 1; }

N=$(grep -c . "$LIST")
[ "$N" -ge 1 ] || { echo "empty list"; exit 1; }

HPCDIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HPCDIR")"
LISTABS="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"

case "$LIST" in
  *bus5*)    RES="select=1:ncpus=5:mem=8gb"  WALL="12:00:00" ;;
  *bus14*)   RES="select=1:ncpus=5:mem=16gb" WALL="23:00:00" ;;
  *bus36*)   RES="select=1:ncpus=5:mem=32gb" WALL="48:00:00" ;;
  *bus118*)  RES="select=1:ncpus=5:mem=48gb" WALL="48:00:00" ;;
  *) echo "cannot infer env from filename '$LIST'"; exit 1 ;;
esac

NAME="rl2g_$(basename "$LIST" .txt | sed 's/experiments_//')"

if [ "$N" -eq 1 ]; then
    JID=$(qsub -N "$NAME" -l "$RES" -l "walltime=$WALL" \
               -v "LIST=$LISTABS,ROOT=$ROOT,PBS_ARRAY_INDEX=1" "$HPCDIR/pbs_array.sh")
else
    JID=$(qsub -N "$NAME" -J 1-"$N" -l "$RES" -l "walltime=$WALL" \
               -v "LIST=$LISTABS,ROOT=$ROOT" "$HPCDIR/pbs_array.sh")
fi
[ -z "$JID" ] && { echo "qsub failed"; exit 1; }
echo "submitted $NAME: $N subjobs as $JID  ($RES, $WALL)"
echo "monitor:  qstat -t -u \$USER | head -30"
echo "collect:  python collect_results.py"
