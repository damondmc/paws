#!/bin/bash
# Move today's v2 followup-1 outliers into their own stage dir (followup-v2-1) and restore followup-1.
#  1. new top-1-per-parent files (written by the run that created $STAMP) -> followup-v2-1/<f>/Outliers/, renamed
#  2. followup-1/<f>/Outliers/v2_top10/  -> followup-v2-1/<f>/Outliers/top10/
#  3. followup-1/<f>/Outliers/old_threshold/* -> back to followup-1/<f>/Outliers/ (original files)
set -u
STAMP=$1   # file created when the top-1 recompute started
SRC=/home/hoitim.cheung/galacticCenter/results/followup-1/GalacticCenter/C00-C01_Gated_G02_1800s
DST=/home/hoitim.cheung/galacticCenter/results/followup-v2-1/GalacticCenter/C00-C01_Gated_G02_1800s
n_new=0; n_top10=0; n_rest=0
for d in $SRC/*/Outliers; do
    f=$(basename $(dirname $d))
    out=$DST/$f/Outliers
    for kind in outlier outlier_clustered; do
        src=$d/GalacticCenter_followup-1_TCoh10_O2_${f}Hz_${kind}.fts
        if [ -e "$src" ] && [ "$src" -nt "$STAMP" ]; then
            mkdir -p $out
            mv -- "$src" "$out/GalacticCenter_followup-v2-1_TCoh10_O2_${f}Hz_${kind}.fts" && n_new=$((n_new + 1))
        fi
    done
    if [ -d $d/v2_top10 ]; then
        mkdir -p $out
        mv -- $d/v2_top10 $out/top10 && n_top10=$((n_top10 + 1))
    fi
    if [ -d $d/old_threshold ]; then
        for o in $d/old_threshold/*.fts; do
            [ -e "$d/$(basename $o)" ] && { echo "SKIP restore, exists: $d/$(basename $o)"; continue; }
            mv -- "$o" "$d/" && n_rest=$((n_rest + 1))
        done
        rmdir $d/old_threshold 2>/dev/null || echo "old_threshold not empty: $d"
    fi
done
echo "moved new files: $n_new; top10 dirs: $n_top10; restored originals: $n_rest"
