#!/bin/bash
# Opt-in check of the current paws code against outputs of the real analysis (not run by pytest).
# Re-creates a few known outputs in a temporary home (real inputs linked read-only) and diffs them:
#   followup-v3-2 DAGs of bands 20 22 23 31, followup-v3-1 outlier files of bands 50 120.
#
# Usage (from paws/):  bash tests/check_against_real.sh /home/hoitim.cheung/galacticCenter
set -u
ANALYSIS=${1:?usage: check_against_real.sh <analysis dir with config/ results/ condorFiles/>}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"/{results,dagFiles,condorFiles}
cp -r "$ANALYSIS/config" "$WORK/config"
sed -i "s#^home_dir: .*#home_dir: $WORK/#" "$WORK/config/config.yaml"
for stage in search-0 followup-1 followup-2 followup-v2-1 followup-v2-2 followup-v3-1 \
             injections-v2-0 injections-v2-1 injections-v2-2 injections-v2-3; do
  ln -s "$ANALYSIS/results/$stage" "$WORK/results/$stage"
done
export PAWS_CONFIG_DIR=$WORK/config
status=0

uv run paws dag followup-v3-2 --bands 20 22 23 31 > /dev/null 2>&1
for band in 20 22 23 31; do
  new=$(cd "$WORK/condorFiles/followup-v3-2/GalacticCenter/$band" && find . -type f | sort | xargs cat \
        | sed "s#$WORK/#$ANALYSIS/#g; s#/0\.0 #/0 #g")
  old=$(cd "$ANALYSIS/condorFiles/followup-v3-2/GalacticCenter/$band" && find . -type f | sort | xargs cat)
  if [ "$new" == "$old" ]; then echo "followup-v3-2 $band Hz DAG: identical"; else echo "followup-v3-2 $band Hz DAG: DIFF"; status=1; fi
done

rm "$WORK/results/followup-v3-1" && mkdir "$WORK/results/followup-v3-1"
uv run paws outliers followup-v3-1 --bands 50 120 > /dev/null 2>&1
for band in 50 120; do
  for new in "$WORK"/results/followup-v3-1/GalacticCenter/*/$band/Outliers/*.fts; do
    old=${new/#$WORK/$ANALYSIS}
    if uv run fitsdiff -q "$new" "$old" > /dev/null 2>&1; then echo "$(basename "$new"): identical"
    else echo "$(basename "$new"): DIFF"; status=1; fi
  done
done
exit $status
