#!/usr/bin/env bash
set -e

outdir="./executed"
mkdir -p "$outdir"

for nb in 1.8.3.*.ipynb; do
  base="$(basename "${nb%.ipynb}")"
  echo "Executing: $nb -> $outdir/${base}.executed.ipynb"
  /usr/bin/python -m jupyter nbconvert \
    --to notebook --execute "$nb" \
    --output "${base}.executed.ipynb" \
    --output-dir "$outdir" \
    --ExecutePreprocessor.timeout=-1
done

echo "All done."
