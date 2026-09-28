#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src deploy
python -m unittest discover -s tests -v

DB_FEATURES=outputs/v3/dino_mean_database.pt
QUERY_FEATURES=outputs/v3/dino_mean_query.pt
if [[ -f "$DB_FEATURES" && -f "$QUERY_FEATURES" ]]; then
  python -m src.evaluate \
    --database "$DB_FEATURES" \
    --query "$QUERY_FEATURES" \
    --top-k 5 --recall-ks 1 5 --precision-k 5 \
    --split-name night_right --min-index 70 --max-index 74 \
    --bootstrap-samples 100
else
  echo "Feature artifacts not present; unit smoke test completed."
fi
