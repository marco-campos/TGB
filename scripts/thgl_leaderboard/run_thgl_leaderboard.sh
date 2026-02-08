#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-thgl_full_$(date +%Y%m%d_%H%M%S)}"

python scripts/thgl_leaderboard/run_thgl_leaderboard.py \
  --run-dir "runs/${RUN_NAME}" \
  --datasets thgl-software thgl-forum thgl-github thgl-myket \
  --models all \
  --num-seeds "${NUM_SEEDS:-5}" \
  --seed-start "${SEED_START:-1}" \
  --compile-sthn-sampler "${COMPILE_STHN_SAMPLER:-true}" \
  --continue-on-error "${CONTINUE_ON_ERROR:-true}" \
  "$@"
