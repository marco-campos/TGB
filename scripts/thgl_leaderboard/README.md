# THGL Leaderboard Reproduction Runner

This folder provides a resumable launcher that orchestrates the existing TGB THGL example scripts and keeps cluster-friendly logs/state.

## What it runs

The runner follows the visible leaderboard setup:

- `thgl-software`: `sthn`, `tgn_edge_type`, `tgn`, `edgebank` (unlimited + fixed time window), `recurrencybaseline`
- `thgl-forum`: `sthn`, `tgn_edge_type`, `tgn`, `edgebank` (unlimited + fixed time window), `recurrencybaseline`
- `thgl-github`: `edgebank` only (unlimited + fixed time window)
- `thgl-myket`: `edgebank` only (unlimited + fixed time window)

## Outputs

For each run directory (`scripts/thgl_leaderboard/runs/<run-name>`):

- `state.json`: durable job state (`pending`/`running`/`completed`/`failed`)
- `logs/*.log`: full stdout/stderr per job
- `results_snapshot.jsonl`: one JSON line appended after each successful job
- `envelopes/*_loss_envelope.json`: mean/std loss envelopes for multi-seed models (`tgn`, `tgn_edge_type`, `sthn`)

Model scripts also continue writing their native result files under:

- `examples/linkproppred/<dataset>/saved_results/*.json`
- `examples/linkproppred/<dataset>/saved_results/loss_curves/*.json` (per-seed training curves)
- `examples/linkproppred/<dataset>/saved_models/*` (checkpoints/per-relation files)

## Quick start

From repo root:

```bash
bash scripts/thgl_leaderboard/run_thgl_leaderboard.sh
```

For long cluster runs:

```bash
nohup bash scripts/thgl_leaderboard/run_thgl_leaderboard.sh > thgl_launcher.out 2>&1 &
```

To resume, re-run the same `--run-dir`; completed jobs are skipped automatically.

## Direct Python usage

```bash
python scripts/thgl_leaderboard/run_thgl_leaderboard.py \
  --run-dir runs/my_thgl_exp \
  --num-seeds 5 \
  --seed-start 1
```

Useful flags:

- `--models all|edgebank|tgn|tgn_edge_type|sthn|recurrencybaseline`
- `--datasets thgl-software thgl-forum thgl-github thgl-myket`
- `--continue-on-error true|false`
- `--compile-sthn-sampler true|false`
- `--dry-run`

## Important notes

- `tgn_edge_type` and `tgn` are toggled via `--use_edge_type 1/0` passed to each THGL `tgn.py`.
- STHN requires sampler compilation (`modules/sthn_sampler_setup.py build_ext --inplace`), done automatically unless disabled.
- This runner resumes at the job level. If a single running job is interrupted, that job is re-run on resume.
