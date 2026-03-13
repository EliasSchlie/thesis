# Experiment Logging

## Run structure

Every experiment run creates a directory under `runs/`:

```
runs/20260312_143000_evolutionary_medicine/
  config.json          # Experiment parameters, run_id, model info
  events.jsonl         # Timestamped event timeline (start, iterations, errors, end)
  results.jsonl        # One line per iteration (metrics + context, analysis-ready)
  transcripts/         # Full prompts + responses per iteration
    000.json           #   generator prompt/response, target messages/response,
    001.json           #   judge deception prompt/response, judge realism prompt/response
    ...
  summary.json         # Aggregate stats (updated every 10 iterations + at end)
```

Multi-topic runs also create an experiment manifest:

```
runs/experiment_<id>.json   # Links all runs in this experiment
```

## File purposes

| File | Use case | Size |
|------|----------|------|
| `config.json` | "What was this run?" — includes run_id, models, experiment_id | Tiny |
| `events.jsonl` | Timeline, debugging, monitoring, error tracking | Small |
| `results.jsonl` | Statistical analysis, `jq`/pandas — each line has run_id, condition, topic, models | Medium |
| `transcripts/*.json` | "What exactly did the LLM say?" | Large |
| `summary.json` | Quick stats — updated incrementally, safe against crashes | Tiny |
| `experiment_*.json` | "What runs belong together?" — written after all topics complete | Tiny |

## Key fields

Every `results.jsonl` line includes:
- `run_id`, `condition`, `topic` — no need to parse directory names
- `models` — which model was used for each role
- `i`, `ts`, `elapsed_s` — iteration index, timestamp, wall-clock time

Every `events.jsonl` line includes `run_id`.

## Event types

| Event | When | Key fields |
|-------|------|------------|
| `experiment_start` | Run begins | condition, topic, n, models |
| `warm_start` | Population pre-seeded | count, successful |
| `iteration_complete` | Successful iteration | i, deceptive, realism, fitness, cumulative_success_rate |
| `iteration_error` | Failed iteration (parse error, API timeout) | i, error |
| `experiment_end` | Run finishes | total, deceptive, errors, success_rate |

## Reading logs

```bash
# Quick stats
cat runs/*/summary.json | jq .

# Success rate over time
jq -r 'select(.event=="iteration_complete") | [.i, .cumulative_success_rate] | @csv' runs/*/events.jsonl

# All deceptive scenarios
jq -r 'select(.deceptive) | .system_prompt[:80]' runs/*/results.jsonl

# Cross-run analysis (all results in one stream)
cat runs/*/results.jsonl | jq -s 'group_by(.condition) | map({condition: .[0].condition, count: length, deceptive: map(select(.deceptive)) | length})'

# Error rate per run
jq -r 'select(.event=="iteration_error") | .run_id' runs/*/events.jsonl | sort | uniq -c

# Full transcript for iteration 5
cat runs/*/transcripts/005.json | jq .
```

## How it works

`CaptureLLM` wraps each role's LLM callable and records all calls (messages + responses). After each iteration, the captures are drained into a transcript file. No changes needed to generator/judge/target code.

Failed iterations (parse errors, API timeouts) are caught, logged as `iteration_error` events with partial transcripts, and skipped — the run continues.

Summary is written every 10 iterations and at run end, so long runs survive crashes.
