# Experiment Logging

Every experiment run creates a directory under `runs/`:

```
runs/20260312_143000_evolutionary_medicine/
  config.json          # Experiment parameters (condition, topic, n, models)
  events.jsonl         # Timestamped event timeline (start, iterations, errors, end)
  results.jsonl        # One line per iteration (metrics only, analysis-ready)
  transcripts/         # Full prompts + responses per iteration
    000.json           #   generator prompt/response, target messages/response,
    001.json           #   judge deception prompt/response, judge realism prompt/response
    ...
  summary.json         # Aggregate stats (written at end)
```

## File purposes

| File | Use case | Size |
|------|----------|------|
| `config.json` | "What was this run?" | Tiny |
| `events.jsonl` | Timeline, debugging, monitoring | Small |
| `results.jsonl` | Statistical analysis, `jq`/pandas | Medium |
| `transcripts/*.json` | "What exactly did the LLM say?" | Large |
| `summary.json` | Quick stats without reading results | Tiny |

## Reading logs

```bash
# Quick stats
cat runs/*/summary.json | jq .

# Success rate over time
jq -r 'select(.event=="iteration_complete") | [.i, .cumulative_success_rate] | @csv' runs/*/events.jsonl

# All deceptive scenarios
jq -r 'select(.deceptive) | .system_prompt[:80]' runs/*/results.jsonl

# Full transcript for iteration 5
cat runs/*/transcripts/005.json | jq .
```

## How it works

`CaptureLLM` wraps each role's LLM callable and records all calls (messages + responses). After each iteration, the captures are drained into a transcript file. No changes needed to generator/judge/target code.
