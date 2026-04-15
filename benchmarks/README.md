# Benchmarks

Regression harness for the debate MCP server. Runs a suite of fixture debates against the running server, parses the responses, and asserts:

- expected phase markers fire (Phase 0 evidence, Pipeline Phases 1–3, Round 1, Phase 1.5 verification, Round 2, Synthesis)
- total cost is within a per-fixture budget
- total latency is within a per-fixture budget
- preflight gate correctly rejects bad inputs and correctly passes through good inputs
- token counts stay within drift bounds of the saved baseline

## Fixtures

Each fixture is a JSON file in `benchmarks/fixtures/` with this shape:

```json
{
  "name": "short-descriptive-slug",
  "description": "what this fixture exercises",
  "tool": "debate",
  "arguments": { ...the MCP tool call args... },
  "expect": {
    "status": "success" | "rejected",
    "markers": ["Phase 0: Evidence", "Round 1", ...],
    "reject_marker": "PREFLIGHT REQUIRED",
    "max_cost_usd": 0.30,
    "max_duration_sec": 240
  }
}
```

## Running

```bash
# Run one fixture
./benchmarks/run.sh fixtures/valid-structured-decision.json

# Run all fixtures and save to benchmarks/results/YYYY-MM-DD.json
./benchmarks/run.sh --all

# Compare latest run against baseline
./benchmarks/run.sh --compare
```

## Baseline

`benchmarks/baseline.json` holds the canonical expected numbers (cost, tokens, latency, phase markers) for each fixture as of the last time it was updated. Check it into git and regenerate with `./run.sh --save-baseline` whenever the server changes materially.

## Cost note

The `rejection-*` fixtures are free (no LLM calls). The `valid-*` fixtures cost ~$0.20–0.25 each. Running the full suite costs ~$0.50–0.75.
