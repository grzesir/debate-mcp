# Thinking Tools MCP

**Three thinking tools that help you make better decisions.** An MCP server for Claude Code with adversarial debate, problem reframing, and divergent idea generation — all grounded in live web search.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![npm](https://img.shields.io/npm/v/debate-mcp)](https://www.npmjs.com/package/debate-mcp)

## The Pipeline

```
reframe → diverge → debate
  (see)    (imagine)  (stress-test)
```

Use one tool alone, or chain all three for hard problems.

## Tools

### `debate` — Stress-test your decisions
Multi-model adversarial critique. Sends your problem to GPT (Skeptic) and Gemini (Steelman) for independent analysis, then anonymized cross-examination. Grounded with Google Search evidence. Claude synthesizes the final recommendation.

**Cost:** ~$0.15-0.25 per run (GPT + Gemini + Google Search)

**Structured inputs** (v5): instead of dumping a single long `context` string, prefer structured fields — the server composes them into an anchored, high-signal prompt the analysts can actually reason about:

| Field | Type | What it's for |
|---|---|---|
| `decision_statement` | string | One-sentence statement of the exact decision being evaluated |
| `options` | string[] | The specific options or paths under consideration |
| `constraints_list` | string[] | Hard constraints the decision must respect |
| `key_evidence` | string[] | Direct verbatim quotes, facts, or numbers — do not paraphrase |
| `unresolved_uncertainties` | string[] | Questions you don't yet have answers to |
| `stakes` | string | What happens if this decision is wrong |
| `resource_uris` | string[] | `file://` URIs the server reads and injects verbatim |
| `context` | string | Optional narrative prose (fallback) |
| `question` | string | Specific question to focus the debate |
| `current_leaning` | string | What you're leaning toward — the Skeptic attacks this |
| `domain` | string | Domain expertise injected into both analysts |

At least one of `context`, `decision_statement`, `options`, `key_evidence`, or `resource_uris` must be provided.

```
You describe your plan
        │
        v
  [Web Search] ── gathers current facts, laws, data
        │
        v
  ┌──────────┐          ┌──────────┐
  │ SKEPTIC  │          │ STEELMAN │
  │  (GPT)   │          │ (Gemini) │
  │          │          │          │
  │ Attacks  │          │ Finds    │
  │ your plan│          │ strongest│
  │ ruthlessly│         │ version, │
  │          │          │ then     │
  │          │          │ stress-  │
  │          │          │ tests it │
  └──────────┘          └──────────┘
        │    Round 2: they     │
        │    read each other   │
        │    (anonymized) and  │
        │    cross-examine     │
        v                      v
  ┌────────────────────────────────┐
  │  Claude synthesizes, forced   │
  │  to preserve disagreements    │
  └────────────────────────────────┘
```

**Use when:** "What am I missing?", "Stress-test this", "Is this a good idea?", high-stakes decisions

### `reframe` — See your problem differently
Takes a stuck situation and shows it from 5-6 different angles using thinking operators: invert incentives, flip the customer, shift scale, dissolve the problem, change the constraint, and more. Evidence-grounded via Google Search, then Claude generates the reframes.

**Cost:** ~$0.05 per run (Google Search only — Claude generates the output)

**Use when:** Going in circles, "What am I not seeing?", "Help me think about this differently"

### `diverge` — Generate non-obvious ideas
Produces 20 ideas in 3 waves — obvious (5), adjacent (10), wild (5) — using cross-domain analogy and constraint inversion. Surfaces the 3-5 non-obvious ideas worth testing with 72-hour scrappy tests. Evidence-grounded, Claude generates.

**Cost:** ~$0.05 per run (Google Search only — Claude generates the output)

**Use when:** "Give me ideas", "What could we try?", "Think outside the box", "Brainstorm this"

## Quick Start

### 1. Get API keys

- **OpenAI** — [platform.openai.com](https://platform.openai.com)
- **Google AI** — [aistudio.google.com](https://aistudio.google.com)

### 2. Clone and install

```bash
git clone https://github.com/grzesir/debate-mcp.git
cd debate-mcp
npm install
```

### 3. Set API keys

Either set environment variables directly:
```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AI..."
```

Or create a `.env` file in the project directory:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

### 4. Add to Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "thinking-tools": {
      "command": "bash",
      "args": ["/path/to/debate-mcp/run.sh"]
    }
  }
}
```

Restart Claude Code. You'll see `debate`, `reframe`, and `diverge` available.

## Logging & Cost Tracking

Every tool invocation is automatically saved to `logs/`:
- **Markdown files** — human-readable transcripts with YAML frontmatter
- **`index.jsonl`** — machine-queryable index for spend tracking

```bash
# Total spend
jq -s 'map(.cost_usd) | add' logs/index.jsonl

# Spend by tool
jq -s 'group_by(.tool) | map({tool: .[0].tool, cost: (map(.cost_usd) | add), runs: length})' logs/index.jsonl

# All debates
jq -s 'map(select(.tool=="debate"))' logs/index.jsonl
```

## Configuration

All tunable via environment variables in `run.sh`:

| Variable | Default | Description |
|---|---|---|
| `GPT_MODEL` | `gpt-5.4` | Model for debate Skeptic |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | Model for Steelman + evidence |
| `GPT_MAX_TOKENS` | `8192` | Max output tokens for GPT |
| `DEBATE_LOG_DIR` | `./logs` | Where to write logs |
| `CALL_TIMEOUT_MS` | `90000` | Timeout per API call |

Pricing defaults (for cost tracking):

| Variable | Default | Description |
|---|---|---|
| `GPT_INPUT_PRICE` | `3.00` | $/1M input tokens |
| `GPT_OUTPUT_PRICE` | `15.00` | $/1M output tokens |
| `GEMINI_INPUT_PRICE` | `2.00` | $/1M input tokens |
| `GEMINI_OUTPUT_PRICE` | `12.00` | $/1M output tokens |
| `GROUNDING_COST_PER_CALL` | `0.035` | Google Search per call |

## Design

**debate** runs GPT and Gemini in parallel with asymmetric roles — Skeptic (attack) and Steelman (strongest version + stress test). Round 2 uses anonymized cross-examination (research shows identity bias degrades debate quality). Claude synthesizes but is forced to preserve disagreements. Based on research from Karpathy's LLM Council, Perplexity Model Council, and NeurIPS/ICML debate protocol papers.

**reframe** uses a library of 15 thinking operators (invert incentives, flip the customer, shift scale, dissolve the problem, etc.) selected dynamically based on your specific problem.

**diverge** uses the Wave method (obvious → adjacent → wild) with cross-domain analogy and constraint inversion — techniques backed by creativity research (Guilford, Torrance, De Bono, IDEO).

**reframe and diverge** gather web evidence via Gemini + Google Search, then return structured instructions that Claude follows. Only the evidence search costs money — the creative generation is done by whatever model is already running your session.

## Why structured inputs (v5, 2026-04)

Earlier versions of this server asked callers to "include ALL relevant details" in a single `context` string. That turned out to be exactly the wrong instruction:

1. **Tool description bloat taxes the caller.** MCP tool descriptions are loaded into the calling model's context on every turn. Verbose instructions like "PASS THE FULL CONTEXT, don't summarize, more is better" permanently eat the caller's token budget even when the tool is not invoked. Research from 2025/2026 found that multi-server MCP setups can burn 50k–80k tokens (up to 40% of the window) on tool descriptions alone before a user types a character.

2. **Dumping raw text triggers context rot in the analysts.** The "Lost in the Middle" effect (Liu et al. TACL 2024, Chroma 2025) shows that frontier model attention concentrates on the beginning and end of prompts — information buried in the middle of a long dump can see 30%+ accuracy drops. The Maximum Effective Context Window is typically 1–10% of the advertised window. A 50k-token `context` blob is not a richer debate, it's a debate with the decisive facts invisible.

3. **The bottleneck is signal quality, not raw volume.** A debate tool's job is to attack a specific decision with specific evidence. Structured fields (`decision_statement`, `options`, `key_evidence`, `unresolved_uncertainties`) force the caller to extract the decision-relevant facts instead of paraphrasing, and pin them in named buckets the analysts can find. Resource URIs let the server fetch full documents directly without bloating the caller's prompt at all.

The fix is architectural, not prose. Tool descriptions are kept to 1–2 sentences. Structured extraction replaces raw dumps. Pass-by-reference via `file://` URIs bypasses the caller's context window entirely.

## License

MIT
