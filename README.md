# Thinking Tools MCP

**Adversarial multi-model debate for high-stakes decisions.** An MCP server for Claude Code that forces better thinking through structured research, problem reframing, divergent ideation, web-grounded evidence, multi-model adversarial critique, targeted verification, and constrained synthesis.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![npm](https://img.shields.io/npm/v/debate-mcp)](https://www.npmjs.com/package/debate-mcp)

## How it works

```
   YOU say "debate this"
          │
          v
   PREFLIGHT (caller-side, enforced by server gate)
   Claude greps claude-mem + filesystem for prior decisions,
   compresses findings into key_evidence + resource_uris
          │
          v
   ┌─────────────────────────────────────────────────────────┐
   │  Phase 0: Google Search Evidence                        │
   │  (Gemini + Google Search grounding)                     │
   ├─────────────────────────────────────────────────────────┤
   │  Pipeline Phases 1-3 (Claude fills in, $0 cost)         │
   │  1. REFRAME: 5-6 thinking operators                     │
   │  2. DIVERGE: 20 ideas (obvious/adjacent/wild)           │
   │  3. COMPRESS: ≤500 words of high-signal background      │
   ├─────────────────────────────────────────────────────────┤
   │  Round 1: Independent Analysis (parallel)               │
   │                                                         │
   │  ┌──────────┐           ┌──────────┐                    │
   │  │ SKEPTIC  │           │ STEELMAN │                    │
   │  │ (GPT-5.4)│           │(Gemini   │                    │
   │  │          │           │ 3.1 Pro) │                    │
   │  │ Attacks  │           │ Strongest│                    │
   │  │ your plan│           │ version +│                    │
   │  │ ruthless │           │ stress-  │                    │
   │  │          │           │ test     │                    │
   │  └──────────┘           └──────────┘                    │
   ├─────────────────────────────────────────────────────────┤
   │  Phase 1.5: Targeted Verification                       │
   │  Parses R1 for UNVERIFIED claims, re-searches them      │
   │  via Google Search, injects verdicts into R2             │
   ├─────────────────────────────────────────────────────────┤
   │  Round 2: Anonymized Cross-Examination (parallel)       │
   │  Each analyst reads the other's R1 (anonymized)         │
   │  + newly verified evidence. Anti-sycophancy enforced.   │
   ├─────────────────────────────────────────────────────────┤
   │  Claude synthesizes using constrained format:           │
   │  Recommendation / Key Agreements / Strongest For/       │
   │  Against / Crux / What Would Falsify / Confidence       │
   └─────────────────────────────────────────────────────────┘
```

**Cost:** ~$0.22 per debate. **Time:** ~3 minutes.

## Tools

### `debate` — Full-pipeline thinking + adversarial critique

The primary tool. Runs everything: preflight gate, evidence gathering, reframe, diverge, Round 1 (GPT Skeptic + Gemini Steelman), targeted verification of UNVERIFIED claims, Round 2 cross-examination, and constrained synthesis.

Use when: "debate this", "think about this", "stress-test", "what am I missing", "is this right", "sanity check", "poke holes in this", high-stakes decisions (tax, legal, architecture, business strategy).

**Key parameters:**

| Field | Type | What it's for |
|---|---|---|
| `decision_statement` | string | One-sentence statement of the exact decision being evaluated |
| `options` | string[] | The specific options or paths under consideration |
| `key_evidence` | string[] | Direct verbatim quotes, facts, or numbers. **Do not paraphrase.** Max 12 items |
| `resource_uris` | string[] | `file://` URIs the server reads and injects verbatim (up to 500KB) |
| `constraints_list` | string[] | Hard constraints the decision must respect |
| `unresolved_uncertainties` | string[] | Questions you don't yet have answers to |
| `stakes` | string | What happens if this decision is wrong |
| `current_leaning` | string | What you're leaning toward (Skeptic will attack this) |
| `domain` | string | Domain expertise injected into both analysts |
| `question` | string | Specific question to focus the debate |
| `context` | string | Optional narrative prose (prefer structured fields) |
| `pipeline` | boolean | Default `true`. Set `false` for fast critique-only (skips reframe/diverge) |
| `additive_context` | string | Compressed supplementary perspectives from prior tool calls |

**Preflight gate (v5.2):** Complex decisions (those with `decision_statement` or `options`) **must** arrive with populated `key_evidence` or `resource_uris`. The server fast-fails with instructions if the caller hasn't done its research.

### `think` — Alias for debate

Same as `debate`. Kept for backward compatibility. If you've been using `think`, it still works.

### `reframe` — See your problem differently

Takes a stuck situation and shows it from 5-6 different angles using thinking operators: invert incentives, flip the customer, shift scale, dissolve the problem, change the constraint, and more. Evidence-grounded via Google Search, then Claude generates the reframes.

**Cost:** ~$0.05 per run. **Use when:** Going in circles, obvious solutions haven't worked.

### `diverge` — Generate non-obvious ideas

Produces 20 ideas in 3 waves (5 obvious, 10 adjacent, 5 wild) using cross-domain analogy and constraint inversion. Surfaces the 3-5 non-obvious ideas worth testing with 72-hour scrappy tests.

**Cost:** ~$0.05 per run. **Use when:** You need creative options, not a single answer.

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

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AI..."
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:GEMINI_API_KEY = "AI..."
```

**Windows (cmd):**
```cmd
set OPENAI_API_KEY=sk-...
set GEMINI_API_KEY=AI...
```

Or create a `.env` file in the project directory (works on all platforms):
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

### 4. Add to Claude Code

**Easiest — use the CLI (cross-platform):**

```bash
# macOS / Linux
claude mcp add thinking-tools -s user -- bash /path/to/debate-mcp/run.sh

# Windows (Git Bash or WSL must be on PATH for bash)
claude mcp add thinking-tools -s user -- bash C:/Users/YOU/debate-mcp/run.sh
```

**Or hand-edit `~/.claude.json`:**

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

**Windows notes:**

- `run.sh` has a `#!/bin/bash` shebang, so you need a working `bash` on PATH. **Git Bash** (ships with Git for Windows) and **WSL** (`bash.exe` on PATH) both work. PowerShell/cmd alone do not.
- Use forward slashes in paths (`C:/Users/.../debate-mcp/run.sh`) inside JSON or the CLI — JSON-escaped backslashes (`C:\\Users\\...`) also work but are noisier.
- If you have multiple `bash.exe` on PATH (Git Bash, WSL, MSYS2), the first one wins. Test with `where bash` to confirm which one Claude Code will invoke.

Restart Claude Code. You'll see `debate`, `think`, `reframe`, and `diverge` available.

## Logging & Cost Tracking

Every tool invocation is automatically saved to `logs/`:
- **Markdown files** with YAML frontmatter (human-readable transcripts)
- **`index.jsonl`** (machine-queryable, one line per run)

```bash
# Total spend
jq -s 'map(.cost_usd) | add' logs/index.jsonl

# Spend by tool
jq -s 'group_by(.tool) | map({tool: .[0].tool, cost: (map(.cost_usd) | add), runs: length})' logs/index.jsonl

# Average cost per debate
jq -s '[.[] | select(.tool=="debate")] | {runs: length, avg_cost: (map(.cost_usd) | add / length), avg_tokens: (map(.tokens) | add / length)}' logs/index.jsonl
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
| `KEY_EVIDENCE_MAX` | `12` | Max key_evidence items before preflight gate rejects |
| `MAX_RESOURCE_BYTES` | `500000` | Total file read budget for resource_uris |
| `MAX_UNVERIFIED_CLAIMS` | `6` | Max UNVERIFIED claims to re-search in Phase 1.5 |

Pricing defaults (for cost tracking):

| Variable | Default | Description |
|---|---|---|
| `GPT_INPUT_PRICE` | `3.00` | $/1M input tokens |
| `GPT_OUTPUT_PRICE` | `15.00` | $/1M output tokens |
| `GEMINI_INPUT_PRICE` | `2.00` | $/1M input tokens |
| `GEMINI_OUTPUT_PRICE` | `12.00` | $/1M output tokens |
| `GROUNDING_COST_PER_CALL` | `0.035` | Google Search per call |

## Benchmarks

A regression harness lives in `benchmarks/`. It runs fixture debates against the server, parses responses, and asserts expected phase markers, cost, and latency are within budget.

```bash
./benchmarks/run.sh --rejections-only    # free, ~0.4s, verifies preflight gate
./benchmarks/run.sh --all                # full suite (~$0.25, ~3 min)
```

Current baseline (`benchmarks/baseline.json`):

| Fixture | Duration | Cost | Status |
|---|---|---|---|
| rejection-empty-evidence | 0.2s | $0.00 | pass |
| rejection-oversized-evidence | 0.2s | $0.00 | pass |
| valid-typescript-packaging | 155s | $0.22 | pass (incl. Phase 1.5 verification) |

Run `./benchmarks/run.sh --all` after any server change to catch regressions. See `benchmarks/README.md` for fixture format and runner internals.

## Design

**debate** runs GPT and Gemini in parallel with asymmetric roles. The Skeptic (GPT) attacks your plan across 8 mandatory sections (fatal flaw, failure mechanism, failure timeline, hidden dependencies, kill shot, impact exposure, minimum salvage, confidence). The Steelman (Gemini) finds the strongest version of your plan, then stress-tests it. Round 2 uses anonymized cross-examination (research shows identity bias degrades debate quality). Phase 1.5 targeted verification re-searches any claims the analysts mark UNVERIFIED before Round 2. Claude synthesizes but is forced to preserve disagreements via a constrained output format. Based on research from Karpathy's LLM Council, Perplexity Model Council, and NeurIPS/ICML debate protocol papers.

**Pipeline phases** (reframe + diverge) are emitted as part of the debate response and filled in by Claude at zero additional cost. This was merged into `debate` in v5.1 to fix the naming mismatch where users say "debate this" but mean "think hard about this."

**Preflight gate** (v5.2) enforces caller-side research before accepting a complex decision. The server cannot access claude-mem or the user's filesystem directly (stdio transport limitation), so the only way to get high-value local context into the debate is via the caller. The gate ensures this happens on every call instead of relying on caller discipline.

**reframe** uses a library of 15 thinking operators (invert incentives, flip the customer, shift scale, dissolve the problem, etc.) selected dynamically based on your specific problem.

**diverge** uses the Wave method (obvious/adjacent/wild) with cross-domain analogy and constraint inversion, backed by creativity research (Guilford, Torrance, De Bono, IDEO).

**reframe and diverge** gather web evidence via Gemini + Google Search, then return structured instructions that Claude follows. Only the evidence search costs money.

## Why structured inputs

Earlier versions asked callers to "include ALL relevant details" in a single `context` string. That was exactly the wrong instruction:

1. **Tool description bloat taxes the caller.** MCP tool descriptions are loaded into the calling model's context on every turn. Verbose instructions permanently eat token budget even when the tool is not invoked. Research from 2025/2026 found multi-server MCP setups can burn 50k-80k tokens (40% of the window) on descriptions alone.

2. **Raw text triggers context rot in the analysts.** The "Lost in the Middle" effect (Liu et al. TACL 2024, Chroma 2025) shows frontier model attention concentrates on prompt beginning/end. Information buried in the middle of a long dump sees 30%+ accuracy drops. The effective context window is typically 1-10% of the advertised window.

3. **The bottleneck is signal quality, not volume.** Structured fields (`decision_statement`, `options`, `key_evidence`) force the caller to extract decision-relevant facts and pin them in named buckets the analysts can find. Resource URIs let the server fetch full documents without bloating the caller's prompt.

The fix is architectural, not prose. Tool descriptions are kept to 1-2 sentences. Structured extraction replaces raw dumps. Pass-by-reference via `file://` URIs bypasses the caller entirely. Preflight gate enforces it.

## Version history

| Version | Date | What changed |
|---|---|---|
| v5.2.0 | 2026-04-15 | Preflight gate (enforced research), benchmark harness |
| v5.1.0 | 2026-04-15 | Phase 1.5 targeted verification, pipeline merge (debate = think), resource_uris on reframe/diverge |
| v5.0.0 | 2026-04-15 | Structured inputs, resource_uris pass-by-reference, 68% shorter tool descriptions |
| v4.0.0 | 2026-04-12 | Added reframe + diverge + think tools, logging, cost tracking |
| v3.0.0 | 2026-04-11 | Anonymized cross-examination, topic drift fix, evidence gathering |

## License

MIT
