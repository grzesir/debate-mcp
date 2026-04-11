# Debate MCP

**Stress-test your decisions before you commit.** An MCP server that runs adversarial AI debates between frontier models, grounded in live web search.

Most AI tools optimize for consensus. Debate MCP optimizes for **finding where your plan breaks.**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![npm](https://img.shields.io/npm/v/debate-mcp)](https://www.npmjs.com/package/debate-mcp)

## How It Works

```
You describe your plan
        |
        v
  [Web Search] -- gathers current facts, laws, regulations
        |
        v
  +-----------+          +-----------+
  |  SKEPTIC  |          | STEELMAN  |
  |  (GPT)    |          | (Gemini)  |
  |           |          |           |
  | Attacks   |          | Finds the |
  | your plan |          | strongest |
  | ruthlessly|          | version,  |
  |           |          | then      |
  |           |          | stress-   |
  |           |          | tests it  |
  +-----------+          +-----------+
        |    Round 2: they     |
        |    read each other   |
        |    (anonymized) and  |
        +--- argue back -------+
                  |
                  v
        [Structured synthesis]
        Recommendation + Crux +
        What Would Falsify +
        Unresolved disagreements
```

## Quick Start

**1. Install**
```bash
npx debate-mcp
```

**2. Add to Claude Code**
```bash
claude mcp add debate npx debate-mcp \
  -e OPENAI_API_KEY=sk-... \
  -e GEMINI_API_KEY=AI...
```

**3. Use it**

Just tell Claude: *"debate this"*, *"what am I missing"*, *"stress-test this plan"*, or *"is this the right call"*.

> [!TIP]
> You can also trigger it with `domain` and `current_leaning` for targeted debates:
> *"Debate this as a tax attorney. I'm leaning toward electing S-Corp."*

## What Makes This Different

| Feature | Why it matters |
|---------|---------------|
| **Asymmetric roles** | One model attacks (Skeptic), one defends then stress-tests (Steelman). Research shows this outperforms giving both models the same prompt. |
| **Anonymized cross-examination** | In Round 2, models see each other's work labeled "another analyst" to prevent identity bias. Based on NeurIPS 2025 research. |
| **Web search grounding** | Before the debate, the server searches for current facts, laws, and regulations. Both models receive this as VERIFIED evidence and must flag ungrounded claims as UNVERIFIED. |
| **Confirmation bias attack** | Tell it what you're leaning toward. The Skeptic will specifically attack that leaning. |
| **Domain expertise** | Pass `domain: "tax attorney"` or `"systems architect"` to make both analysts domain-specific. |
| **Constrained synthesis** | The output forces a structured format: Recommendation, Crux of Disagreement, What Would Falsify, Risk of Acting vs Waiting. Prevents AI from smoothing real disagreements into false consensus. |

## Example

**Input:** *"Should we elect S-Corp status? Net profit $40K, based in NYC."*
**Domain:** `tax attorney`
**Current leaning:** *"I think S-Corp will save on self-employment tax"*

**What happens:**
1. Web search pulls current NYC tax rates, QBI rules, IRS thresholds
2. Skeptic leads with: *"At $40K net profit in NYC, S-Corp election is mathematically guaranteed to lose you money"* and explains exactly why
3. Steelman finds the strongest case for S-Corp, then stress-tests it against NYC-specific tax penalties
4. Cross-examination: Skeptic concedes the QBI interaction point, Steelman concedes the compliance cost erasure
5. Synthesis: **Don't elect. Here's the specific profit threshold where it flips.**

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `GEMINI_API_KEY` | Yes | - | Your Google AI API key |
| `GPT_MODEL` | No | `gpt-5.4` | OpenAI model to use |
| `GEMINI_MODEL` | No | `gemini-3.1-pro-preview` | Google model to use |
| `CALL_TIMEOUT_MS` | No | `90000` | Timeout per API call (ms) |

### MCP Configuration (`.mcp.json`)

```json
{
  "mcpServers": {
    "debate": {
      "command": "npx",
      "args": ["-y", "debate-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

> [!NOTE]
> **Bring your own API keys.** Debate MCP calls OpenAI and Google APIs directly. You are responsible for your own API usage and costs. A typical debate uses ~20,000-30,000 tokens across both providers.

## Tool Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `context` | Yes | The plan, decision, or situation to debate. Include all relevant details. |
| `question` | No | Specific question to focus the debate on. |
| `domain` | No | Domain expertise: `"tax attorney"`, `"systems architect"`, `"financial advisor"`, etc. |
| `current_leaning` | No | What you're leaning toward. The Skeptic attacks this to counter confirmation bias. |

## The Research Behind It

Debate MCP's design is based on peer-reviewed research on multi-agent debate:

- **Asymmetric roles** outperform identical prompts (*"Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate"*, 2025)
- **Anonymized cross-examination** prevents identity bias (*"When Identity Skews Debate"*, NeurIPS 2025)
- **Steelmanning before disagreeing** forces genuine engagement (*Kahneman's Adversarial Collaboration framework*)
- **Re-stating the original question each round** prevents context drift (*"Talk Isn't Always Cheap"*, ICML 2025)
- **Caller-model synthesis** avoids positional commitment bias from debaters (*"Auditing Multi-Agent LLM Reasoning Trees"*, 2025)
- **Ray Dalio's triangulation method**: get independent expert opinions, map convergence and divergence, then decide

## When To Use It

**Good for:** Taxes, legal decisions, financial planning, business strategy, architecture choices, investment analysis, contract terms, hiring decisions, production deployments.

**Not for:** Simple coding tasks, quick lookups, routine bug fixes, or questions with obvious answers.

## License

MIT

