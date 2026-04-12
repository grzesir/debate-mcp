#!/bin/bash
# === Thinking Tools MCP Server ===
# Three tools: debate (critique), reframe (new angles), diverge (ideas).
# Change models here when newer versions release:
export GPT_MODEL="${GPT_MODEL:-gpt-5.4}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"

# === Pricing ($ per 1M tokens) — update when model prices change ===
export GPT_INPUT_PRICE="${GPT_INPUT_PRICE:-3.00}"
export GPT_OUTPUT_PRICE="${GPT_OUTPUT_PRICE:-15.00}"
export GEMINI_INPUT_PRICE="${GEMINI_INPUT_PRICE:-2.00}"
export GEMINI_OUTPUT_PRICE="${GEMINI_OUTPUT_PRICE:-12.00}"
export GROUNDING_COST_PER_CALL="${GROUNDING_COST_PER_CALL:-0.035}"

# Where to write logs (markdown + JSONL index).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export DEBATE_LOG_DIR="${DEBATE_LOG_DIR:-$SCRIPT_DIR/logs}"

# --- API Keys ---
# Option 1: Set these directly
# export OPENAI_API_KEY="sk-..."
# export GEMINI_API_KEY="AI..."

# Option 2: Load from a .env file
if [ -z "$OPENAI_API_KEY" ] || [ -z "$GEMINI_API_KEY" ]; then
  ENV_FILE="${ENV_FILE:-.env}"
  if [ -f "$ENV_FILE" ]; then
    extract_key() { grep "^$1=" "$ENV_FILE" | head -1 | cut -d'=' -f2- | sed 's/^"//;s/"$//'; }
    [ -z "$OPENAI_API_KEY" ] && export OPENAI_API_KEY="$(extract_key OPENAI_API_KEY)"
    [ -z "$GEMINI_API_KEY" ] && export GEMINI_API_KEY="$(extract_key GEMINI_API_KEY)"
  fi
fi

exec node "$SCRIPT_DIR/index.js"
