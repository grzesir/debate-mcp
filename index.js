#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// --- Configurable Models (change via env vars in run.sh) ---
const GPT_MODEL = process.env.GPT_MODEL || "gpt-5.4";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
const CALL_TIMEOUT_MS = parseInt(process.env.CALL_TIMEOUT_MS || "90000");

// --- Pricing ($ per 1M tokens; override via env vars as pricing changes) ---
//
// These are used to compute dollar cost per debate. Update them in run.sh
// when prices change — no code edit required. Defaults err on the side of
// slightly OVER-estimating so we don't undercount spend.
//
// Gemini 3.1 Pro Preview pricing was not publicly posted at time of writing
// (per reference_gemini_model_pricing_apr2026.md). Default below is an
// educated estimate — override if Google publishes a lower number.
const PRICING = {
  [GPT_MODEL]: {
    input: parseFloat(process.env.GPT_INPUT_PRICE || "3.00"),
    output: parseFloat(process.env.GPT_OUTPUT_PRICE || "15.00"),
  },
  [GEMINI_MODEL]: {
    input: parseFloat(process.env.GEMINI_INPUT_PRICE || "2.00"),
    output: parseFloat(process.env.GEMINI_OUTPUT_PRICE || "12.00"),
  },
};

// Google Search grounding is billed separately from tokens.
// ~$35 / 1000 grounded queries is the standard rate; override if your
// billing differs. This is a flat per-call charge.
const GROUNDING_COST_PER_CALL = parseFloat(
  process.env.GROUNDING_COST_PER_CALL || "0.035"
);

function computeCost(model, inputTokens, outputTokens, grounded = false) {
  const p = PRICING[model];
  if (!p) return 0;
  const tokenCost = (inputTokens * p.input + outputTokens * p.output) / 1_000_000;
  const groundCost = grounded ? GROUNDING_COST_PER_CALL : 0;
  return tokenCost + groundCost;
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// --- Timeout Utility ---

function withTimeout(promise, ms) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`Timed out after ${ms / 1000}s`)), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}

// --- Asymmetric Role Prompts ---

const SKEPTIC_BASE = `Describe the strongest case that the plan below fails. Write in a direct, analytical style.

Your analysis MUST include these sections — fill every one:

1. FATAL FLAW: The single biggest reason this plan fails. Name it in one sentence.
2. FAILURE MECHANISM: Exactly how that flaw causes failure. Be specific about the chain of events.
3. FAILURE TIMELINE: When does the failure become visible? What are the earliest warning signs?
4. HIDDEN DEPENDENCIES: At least 3 unstated assumptions that must be true for this plan to work.
5. KILL SHOT: One specific piece of evidence that, if discovered, would prove this plan is doomed.
6. IMPACT EXPOSURE: Quantify the consequence in domain-appropriate units (cost, time delay, user loss, reliability drop, opportunity cost — whatever fits the situation). Ranges are acceptable; vagueness is not.
7. MINIMUM SALVAGE: The smallest change that would materially reduce the fatal flaw.
8. CONFIDENCE: Rate your confidence in this bear case (0-100) with a one-sentence justification.

Constraints:
- Lead with the biggest problem. Never open with praise, agreement, or summary.
- No "on the other hand." You are making the case AGAINST.
- Name specific risks with specific consequences.
- When citing facts, distinguish VERIFIED (from provided evidence) from UNVERIFIED (from training data).
- If a quantity is uncertain, give a range and state what drives the uncertainty.`;

const STEELMAN_BASE = `You are a Steelman Analyst. Your job is to find the strongest version of every argument, then stress-test it.

Your approach:
1. First, identify what is genuinely strong about the plan. What would make this succeed?
2. Then, find weak points by asking: "What would a domain expert say is wrong here?"
3. For each criticism, provide the strongest counter-argument AND the strongest defense.
4. Identify where reasonable people would disagree and explain both sides fairly.
5. Look for hidden dependencies, unstated assumptions, and second-order effects.
6. Rate your confidence for each major claim: HIGH, MEDIUM, or LOW.
7. When you disagree with another analyst, first restate their argument in its strongest form before explaining why you disagree.
8. Suggest improvements that preserve the plan's strengths while fixing its weaknesses.
9. When citing facts, distinguish between VERIFIED (from provided evidence/search results) and UNVERIFIED (from your training data). Flag any claim you are not certain about.

Be thorough but concise. No filler. Every sentence should carry information.`;

function buildSkepticPrompt(domain, currentLeaning) {
  let prompt = SKEPTIC_BASE;
  if (domain) {
    prompt = `You are a senior ${domain}.\n\n${prompt}`;
  }
  if (currentLeaning) {
    prompt += `\n\nCRITICAL FOCUS: The decision-maker is currently leaning toward: "${currentLeaning}". Your FATAL FLAW and KILL SHOT sections must specifically target this leaning. Find every reason it could be wrong. The decision-maker needs you to counter their confirmation bias.`;
  }
  return prompt;
}

function buildSteelmanPrompt(domain) {
  let prompt = STEELMAN_BASE;
  if (domain) {
    prompt = prompt.replace(
      "You are a Steelman Analyst.",
      `You are a senior ${domain} operating as a Steelman Analyst.`
    );
  }
  return prompt;
}

// --- Model Callers (with timeout) ---

async function callGPT(prompt, systemPrompt) {
  try {
    const response = await withTimeout(
      openai.chat.completions.create({
        model: GPT_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt },
        ],
        max_completion_tokens: parseInt(process.env.GPT_MAX_TOKENS || "8192"),
      }),
      CALL_TIMEOUT_MS
    );
    const text = response.choices[0]?.message?.content;
    const finishReason = response.choices[0]?.finish_reason;
    const input_tokens = response.usage?.prompt_tokens || 0;
    const output_tokens = response.usage?.completion_tokens || 0;

    // Catch silent empty responses — the #1 cause of blank Skeptic R1.
    if (!text || text.trim().length === 0) {
      return {
        ok: false,
        text: `[GPT EMPTY RESPONSE: finish_reason=${finishReason}, input=${input_tokens}, output=${output_tokens}]`,
        model: GPT_MODEL,
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cost: computeCost(GPT_MODEL, input_tokens, output_tokens, false),
        grounded: false,
      };
    }

    return {
      ok: true,
      text,
      model: GPT_MODEL,
      input_tokens,
      output_tokens,
      total_tokens: input_tokens + output_tokens,
      cost: computeCost(GPT_MODEL, input_tokens, output_tokens, false),
      grounded: false,
    };
  } catch (err) {
    return {
      ok: false,
      text: `[GPT ERROR: ${err.message}]`,
      model: GPT_MODEL,
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      cost: 0,
      grounded: false,
    };
  }
}

async function callGemini(prompt, systemPrompt, enableSearch = false) {
  try {
    const config = {
      model: GEMINI_MODEL,
      systemInstruction: systemPrompt,
    };
    if (enableSearch) {
      config.tools = [{ googleSearch: {} }];
    }
    const model = genAI.getGenerativeModel(config);
    const result = await withTimeout(
      model.generateContent(prompt),
      CALL_TIMEOUT_MS
    );
    const usage = result.response.usageMetadata;
    const input_tokens = usage?.promptTokenCount || 0;
    const output_tokens = usage?.candidatesTokenCount || 0;
    return {
      ok: true,
      text: result.response.text(),
      model: GEMINI_MODEL,
      input_tokens,
      output_tokens,
      total_tokens: input_tokens + output_tokens,
      cost: computeCost(GEMINI_MODEL, input_tokens, output_tokens, enableSearch),
      grounded: enableSearch,
    };
  } catch (err) {
    return {
      ok: false,
      text: `[GEMINI ERROR: ${err.message}]`,
      model: GEMINI_MODEL,
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      cost: 0,
      grounded: false,
    };
  }
}

// --- Targeted Verification (Phase 1.5) ---
//
// After Round 1, the analysts surface claims they've explicitly tagged
// UNVERIFIED. Instead of letting those float into Round 2 unresolved,
// we re-run a small Google-Search-grounded query against ONLY those
// specific claims, then inject the results back into the R2 prompts.
// Cost: one extra Gemini call (~$0.04-0.06) to convert "we're not sure"
// into "verified or refuted with sources" before cross-examination.

const MAX_UNVERIFIED_CLAIMS = parseInt(process.env.MAX_UNVERIFIED_CLAIMS || "6");

function extractUnverifiedClaims(text, maxClaims = MAX_UNVERIFIED_CLAIMS) {
  if (!text) return [];
  const lines = text.split("\n");
  const claims = [];
  const seen = new Set();
  for (const rawLine of lines) {
    if (!/UNVERIFIED/i.test(rawLine)) continue;
    // Strip markdown bullets, headers, bold/italics, code ticks
    let cleaned = rawLine
      .replace(/^[\s>#-]+/, "")
      .replace(/[*_`]/g, "")
      .replace(/^\d+\.\s*/, "")
      .trim();
    // Drop the "UNVERIFIED:" / "UNVERIFIED but plausible:" prefix if it dominates the line
    cleaned = cleaned.replace(/^UNVERIFIED[^:]*:\s*/i, "").trim();
    if (cleaned.length < 25 || cleaned.length > 600) continue;
    const key = cleaned.toLowerCase().slice(0, 120);
    if (seen.has(key)) continue;
    seen.add(key);
    claims.push(cleaned);
    if (claims.length >= maxClaims) break;
  }
  return claims;
}

async function verifyClaims(claims) {
  if (!claims || claims.length === 0) return null;
  const prompt = `Research and verify or refute each of the following specific claims using current authoritative sources from web search. For each claim, output a single line in this format:

[CLAIM N]: VERIFIED | REFUTED | INCONCLUSIVE — one-sentence finding with source domain (e.g. "irs.gov", "anthropic.com").

Be terse. Two sentences max per claim. Do not editorialize. Do not add caveats beyond the source.

CLAIMS TO VERIFY:
${claims.map((c, i) => `${i + 1}. ${c}`).join("\n")}`;

  const result = await callGemini(
    prompt,
    "You are a fact-checking research assistant. Use Google Search to verify or refute each claim with current authoritative sources. Return a terse, parseable verdict per claim. Do not give opinions.",
    true
  );
  return result;
}

// --- Evidence Gathering (Web Search via Gemini + Google Search) ---

async function gatherEvidence(context, question) {
  const searchPrompt = `Research the following topic. Find the most relevant, current, authoritative facts including:
- Current laws, regulations, and legal requirements
- Recent changes or updates (within the last year)
- Expert consensus and common pitfalls
- Specific numbers, thresholds, deadlines, and requirements
- Relevant case studies or precedents

Be factual and specific. Cite sources where possible. Do not analyze or recommend; just compile the facts.

TOPIC: ${question || context}

CONTEXT FOR WHAT MATTERS: ${context.substring(0, 2000)}`;

  const result = await callGemini(
    searchPrompt,
    "You are a research assistant. Compile factual information from web search results. Be precise and cite sources. Do not give opinions or recommendations.",
    true // enable Google Search
  );

  return result;
}

// --- Tool Logging (shared across debate, reframe, diverge) ---
//
// Every tool invocation is persisted to disk. Two artifacts per run:
//   1. A human-readable markdown file with YAML frontmatter (greppable).
//   2. An append-only JSONL index line (machine-queryable, one per run).
//
// Failure-mode policy: logging is best-effort. If disk writes fail,
// we swallow the error and surface a warning — never break a tool run.

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const LOG_DIR = process.env.DEBATE_LOG_DIR || path.join(__dirname, "logs");

function slugify(str, maxLen = 60) {
  return (str || "untitled")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, maxLen) || "untitled";
}

function tsParts(date = new Date()) {
  const iso = date.toISOString();
  const safe = iso.replace(/:/g, "-").replace(/\./g, "-");
  return { iso, safe };
}

// Format a phase-cost table shared across all tools.
function formatCostTable(phases, totals) {
  const rows = phases.map(p =>
    `| ${p.phase} | ${p.model} | ${p.input_tokens.toLocaleString()} | ${p.output_tokens.toLocaleString()} | ${p.total_tokens.toLocaleString()} | $${p.cost.toFixed(4)}${p.grounded ? " (+grounding)" : ""} |`
  ).join("\n");
  return [
    "| Phase | Model | Input tokens | Output tokens | Total tokens | Cost |",
    "|---|---|---|---|---|---|",
    rows,
    `| **TOTAL** | — | **${totals.tokens_input.toLocaleString()}** | **${totals.tokens_output.toLocaleString()}** | **${totals.tokens.toLocaleString()}** | **$${totals.cost.toFixed(4)}** |`,
  ].join("\n");
}

// Generic log writer. Each tool passes its own markdown body and extra index fields.
async function writeToolLog(toolName, record, markdownBody, extraIndex = {}) {
  try {
    await fs.mkdir(LOG_DIR, { recursive: true });

    const { iso, safe } = tsParts(new Date(record.timestamp));
    const slug = slugify(record.question || record.situation || record.context, 60);
    const basename = `${safe}_${toolName}_${slug}`;
    const mdPath = path.join(LOG_DIR, `${basename}.md`);
    const indexPath = path.join(LOG_DIR, "index.jsonl");

    const fm = [
      "---",
      `id: ${basename}`,
      `tool: ${toolName}`,
      `timestamp: ${iso}`,
      `duration_ms: ${record.duration_ms}`,
      `status: ${record.status}`,
      `tokens_input: ${record.tokens_input}`,
      `tokens_output: ${record.tokens_output}`,
      `tokens_total: ${record.tokens}`,
      `cost_usd: ${record.cost.toFixed(6)}`,
      "---",
      "",
    ].join("\n");

    await fs.writeFile(mdPath, fm + markdownBody, "utf8");

    const indexLine = JSON.stringify({
      id: basename,
      tool: toolName,
      ts: iso,
      file: `${basename}.md`,
      status: record.status,
      duration_ms: record.duration_ms,
      tokens_input: record.tokens_input,
      tokens_output: record.tokens_output,
      tokens: record.tokens,
      cost_usd: +record.cost.toFixed(6),
      phases: record.phases.map(p => ({
        phase: p.phase,
        model: p.model,
        input: p.input_tokens,
        output: p.output_tokens,
        cost: +p.cost.toFixed(6),
        grounded: p.grounded,
      })),
      ...extraIndex,
    }) + "\n";

    await fs.appendFile(indexPath, indexLine, "utf8");
    return { ok: true, file: `${basename}.md` };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// --- MCP Server ---

// --- Structured input helpers ---
//
// Philosophy: instead of encouraging callers to "dump everything" into a
// single context blob (which bloats the caller's prompt AND triggers
// "lost in the middle" / context rot in the analyst models), we accept
// STRUCTURED fields and file:// resource URIs and compose the payload
// on the server side. This keeps the caller's token budget intact and
// gives the analysts high-signal, anchored context instead of a wall of text.
//
// The context rot research (Liu et al. 2024, Chroma 2025) shows attention
// drops sharply for information buried in the middle of long prompts.
// Structured extraction forces the caller to pull the decisive facts into
// specific, named buckets where the analysts can actually find them.

const MAX_RESOURCE_BYTES = parseInt(process.env.MAX_RESOURCE_BYTES || "500000");

async function readResourceUris(uris = []) {
  const contents = [];
  const errors = [];
  if (!uris || uris.length === 0) return { contents, errors };
  let totalBytes = 0;
  for (const uri of uris) {
    try {
      if (!uri.startsWith("file://")) {
        errors.push(`${uri}: only file:// URIs are supported`);
        continue;
      }
      const filePath = fileURLToPath(uri);
      const stat = await fs.stat(filePath);
      if (stat.size > MAX_RESOURCE_BYTES) {
        errors.push(`${uri}: ${stat.size} bytes exceeds per-file cap ${MAX_RESOURCE_BYTES}`);
        continue;
      }
      if (totalBytes + stat.size > MAX_RESOURCE_BYTES) {
        errors.push(`${uri}: total resource budget of ${MAX_RESOURCE_BYTES} bytes exhausted`);
        continue;
      }
      const text = await fs.readFile(filePath, "utf8");
      contents.push({ uri, path: filePath, text });
      totalBytes += stat.size;
    } catch (err) {
      errors.push(`${uri}: ${err.message}`);
    }
  }
  return { contents, errors };
}

function composeStructuredContext({
  narrative,
  decision_statement,
  options,
  constraints_list,
  key_evidence,
  unresolved_uncertainties,
  stakes,
  resources,
  resourceErrors,
}) {
  const parts = [];
  if (decision_statement) parts.push(`## Decision under evaluation\n${decision_statement}`);
  if (options && options.length) {
    parts.push(`## Options\n${options.map((o, i) => `${i + 1}. ${o}`).join("\n")}`);
  }
  if (constraints_list && constraints_list.length) {
    parts.push(`## Constraints\n${constraints_list.map(c => `- ${c}`).join("\n")}`);
  }
  if (key_evidence && key_evidence.length) {
    parts.push(`## Key evidence (verbatim, not summarized)\n${key_evidence.map(e => `- ${e}`).join("\n")}`);
  }
  if (unresolved_uncertainties && unresolved_uncertainties.length) {
    parts.push(`## Unresolved uncertainties\n${unresolved_uncertainties.map(u => `- ${u}`).join("\n")}`);
  }
  if (stakes) parts.push(`## Stakes\n${stakes}`);
  if (narrative) parts.push(`## Narrative context\n${narrative}`);
  if (resources && resources.length) {
    parts.push(`## Attached documents (fetched by server from file:// URIs)`);
    for (const r of resources) {
      parts.push(`### ${r.uri}\n\`\`\`\n${r.text}\n\`\`\``);
    }
  }
  if (resourceErrors && resourceErrors.length) {
    parts.push(`## Resource read errors\n${resourceErrors.map(e => `- ${e}`).join("\n")}`);
  }
  return parts.join("\n\n");
}

const STRUCTURED_FIELD_SHAPE = {
  decision_statement: z.string().optional().describe("One-sentence statement of the exact decision or recommendation being evaluated. Prefer this over narrative prose."),
  options: z.array(z.string()).optional().describe("The specific options or paths under consideration, one per array entry. Don't summarize — list them."),
  constraints_list: z.array(z.string()).optional().describe("Hard constraints the decision must respect (budget, time, legal, technical, stakeholder). One per entry."),
  key_evidence: z.array(z.string()).optional().describe("Direct verbatim quotes, facts, or numbers the analysts must reason about. Paste exact text — do not paraphrase. Each entry should be decision-relevant."),
  unresolved_uncertainties: z.array(z.string()).optional().describe("Open questions you don't yet have answers to. Steers the analysts toward the real cruxes instead of strawmanning settled facts."),
  stakes: z.string().optional().describe("What happens if this decision is wrong. Calibrates how hard the Skeptic attacks."),
  resource_uris: z.array(z.string()).optional().describe("file:// URIs the server will read and inject verbatim into the analyst prompts. Use this instead of pasting long documents — preserves full fidelity without bloating the caller's context window. Per-file and total caps apply (default 500KB)."),
};

const server = new McpServer({
  name: "debate",
  version: "5.2.0",
});

const TOOL_DESCRIPTION = `Full-pipeline thinking + adversarial debate. Runs reframe + diverge + Google-grounded evidence + GPT Skeptic vs Gemini Steelman with anonymized cross-examination, plus a targeted verification round that re-searches any claims the analysts mark UNVERIFIED. Use this whenever the user says "debate", "think about", "stress-test", "what am I missing", or wants a high-stakes decision evaluated. Prefer structured fields (decision_statement, options, key_evidence) and resource_uris — pre-flight grep claude-mem and the user's filesystem and pass relevant findings as key_evidence/resource_uris before calling. Claude MUST synthesize the transcript using the format printed at the end.`;

server.tool(
  "debate",
  TOOL_DESCRIPTION,
  {
    context: z.string().optional().describe("Narrative context for the decision. Optional if you provide structured fields (decision_statement/options/key_evidence) or resource_uris. Prefer structured fields over long prose — raw dumps trigger context rot in the analyst models."),
    question: z.string().optional().describe("Specific question to focus the debate. If omitted, models do a general critical analysis."),
    domain: z.string().optional().describe("Domain expertise to inject into both analysts. Examples: 'tax attorney', 'systems architect', 'financial advisor'."),
    current_leaning: z.string().optional().describe("What the user is currently leaning toward. The Skeptic will specifically attack this leaning to counter confirmation bias."),
    additive_context: z.string().optional().describe("Supplementary perspectives from reframe/diverge tools. Compressed one-line summaries only, not raw dumps — treated as background intelligence, not as the topic being debated."),
    pipeline: z.boolean().optional().describe("Default true. When true, also emits reframe + diverge phase instructions (free, Claude generates) so a single `debate` call gives you the full think pipeline. Set false for fast critique-only mode."),
    ...STRUCTURED_FIELD_SHAPE,
  },
  async ({
    context,
    question,
    domain,
    current_leaning,
    additive_context,
    pipeline,
    decision_statement,
    options,
    constraints_list,
    key_evidence,
    unresolved_uncertainties,
    stakes,
    resource_uris,
  }) => {
    const runPipeline = pipeline !== false;

    // =========================================================
    // PREFLIGHT GATE (v5.2)
    //
    // The point: force the caller (Claude) to do the claude-mem +
    // filesystem + prior-debate-log research BEFORE invoking debate,
    // not after. The server cannot reach those sources itself — stdio
    // MCP has no filesystem or MCP-client access beyond its own dir.
    // So the caller is the only place high-value local context can
    // be assembled. If we don't gate here, "more research" stays a
    // vibe instead of an enforced property.
    //
    // Also caps key_evidence at KEY_EVIDENCE_MAX items to prevent
    // the caller from dumping raw grep output — context rot kicks in
    // hard once prompts pass ~50% of the model's effective window.
    // =========================================================
    const KEY_EVIDENCE_MAX = parseInt(process.env.KEY_EVIDENCE_MAX || "12");
    const looksComplex = !!(decision_statement || (options && options.length > 0));
    const hasEvidenceSignal = (key_evidence && key_evidence.length > 0)
      || (resource_uris && resource_uris.length > 0);

    if (looksComplex && !hasEvidenceSignal) {
      return {
        content: [{
          type: "text",
          text: [
            "PREFLIGHT REQUIRED — this looks like a complex decision (decision_statement or options provided) but neither `key_evidence` nor `resource_uris` is populated.",
            "",
            "Before calling `debate` again, do this research pass in YOUR OWN context (not the server's):",
            "1. **claude-mem**: search for prior related decisions, conclusions, and work history. Use mcp__plugin_claude-mem_mcp-search__search or smart_search.",
            "2. **Filesystem grep**: search /Users/robertgrzesik/Documents/Development/ for relevant docs, plans, code, prior debate logs (in mcp_servers/debate/logs/), or project files bearing on the question.",
            "3. **Prior debate logs**: check mcp_servers/debate/logs/index.jsonl for historical debates on similar topics.",
            "",
            "Then compress the findings to 5–12 verbatim quotes and pass them as `key_evidence` (string array), plus the most decision-relevant doc paths as `resource_uris` (file:// URIs). Do NOT paraphrase — use exact quotes. Do NOT exceed " + KEY_EVIDENCE_MAX + " key_evidence items (context rot kicks in above that).",
            "",
            "If claude-mem and filesystem genuinely have nothing relevant, pass `key_evidence: [\"(preflight completed; no prior context found in claude-mem or filesystem)\"]` to bypass this gate — the explicit empty is different from the implicit empty and tells the analysts you checked.",
          ].join("\n"),
        }],
      };
    }

    if (key_evidence && key_evidence.length > KEY_EVIDENCE_MAX) {
      return {
        content: [{
          type: "text",
          text: [
            "PREFLIGHT TOO NOISY — `key_evidence` has " + key_evidence.length + " items (max " + KEY_EVIDENCE_MAX + ").",
            "",
            "Context rot research (Wang et al. Jan 2026, Chroma Mar 2026) shows model reasoning degrades catastrophically once prompts pass ~40–50% of the effective window. Dumping raw grep output defeats the purpose of structured extraction.",
            "",
            "Deduplicate. Keep the " + KEY_EVIDENCE_MAX + " most decision-relevant verbatim quotes. For bulk documents, use `resource_uris` (file://...) instead — the server reads them pass-by-reference without bloating caller context.",
          ].join("\n"),
        }],
      };
    }

    const { contents: resources, errors: resourceErrors } = await readResourceUris(resource_uris);
    const composedContext = composeStructuredContext({
      narrative: context,
      decision_statement,
      options,
      constraints_list,
      key_evidence,
      unresolved_uncertainties,
      stakes,
      resources,
      resourceErrors,
    });
    if (!composedContext) {
      return {
        content: [{
          type: "text",
          text: "Error: debate requires at least one of `context`, `decision_statement`, `options`, `key_evidence`, or `resource_uris`.",
        }],
      };
    }
    context = composedContext;
    const startTime = Date.now();
    const sections = [];

    // Per-phase token + cost accounting. Every model call appends one entry.
    // Totals are computed at the end for display and logging.
    const phases = [];
    const recordPhase = (phase, result) => {
      phases.push({
        phase,
        model: result.model,
        input_tokens: result.input_tokens,
        output_tokens: result.output_tokens,
        total_tokens: result.total_tokens,
        cost: result.cost,
        grounded: result.grounded,
        ok: result.ok,
      });
    };
    const sumField = (field) => phases.reduce((a, p) => a + (p[field] || 0), 0);

    const focusQuestion = question || "Analyze this thoroughly. What are the problems? What's missing? What are the risks? What should be done differently?";

    // Build role-specific prompts
    const skepticPrompt = buildSkepticPrompt(domain, current_leaning);
    const steelmanPrompt = buildSteelmanPrompt(domain);

    // =========================================================
    // PHASE 0: Evidence Gathering (Web Search)
    // Uses Gemini + Google Search grounding to find current facts.
    // If search fails, debate proceeds without evidence (graceful).
    // =========================================================

    let evidencePack = null;
    const evidence = await gatherEvidence(context, question);
    recordPhase("evidence", evidence);

    if (evidence.ok) {
      evidencePack = evidence.text;
      sections.push(
        `# DEBATE TRANSCRIPT`,
        `_Models: Analyst A (${GPT_MODEL}, Principled Skeptic${domain ? `, ${domain}` : ""}) | Analyst B (${GEMINI_MODEL}, Steelman Analyst${domain ? `, ${domain}` : ""})_`,
        current_leaning ? `_Decision-maker's current leaning: "${current_leaning}" (Skeptic will specifically attack this)_\n` : `\n`,
        `## Phase 0: Evidence Gathered (via Google Search)\n`,
        `${evidencePack}\n`,
      );
    } else {
      sections.push(
        `# DEBATE TRANSCRIPT`,
        `_Models: Analyst A (${GPT_MODEL}, Principled Skeptic${domain ? `, ${domain}` : ""}) | Analyst B (${GEMINI_MODEL}, Steelman Analyst${domain ? `, ${domain}` : ""})_`,
        current_leaning ? `_Decision-maker's current leaning: "${current_leaning}" (Skeptic will specifically attack this)_\n` : `\n`,
        `_Note: Web search failed (${evidence.text}). Debate proceeding without search-grounded evidence._\n`,
      );
    }

    // =========================================================
    // PIPELINE PHASES (reframe + diverge + compress)
    // Free — generated by Claude (the calling model) as it reads
    // the response. Same machinery as the `think` tool, but emitted
    // inside `debate` because most users say "debate this" when
    // they mean "think hard about this."
    // =========================================================
    if (runPipeline) {
      sections.push(
        `\n---\n`,
        `## Pipeline Phase 1: REFRAME (Claude: generate 5-6 reframes of the original question)\n`,
        REFRAME_SYSTEM,
        ``,
        `**SITUATION:** ${context}`,
        ``,
        `Apply 5-6 thinking operators. Then pick the 1-2 most promising angles.`,
        ``,
        `## Pipeline Phase 2: DIVERGE (Claude: generate 20 ideas from the best reframe)\n`,
        DIVERGE_SYSTEM,
        ``,
        `Generate 20 ideas through the lens of your best reframe. Then pick Top 3-5.`,
        ``,
        `## Pipeline Phase 3: COMPRESS (Claude: prepare additive context)`,
        ``,
        `Compress reframe + diverge into ≤500 words of high-signal background intelligence. The debate transcript below is about the ORIGINAL QUESTION — these phases are supplementary perspectives Claude folds into the final synthesis, not the topic being debated.`,
        ``,
        `---\n`,
      );
    }

    // =========================================================
    // ROUND 1: Independent Analysis (parallel, with evidence)
    // =========================================================

    const evidenceSection = evidencePack
      ? `\nVERIFIED EVIDENCE (from web search - cite this when making factual claims, mark claims not supported by this evidence as UNVERIFIED):\n${evidencePack}\n\n`
      : "";

    // Prompt anchoring pattern based on Liu et al. TACL 2024 "Lost in the Middle":
    // Original question at TOP (primacy attention) and BOTTOM (recency attention).
    // Additive context in the MIDDLE where models naturally pay less attention.
    // This prevents additive context from becoming the dominant anchor.
    const additiveSection = additive_context
      ? `\n--- SUPPLEMENTARY CONTEXT (from prior reframe/diverge analysis — use as background evidence only, do NOT debate these sub-topics) ---\n${additive_context}\n--- END SUPPLEMENTARY CONTEXT ---\n\n`
      : "";

    const r1Prompt = `ORIGINAL QUESTION (this is what you are debating):\n${focusQuestion}\n\nCONTEXT:\n${context}\n${evidenceSection}${additiveSection}ANCHOR REMINDER — your analysis MUST directly answer this original question:\n${focusQuestion}\n\nProvide your analysis. Structure it with clear sections.`;

    const [skepticR1, steelmanR1] = await Promise.all([
      callGPT(r1Prompt, skepticPrompt),
      callGemini(r1Prompt, steelmanPrompt),
    ]);
    recordPhase("r1_skeptic", skepticR1);
    recordPhase("r1_steelman", steelmanR1);

    sections.push(
      `## Round 1: Independent Analysis\n`,
      `### Analyst A (Skeptic)\n${skepticR1.text}\n`,
      `### Analyst B (Steelman)\n${steelmanR1.text}`,
    );

    // =========================================================
    // PHASE 1.5: Targeted Verification of UNVERIFIED claims
    // Both analysts mark uncertain claims with "UNVERIFIED" tags.
    // We pull those out and run a single targeted Gemini+Search
    // call to verify or refute them, then inject the new evidence
    // into the R2 cross-exam prompt. Skipped if no claims surfaced.
    // =========================================================

    let verificationPack = null;
    let verifiedClaims = [];
    if (skepticR1.ok && steelmanR1.ok) {
      const skepticUnverified = extractUnverifiedClaims(skepticR1.text);
      const steelmanUnverified = extractUnverifiedClaims(steelmanR1.text);
      const seen = new Set();
      verifiedClaims = [...skepticUnverified, ...steelmanUnverified].filter(c => {
        const key = c.toLowerCase().slice(0, 120);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).slice(0, MAX_UNVERIFIED_CLAIMS);

      if (verifiedClaims.length > 0) {
        const verification = await verifyClaims(verifiedClaims);
        if (verification) {
          recordPhase("verification", verification);
          if (verification.ok) {
            verificationPack = verification.text;
            sections.push(
              `\n## Phase 1.5: Targeted Verification`,
              `_Re-searched ${verifiedClaims.length} claim${verifiedClaims.length === 1 ? "" : "s"} marked UNVERIFIED in Round 1, with fresh Google Search grounding._\n`,
              verificationPack,
              ``,
            );
          }
        }
      }
    }

    // Shared logging record — filled in progressively so errored debates
    // still get persisted with whatever we managed to collect.
    const logRecord = {
      timestamp: new Date(startTime).toISOString(),
      duration_ms: 0,
      status: "success",
      models: { skeptic: GPT_MODEL, steelman: GEMINI_MODEL },
      tokens: 0,
      tokens_input: 0,
      tokens_output: 0,
      cost: 0,
      phases,
      domain: domain || null,
      current_leaning: current_leaning || null,
      question: question || null,
      context,
      evidence_ok: evidence.ok,
      evidence: evidencePack || evidence.text,
      skeptic_r1: skepticR1.text,
      steelman_r1: steelmanR1.text,
      skeptic_r2: null,
      steelman_r2: null,
    };

    const finalizeTotals = () => {
      logRecord.tokens_input = sumField("input_tokens");
      logRecord.tokens_output = sumField("output_tokens");
      logRecord.tokens = sumField("total_tokens");
      logRecord.cost = sumField("cost");
      logRecord.duration_ms = Date.now() - startTime;
    };

    // If both errored, return what we have
    if (!skepticR1.ok && !steelmanR1.ok) {
      sections.push(`\n---\n_Both models errored. Check API keys and model IDs in run.sh._`);
      logRecord.status = "both_errored";
      finalizeTotals();
      const logResult = await writeToolLog("debate", logRecord, sections.join("\n"), {
        domain: logRecord.domain,
        has_leaning: !!logRecord.current_leaning,
        evidence_ok: logRecord.evidence_ok,
        question: logRecord.question,
        context_preview: (logRecord.context || "").slice(0, 200),
      });
      if (logResult.ok) sections.push(`_Logged: ${logResult.file} — $${logRecord.cost.toFixed(4)}_`);
      return { content: [{ type: "text", text: sections.join("\n") }] };
    }
    if (!skepticR1.ok || !steelmanR1.ok) {
      sections.push(`\n---\n_One model errored. Returning partial results._`);
      logRecord.status = "partial";
      finalizeTotals();
      const logResult = await writeToolLog("debate", logRecord, sections.join("\n"), {
        domain: logRecord.domain,
        has_leaning: !!logRecord.current_leaning,
        evidence_ok: logRecord.evidence_ok,
        question: logRecord.question,
        context_preview: (logRecord.context || "").slice(0, 200),
      });
      if (logResult.ok) sections.push(`_Logged: ${logResult.file} — ${logRecord.tokens.toLocaleString()} tokens, $${logRecord.cost.toFixed(4)}_`);
      return { content: [{ type: "text", text: sections.join("\n") }] };
    }

    // =========================================================
    // ROUND 2: Anonymized Cross-Examination (parallel)
    // =========================================================

    const crossExamInstructions = `ORIGINAL QUESTION (re-stated to prevent drift): ${focusQuestion}

ORIGINAL CONTEXT:
${context}
${evidencePack ? `\nVERIFIED EVIDENCE (from web search):\n${evidencePack}\n` : ""}${verificationPack ? `\nNEWLY VERIFIED EVIDENCE (targeted re-search of UNVERIFIED Round 1 claims — treat these verdicts as authoritative):\n${verificationPack}\n` : ""}
YOUR PREVIOUS ANALYSIS:
{MY_ANALYSIS}

ANOTHER ANALYST'S ANALYSIS:
{OTHER_ANALYSIS}

ANTI-SYCOPHANCY RULE: Do NOT agree just to be agreeable. Research shows multi-agent debates collapse into premature consensus when agents prioritize harmony over truth. Maintain your position where evidence supports it. Agreement must be earned with specific evidence, not offered as a courtesy.

RESPOND TO THE OTHER ANALYST. Follow these steps exactly:
1. STEELMAN: Restate the other analyst's single strongest argument in your own words. Prove you understood it.
2. CONCEDE: Name anything they caught that you missed. Be honest — but ONLY concede points with genuine merit.
3. DISAGREE: For each point of disagreement, state the specific claim you reject and your specific counter-evidence or reasoning. Distinguish VERIFIED facts (from evidence) from UNVERIFIED claims.
4. UPDATE: State what (if anything) you changed your mind about. If nothing, say "I did not change my mind" and explain why.
5. FIRMEST BELIEF: What is the one thing you are most confident about? Rate: 0-100 with justification.
6. UNRESOLVED: Name the top 1-2 disagreements that matter most and have not been settled.

ANCHOR REMINDER: Every point must relate back to the ORIGINAL QUESTION: ${focusQuestion}`;

    const [skepticR2, steelmanR2] = await Promise.all([
      callGPT(
        crossExamInstructions
          .replace("{MY_ANALYSIS}", skepticR1.text)
          .replace("{OTHER_ANALYSIS}", steelmanR1.text),
        skepticPrompt
      ),
      callGemini(
        crossExamInstructions
          .replace("{MY_ANALYSIS}", steelmanR1.text)
          .replace("{OTHER_ANALYSIS}", skepticR1.text),
        steelmanPrompt
      ),
    ]);
    recordPhase("r2_skeptic", skepticR2);
    recordPhase("r2_steelman", steelmanR2);

    sections.push(
      `\n---\n`,
      `## Round 2: Cross-Examination (anonymized)\n`,
      `### Analyst A (Skeptic) responds\n${skepticR2.text}\n`,
      `### Analyst B (Steelman) responds\n${steelmanR2.text}`,
    );

    // =========================================================
    // CONSTRAINED SYNTHESIS INSTRUCTIONS
    // Forces Claude to preserve adversarial structure, not smooth
    // disagreements into false consensus.
    // =========================================================

    sections.push(
      `\n---\n`,
      `## Synthesis Required (Claude: complete this exact structure)`,
      ``,
      `You MUST synthesize this debate using the following format. Do not free-form summarize. Do not invent a compromise if none exists. If the analysts fundamentally disagree, present both paths.`,
      ``,
      `**Recommendation:** [Your synthesized recommendation based on the full debate]`,
      `**Key Agreements:** [Where both analysts converged with high confidence]`,
      `**Strongest Argument For:** [The single best argument supporting the recommendation]`,
      `**Strongest Argument Against:** [The single best counter-argument the recommendation must survive]`,
      `**Crux of Disagreement:** [The one assumption or fact that determines which path is correct]`,
      `**What Would Resolve the Crux:** [Specific evidence, action, or consultation needed]`,
      `**What Would Falsify the Recommendation:** [If this turns out to be true, the recommendation is wrong]`,
      `**Risk of Acting Now:** [Downside of proceeding immediately]`,
      `**Risk of Waiting:** [Downside of delaying the decision]`,
      `**Confidence:** [HIGH/MEDIUM/LOW with brief explanation]`,
      `**Unverified Claims to Check:** [Any factual claims from the debate that were not grounded in the web search evidence and should be independently verified before acting]`,
      ``,
    );

    // Persist the full debate transcript with cost breakdown.
    logRecord.skeptic_r2 = skepticR2.text;
    logRecord.steelman_r2 = steelmanR2.text;
    if (!skepticR2.ok || !steelmanR2.ok) logRecord.status = "r2_partial";
    finalizeTotals();

    // Cost/token footer shown inline in the transcript.
    const phaseSummary = phases.map(p =>
      `  - ${p.phase}: ${p.input_tokens.toLocaleString()} in / ${p.output_tokens.toLocaleString()} out → $${p.cost.toFixed(4)}${p.grounded ? " (+ grounding)" : ""}`
    ).join("\n");
    sections.push(
      ``,
      `### Usage & Cost`,
      phaseSummary,
      `  - **TOTAL: ${logRecord.tokens_input.toLocaleString()} input + ${logRecord.tokens_output.toLocaleString()} output = ${logRecord.tokens.toLocaleString()} tokens → $${logRecord.cost.toFixed(4)}**`,
      ``,
    );

    const logResult = await writeToolLog("debate", logRecord, sections.join("\n"), {
        domain: logRecord.domain,
        has_leaning: !!logRecord.current_leaning,
        evidence_ok: logRecord.evidence_ok,
        question: logRecord.question,
        context_preview: (logRecord.context || "").slice(0, 200),
      });
    if (logResult.ok) {
      sections.push(`_Logged to: ${logResult.file}_`);
    } else {
      sections.push(`_(Logging failed: ${logResult.error})_`);
    }

    return { content: [{ type: "text", text: sections.join("\n") }] };
  }
);

// =========================================================
// TOOL 2: REFRAME
// Shows a stuck problem from completely different angles
// using thinking operators. Does NOT solve the problem —
// changes how you see it so new solutions become visible.
// =========================================================

const REFRAME_SYSTEM = `You are a Problem Reframing Specialist. Your job is to take a stuck situation and show it from angles nobody has considered. You do NOT solve the problem. You change how the person sees it.

You have a library of thinking operators. For each problem, pick the 5-6 most relevant operators and apply them. Each reframe should make the person think "oh wait, maybe THAT's the real issue."

THINKING OPERATORS (pick the best 5-6 for this problem):

1. INVERT INCENTIVES — "What if you incentivized the opposite behavior?"
2. SHIFT THE VALUE CHAIN — "Where else could the revenue/value actually come from?"
3. FEATURE → BEHAVIOR — "What if this isn't a product/feature problem but a habit/behavior problem?"
4. FLIP THE CUSTOMER — "What if the real buyer, user, or beneficiary is someone else entirely?"
5. REPLACE THE METRIC — "What if you're optimizing for the wrong thing entirely?"
6. FIND THE REAL BOTTLENECK — "Which part of the system is actually stuck? Not where you think."
7. REMOVE A LAYER — "What if you went direct and cut out an intermediary?"
8. ADD A LAYER — "What if someone or something else handled the hardest part?"
9. SHIFT THE TIME HORIZON — "How does this look in 10 years? In 10 days?"
10. INVERT THE SCALE — "What if you served 10x fewer people at 10x the price? Or the reverse?"
11. REFRAME THE COMPETITION — "What if your real competition isn't who you think it is?"
12. DISSOLVE THE PROBLEM — "What if this problem doesn't actually need to be solved?"
13. REVERSE THE FLOW — "What if the customer came to you instead of you going to them?"
14. CHANGE THE CONSTRAINT — "What if the thing you think is fixed is actually variable?"
15. SPLIT OR MERGE — "What if this is actually two separate problems? Or what if two problems are actually one?"

For EACH reframe you produce, output:
1. **Operator:** Which thinking operator you used
2. **Reframe:** The problem restated through this lens (2-3 sentences max)
3. **What changes:** What new solutions, strategies, or actions become visible from this angle (2-3 sentences)
4. **Test question:** One specific question the person could answer to check if this reframe is the right one

Be specific to the actual situation. Do not produce generic reframes. Every reframe must directly reference details from the problem.

Do NOT solve the problem. Do NOT recommend actions. Only show new angles.`;

const REFRAME_DESCRIPTION = `Reframes a stuck situation through 5-6 thinking operators (invert incentives, flip the customer, dissolve the problem, change the constraint, etc.). Use when the user is going in circles or the obvious solutions haven't worked. Returns research + a reframe template Claude fills in.`;

server.tool(
  "reframe",
  REFRAME_DESCRIPTION,
  {
    situation: z.string().describe("The stuck situation, problem, or challenge. Include what you've tried and what constraints exist."),
    focus: z.string().optional().describe("Optional: what specifically to reframe (e.g., 'our pricing model'). If omitted, the whole situation is reframed."),
    resource_uris: z.array(z.string()).optional().describe("file:// URIs the server will read and append to the situation. Per-file and total caps apply (default 500KB)."),
  },
  async ({ situation, focus, resource_uris }) => {
    const { contents: reframeResources, errors: reframeResourceErrors } = await readResourceUris(resource_uris);
    if (reframeResources.length) {
      const resourceBlock = reframeResources
        .map(r => `### ${r.uri}\n\`\`\`\n${r.text}\n\`\`\``)
        .join("\n\n");
      situation = `${situation}\n\n## Attached documents\n${resourceBlock}`;
    }
    if (reframeResourceErrors.length) {
      situation = `${situation}\n\n## Resource read errors\n${reframeResourceErrors.map(e => `- ${e}`).join("\n")}`;
    }
    const startTime = Date.now();
    const sections = [];
    const phases = [];
    const recordPhase = (phase, result) => {
      phases.push({
        phase, model: result.model,
        input_tokens: result.input_tokens, output_tokens: result.output_tokens,
        total_tokens: result.total_tokens, cost: result.cost,
        grounded: result.grounded, ok: result.ok,
      });
    };
    const sumField = (field) => phases.reduce((a, p) => a + (p[field] || 0), 0);

    // Evidence gathering via Google Search (the only API cost).
    // The reframe generation itself is done by Claude (the calling model),
    // so reframe + diverge are nearly free compared to debate.
    const evidence = await gatherEvidence(situation, focus);
    recordPhase("evidence", evidence);

    sections.push(`# REFRAME: Apply Thinking Operators\n`);

    if (evidence.ok) {
      sections.push(
        `## Background Research (via Google Search)\n`,
        `${evidence.text}\n`,
      );
    }

    // Return the situation + operator library + instructions for Claude
    // to generate the reframes itself. This avoids a GPT/Gemini call.
    sections.push(
      `## Situation\n`,
      situation,
      focus ? `\n**Specific focus:** ${focus}` : "",
      ``,
      `## Your Task (Claude: generate the reframes)`,
      ``,
      REFRAME_SYSTEM,
      ``,
      `**SITUATION TO REFRAME:**`,
      ``,
      situation,
      focus ? `\n**SPECIFIC FOCUS:** ${focus}` : "",
      evidence.ok ? `\n**EVIDENCE FROM WEB RESEARCH:**\n${evidence.text}` : "",
      ``,
      `Apply your 5-6 best thinking operators now. Be specific — reference actual details from the situation. After producing the reframes, help the user identify which 1-2 feel most promising and suggest feeding those into the **diverge** tool.`,
      ``,
    );

    // Log (only evidence cost since Claude generates the reframes)
    const logRecord = {
      timestamp: new Date(startTime).toISOString(),
      duration_ms: Date.now() - startTime,
      status: evidence.ok ? "success" : "evidence_failed",
      tokens_input: sumField("input_tokens"),
      tokens_output: sumField("output_tokens"),
      tokens: sumField("total_tokens"),
      cost: sumField("cost"),
      phases,
      situation,
      question: focus || situation.slice(0, 200),
    };

    sections.push(
      `### Evidence Cost`,
      `  - evidence: ${evidence.input_tokens.toLocaleString()} in / ${evidence.output_tokens.toLocaleString()} out → $${evidence.cost.toFixed(4)}${evidence.grounded ? " (+ grounding)" : ""}`,
      `  - reframe generation: **$0 (generated by Claude)**`,
      ``,
    );

    const logResult = await writeToolLog("reframe", logRecord, sections.join("\n"), {
      focus: focus || null,
      situation_preview: situation.slice(0, 200),
    });
    if (logResult.ok) sections.push(`_Logged to: ${logResult.file}_`);

    return { content: [{ type: "text", text: sections.join("\n") }] };
  }
);

// =========================================================
// TOOL 3: DIVERGE
// Generates a high volume of ideas with zero judgment,
// using cross-domain analogy and constraint manipulation.
// Surfaces the non-obvious outliers worth testing.
// =========================================================

const DIVERGE_SYSTEM = `You are a Divergent Idea Generator. Your job is to produce a HIGH VOLUME of ideas — most will be mediocre, and that's the point. Creative breakthroughs hide in quantity. You generate first, judge later.

PROCESS (follow this exact sequence):

WAVE A — THE OBVIOUS (5 ideas)
List the 5 most obvious solutions anyone would suggest. Get them out of the way. Label each "[OBVIOUS]".

WAVE B — THE ADJACENT (10 ideas)
Now push sideways. For each, use one of these moves:
- Steal from another industry ("How does [unrelated field] solve a similar problem?")
- Combine two unrelated concepts
- Change the audience
- Change the delivery mechanism
- Change the business model
Label each "[ADJACENT]" and note which move you used.

WAVE C — THE WILD (5 ideas)
Now break rules. For each, apply one of these forcing constraints:
- "What if the budget were $0?"
- "What if you had to launch in 24 hours?"
- "What if you could only do ONE thing?"
- "What if you did the exact opposite of conventional wisdom?"
- "What would be embarrassing but might actually work?"
Label each "[WILD]" and note which constraint you used.

After all 20 ideas, do this:

TOP PICKS (choose 3-5)
From all 20, pick the 3-5 ideas that are:
1. NOT obvious (skip Wave A picks)
2. Specifically testable within 72 hours
3. Have asymmetric upside (cheap to try, big if it works)

For each top pick, provide:
- **The idea** (1-2 sentences)
- **Why it's non-obvious** (what makes this different from the default approach)
- **72-hour test** (exactly how to test this cheaply and quickly)
- **Success signal** (what would tell you this is working)
- **Cross-domain source** (if applicable — what industry/field inspired this)

Rules:
- No generic advice ("improve your marketing", "focus on value"). Every idea must be specific enough to act on.
- No judgment during Waves A-C. Save all evaluation for Top Picks.
- Reference specific details from the user's situation in every idea.
- If the user provided constraints, respect them in Waves A-B but deliberately break them in Wave C.`;

const DIVERGE_DESCRIPTION = `Generates 20 ideas in 3 waves (5 obvious, 10 adjacent, 5 wild) then picks the 3-5 non-obvious ones with 72-hour tests. Use when the user needs creative options, not a single answer. Returns research + an ideation template Claude fills in.`;

server.tool(
  "diverge",
  DIVERGE_DESCRIPTION,
  {
    situation: z.string().describe("The problem, challenge, or opportunity to generate ideas for. Include what's been tried and what resources are available."),
    constraints: z.string().optional().describe("Real constraints to respect in Waves A and B (e.g., 'budget under $1000', 'B2B only'). Wave C deliberately breaks these."),
    avoid: z.string().optional().describe("Obvious solutions to explicitly skip (e.g., 'we already tried paid ads and SEO')."),
    reframe: z.string().optional().describe("A specific reframe/angle to generate ideas from (often output of the reframe tool). Focuses all 20 ideas through this lens."),
    resource_uris: z.array(z.string()).optional().describe("file:// URIs the server will read and append to the situation. Per-file and total caps apply (default 500KB)."),
  },
  async ({ situation, constraints, avoid, reframe: reframeAngle, resource_uris }) => {
    const { contents: divergeResources, errors: divergeResourceErrors } = await readResourceUris(resource_uris);
    if (divergeResources.length) {
      const resourceBlock = divergeResources
        .map(r => `### ${r.uri}\n\`\`\`\n${r.text}\n\`\`\``)
        .join("\n\n");
      situation = `${situation}\n\n## Attached documents\n${resourceBlock}`;
    }
    if (divergeResourceErrors.length) {
      situation = `${situation}\n\n## Resource read errors\n${divergeResourceErrors.map(e => `- ${e}`).join("\n")}`;
    }
    const startTime = Date.now();
    const sections = [];
    const phases = [];
    const recordPhase = (phase, result) => {
      phases.push({
        phase, model: result.model,
        input_tokens: result.input_tokens, output_tokens: result.output_tokens,
        total_tokens: result.total_tokens, cost: result.cost,
        grounded: result.grounded, ok: result.ok,
      });
    };
    const sumField = (field) => phases.reduce((a, p) => a + (p[field] || 0), 0);

    // Evidence gathering via Google Search (the only API cost).
    // Idea generation is done by Claude (the calling model) — free.
    const evidence = await gatherEvidence(situation, reframeAngle || null);
    recordPhase("evidence", evidence);

    sections.push(`# DIVERGE: Generate Ideas in 3 Waves\n`);

    if (evidence.ok) {
      sections.push(
        `## Background Research (via Google Search)\n`,
        `${evidence.text}\n`,
      );
    }

    // Return the situation + wave structure + instructions for Claude
    // to generate the 20 ideas itself. No GPT/Gemini call needed.
    sections.push(
      `## Your Task (Claude: generate the ideas)`,
      ``,
      DIVERGE_SYSTEM,
      ``,
      `**SITUATION:**`,
      ``,
      situation,
      reframeAngle ? `\n**IMPORTANT — GENERATE ALL IDEAS THROUGH THIS SPECIFIC ANGLE:**\n${reframeAngle}` : "",
      constraints ? `\n**CONSTRAINTS (respect in Waves A-B, break in Wave C):**\n${constraints}` : "",
      avoid ? `\n**AVOID THESE (already tried or too obvious):**\n${avoid}` : "",
      evidence.ok ? `\n**EVIDENCE FROM WEB RESEARCH:**\n${evidence.text}` : "",
      ``,
      `Generate all 20 ideas now (Wave A: 5 obvious, Wave B: 10 adjacent, Wave C: 5 wild), then pick your Top 3-5. Be specific to this situation. After producing the ideas, help the user evaluate the Top Picks and suggest sending the most promising one to the **debate** tool for stress-testing.`,
      ``,
    );

    // Log (only evidence cost)
    const logRecord = {
      timestamp: new Date(startTime).toISOString(),
      duration_ms: Date.now() - startTime,
      status: evidence.ok ? "success" : "evidence_failed",
      tokens_input: sumField("input_tokens"),
      tokens_output: sumField("output_tokens"),
      tokens: sumField("total_tokens"),
      cost: sumField("cost"),
      phases,
      situation,
      question: (reframeAngle || situation).slice(0, 200),
    };

    sections.push(
      `### Evidence Cost`,
      `  - evidence: ${evidence.input_tokens.toLocaleString()} in / ${evidence.output_tokens.toLocaleString()} out → $${evidence.cost.toFixed(4)}${evidence.grounded ? " (+ grounding)" : ""}`,
      `  - idea generation: **$0 (generated by Claude)**`,
      ``,
    );

    const logResult = await writeToolLog("diverge", logRecord, sections.join("\n"), {
      has_constraints: !!constraints,
      has_avoid: !!avoid,
      has_reframe: !!reframeAngle,
      situation_preview: situation.slice(0, 200),
    });
    if (logResult.ok) sections.push(`_Logged to: ${logResult.file}_`);

    return { content: [{ type: "text", text: sections.join("\n") }] };
  }
);

// =========================================================
// TOOL 4: THINK
// Full pipeline: evidence → reframe (Claude) → diverge
// (Claude) → compress → REAL debate (GPT + Gemini) on the
// ORIGINAL question, enriched with reframe/diverge context.
// =========================================================

const THINK_DESCRIPTION = `Full pipeline: reframe + diverge + real multi-model adversarial debate on the original question, with Google Search grounding. Use for complex high-stakes decisions that need both creative exploration and stress-testing. Supports structured fields and resource_uris to avoid context rot. Claude MUST synthesize the debate using the format printed at the end.`;

server.tool(
  "think",
  THINK_DESCRIPTION,
  {
    situation: z.string().optional().describe("Narrative context for the problem. Optional if you provide structured fields or resource_uris. Prefer structured extraction over long prose."),
    question: z.string().optional().describe("The specific question that gets DEBATED at the end. If omitted, a general analysis is performed."),
    domain: z.string().optional().describe("Domain expertise for the debate phase. Examples: 'tax attorney', 'systems architect'."),
    constraints: z.string().optional().describe("Real constraints for the reframe/diverge phases (shorthand form). For the debate phase, prefer constraints_list."),
    avoid: z.string().optional().describe("Obvious solutions to skip during idea generation."),
    current_leaning: z.string().optional().describe("What you're currently leaning toward. The Skeptic will attack this."),
    ...STRUCTURED_FIELD_SHAPE,
  },
  async ({
    situation,
    question,
    domain,
    constraints,
    avoid,
    current_leaning,
    decision_statement,
    options,
    constraints_list,
    key_evidence,
    unresolved_uncertainties,
    stakes,
    resource_uris,
  }) => {
    const { contents: resources, errors: resourceErrors } = await readResourceUris(resource_uris);
    const composedContext = composeStructuredContext({
      narrative: situation,
      decision_statement,
      options,
      constraints_list,
      key_evidence,
      unresolved_uncertainties,
      stakes,
      resources,
      resourceErrors,
    });
    if (!composedContext) {
      return {
        content: [{
          type: "text",
          text: "Error: think requires at least one of `situation`, `decision_statement`, `options`, `key_evidence`, or `resource_uris`.",
        }],
      };
    }
    situation = composedContext;
    const startTime = Date.now();
    const sections = [];
    const phases = [];
    const recordPhase = (phase, result) => {
      phases.push({
        phase, model: result.model,
        input_tokens: result.input_tokens, output_tokens: result.output_tokens,
        total_tokens: result.total_tokens, cost: result.cost,
        grounded: result.grounded, ok: result.ok,
      });
    };
    const sumField = (field) => phases.reduce((a, p) => a + (p[field] || 0), 0);

    const focusQuestion = question || "Analyze this thoroughly. What should be done? What are the risks? What's missing?";

    // =========================================================
    // PHASE 0: Evidence Gathering (shared across all phases)
    // =========================================================
    const evidence = await gatherEvidence(situation, question);
    recordPhase("evidence", evidence);

    const evidencePack = evidence.ok ? evidence.text : null;

    sections.push(
      `# THINK: Full Pipeline\n`,
      `**Original question:** ${focusQuestion}\n`,
    );

    if (evidencePack) {
      sections.push(`## Background Research (via Google Search)\n`, `${evidencePack}\n`);
    }

    // =========================================================
    // PHASE 1 & 2: Reframe + Diverge (Claude generates)
    // These are returned as instructions for Claude to fill in.
    // Claude's output from these phases will be in the conversation,
    // but the DEBATE below will also receive compressed versions.
    // =========================================================

    sections.push(
      `---\n`,
      `## Phase 1: REFRAME (Claude: generate 5-6 reframes of the original question)\n`,
      ``,
      REFRAME_SYSTEM,
      ``,
      `**SITUATION:** ${situation}`,
      constraints ? `\n**CONSTRAINTS:** ${constraints}` : "",
      avoid ? `\n**AVOID:** ${avoid}` : "",
      evidencePack ? `\n**EVIDENCE:** Use the research above.` : "",
      ``,
      `Apply 5-6 thinking operators. Then pick the 1-2 most promising angles.`,
      ``,
      `## Phase 2: DIVERGE (Claude: generate 20 ideas from the best reframe)\n`,
      ``,
      DIVERGE_SYSTEM,
      ``,
      `Generate 20 ideas through the lens of your best reframe. Then pick Top 3-5.`,
      ``,
      `## Phase 3: COMPRESS (Claude: prepare additive context for the debate)`,
      ``,
      `CRITICAL: Compress your reframe and diverge outputs into a SHORT additive brief. Research shows additive context over ~1,000 tokens causes "context rot" that degrades the debate. Keep it tight:`,
      `- Top 2 reframes (ONE LINE EACH, max 50 words per line)`,
      `- Top 3 ideas (ONE LINE EACH, max 50 words per line)`,
      `- Total compression must be under 500 words`,
      `- End with: "These are supplementary perspectives. The debate below is about the ORIGINAL QUESTION: ${focusQuestion}"`,
      ``,
    );

    // =========================================================
    // PHASE 4: REAL DEBATE (GPT Skeptic + Gemini Steelman)
    // Debates the ORIGINAL question, with compressed reframe/diverge
    // as additive context. This is the core value.
    // =========================================================

    const skepticPrompt = buildSkepticPrompt(domain, current_leaning);
    const steelmanPrompt = buildSteelmanPrompt(domain);

    // Prompt anchoring: original question at TOP and BOTTOM (U-curve attention
    // pattern from Liu et al. TACL 2024). Evidence and context in the middle.
    const debatePrompt = [
      `ORIGINAL QUESTION (this is what you are debating):`,
      focusQuestion,
      ``,
      `CONTEXT:`,
      situation,
      evidencePack ? `\nVERIFIED EVIDENCE (from web search):\n${evidencePack}` : "",
      constraints ? `\nCONSTRAINTS: ${constraints}` : "",
      ``,
      `ANCHOR REMINDER — your analysis MUST directly answer this original question:`,
      focusQuestion,
      ``,
      `Provide your analysis. Structure it with clear sections.`,
    ].join("\n");

    const [skepticR1, steelmanR1] = await Promise.all([
      callGPT(debatePrompt, skepticPrompt),
      callGemini(debatePrompt, steelmanPrompt),
    ]);
    recordPhase("r1_skeptic", skepticR1);
    recordPhase("r1_steelman", steelmanR1);

    sections.push(
      `## Phase 4: DEBATE on the Original Question\n`,
      `_Models: Analyst A (${GPT_MODEL}, Skeptic${domain ? `, ${domain}` : ""}) | Analyst B (${GEMINI_MODEL}, Steelman${domain ? `, ${domain}` : ""})_`,
      current_leaning ? `_Current leaning: "${current_leaning}" (Skeptic targets this)_\n` : `\n`,
      `### Skeptic Analysis\n${skepticR1.text}\n`,
      `### Steelman Analysis\n${steelmanR1.text}\n`,
    );

    // Phase 1.5: targeted verification of UNVERIFIED claims from R1
    let thinkVerificationPack = null;
    if (skepticR1.ok && steelmanR1.ok) {
      const skepticUnverified = extractUnverifiedClaims(skepticR1.text);
      const steelmanUnverified = extractUnverifiedClaims(steelmanR1.text);
      const seen = new Set();
      const uniqueUnverified = [...skepticUnverified, ...steelmanUnverified].filter(c => {
        const key = c.toLowerCase().slice(0, 120);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).slice(0, MAX_UNVERIFIED_CLAIMS);
      if (uniqueUnverified.length > 0) {
        const verification = await verifyClaims(uniqueUnverified);
        if (verification) {
          recordPhase("verification", verification);
          if (verification.ok) {
            thinkVerificationPack = verification.text;
            sections.push(
              `## Phase 4.5: Targeted Verification`,
              `_Re-searched ${uniqueUnverified.length} claim${uniqueUnverified.length === 1 ? "" : "s"} marked UNVERIFIED in Round 1._\n`,
              thinkVerificationPack,
              ``,
            );
          }
        }
      }
    }

    // Round 2 cross-examination (if both succeeded)
    if (skepticR1.ok && steelmanR1.ok) {
      const crossExam = `ORIGINAL QUESTION (re-stated to prevent drift): ${focusQuestion}

ORIGINAL CONTEXT:
${situation}
${evidencePack ? `\nVERIFIED EVIDENCE:\n${evidencePack}\n` : ""}${thinkVerificationPack ? `\nNEWLY VERIFIED EVIDENCE (targeted re-search of UNVERIFIED Round 1 claims — treat as authoritative):\n${thinkVerificationPack}\n` : ""}
YOUR PREVIOUS ANALYSIS:
{MY_ANALYSIS}

ANOTHER ANALYST'S ANALYSIS:
{OTHER_ANALYSIS}

RESPOND. Follow these steps:
1. STEELMAN: Restate their single strongest argument.
2. CONCEDE: What did they catch that you missed?
3. DISAGREE: Specific claims you reject and why.
4. UPDATE: What did you change your mind about?
5. FIRMEST BELIEF: Your highest-confidence conclusion. Rate 0-100.
6. UNRESOLVED: Top 1-2 unsettled disagreements.`;

      const [skepticR2, steelmanR2] = await Promise.all([
        callGPT(
          crossExam.replace("{MY_ANALYSIS}", skepticR1.text).replace("{OTHER_ANALYSIS}", steelmanR1.text),
          skepticPrompt
        ),
        callGemini(
          crossExam.replace("{MY_ANALYSIS}", steelmanR1.text).replace("{OTHER_ANALYSIS}", skepticR1.text),
          steelmanPrompt
        ),
      ]);
      recordPhase("r2_skeptic", skepticR2);
      recordPhase("r2_steelman", steelmanR2);

      sections.push(
        `### Cross-Examination: Skeptic\n${skepticR2.text}\n`,
        `### Cross-Examination: Steelman\n${steelmanR2.text}\n`,
      );
    }

    // Synthesis instructions
    sections.push(
      `---\n`,
      `## Synthesis Required (Claude: complete this structure)`,
      ``,
      `You have completed reframe, diverge, AND a real multi-model debate — all about the ORIGINAL QUESTION: "${focusQuestion}"`,
      ``,
      `Synthesize everything using this format:`,
      ``,
      `**Recommendation:** [Based on ALL phases — reframes, ideas, and the debate]`,
      `**Key Agreements:** [Where the debate analysts converged]`,
      `**Strongest Argument For:** [The single best argument supporting the recommendation]`,
      `**Strongest Argument Against:** [The single best counter-argument]`,
      `**Crux:** [The one assumption that determines the right path]`,
      `**What Would Resolve It:** [Specific evidence or action needed]`,
      `**72-Hour Test:** [The first concrete action to take this week]`,
      `**Confidence:** [0-100 with explanation]`,
      ``,
    );

    // Cost/token footer
    const finalizeTotals = () => {
      return {
        tokens_input: sumField("input_tokens"),
        tokens_output: sumField("output_tokens"),
        tokens: sumField("total_tokens"),
        cost: sumField("cost"),
      };
    };
    const totals = finalizeTotals();

    const phaseSummary = phases.map(p =>
      `  - ${p.phase}: ${p.input_tokens.toLocaleString()} in / ${p.output_tokens.toLocaleString()} out → $${p.cost.toFixed(4)}${p.grounded ? " (+ grounding)" : ""}`
    ).join("\n");
    sections.push(
      `### Usage & Cost`,
      phaseSummary,
      `  - reframe + diverge + compress: **$0 (generated by Claude)**`,
      `  - **TOTAL: ${totals.tokens_input.toLocaleString()} input + ${totals.tokens_output.toLocaleString()} output = ${totals.tokens.toLocaleString()} tokens → $${totals.cost.toFixed(4)}**`,
      ``,
    );

    const logRecord = {
      timestamp: new Date(startTime).toISOString(),
      duration_ms: Date.now() - startTime,
      status: (skepticR1.ok || steelmanR1.ok) ? "success" : "debate_failed",
      ...totals,
      phases,
      situation,
      question: focusQuestion,
    };

    const logResult = await writeToolLog("think", logRecord, sections.join("\n"), {
      has_domain: !!domain,
      has_constraints: !!constraints,
      has_avoid: !!avoid,
      has_leaning: !!current_leaning,
      situation_preview: situation.slice(0, 200),
    });
    if (logResult.ok) sections.push(`_Logged to: ${logResult.file}_`);

    return { content: [{ type: "text", text: sections.join("\n") }] };
  }
);

// --- Start ---

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
