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

const server = new McpServer({
  name: "thinking-tools",
  version: "4.0.0",
});

const TOOL_DESCRIPTION = `Multi-model adversarial debate with web search grounding. Gathers evidence via Google Search, then sends context to GPT-5.4 (Skeptic) and Gemini 3.1 Pro (Steelman) for independent critique and anonymized cross-examination. Returns structured debate transcript for you to synthesize.

USE THIS TOOL WHEN:
- User says "debate this", "challenge this", "get other opinions", "what am I missing", "is this right", "stress-test this", "play devil's advocate", "poke holes in this", "sanity check this", "check my thinking"
- High-consequence or irreversible decisions: taxes, legal, financial planning, investment strategy, business decisions, health, production deployments, contracts, architecture with lock-in
- You are uncertain about a complex recommendation with significant tradeoffs
- User expresses doubt or skepticism ("I'm not sure about...", "this feels risky", "what could go wrong")
- Complex tradeoffs with no clear winner where reasonable experts would disagree

DO NOT USE WHEN:
- Simple coding tasks, bug fixes, or routine file operations
- Questions with clear, well-established answers
- Quick factual lookups or calculations
- The user just wants something done, not evaluated
- Every recommendation (costs real money and takes 30-60 seconds)

PIPELINE: For complex problems, STRONGLY RECOMMEND running the full pipeline: reframe → diverge → debate. Or use the "think" tool which runs all three phases in one call. Don't just run debate alone unless the user specifically asked for critique only.

Returns structured debate transcript. You (Claude) MUST synthesize using the constrained format at the end of the transcript. Do not free-form summarize.`;

server.tool(
  "debate",
  TOOL_DESCRIPTION,
  {
    context: z.string().describe("The FULL context: plan, decision, strategy, or situation. Include ALL relevant details, background, constraints, and stakes. The models only know what you send them."),
    question: z.string().optional().describe("Specific question to focus the debate. If omitted, models do a general critical analysis. THIS is what gets debated — the original question the user asked about."),
    domain: z.string().optional().describe("Domain expertise to inject into both analysts. Examples: 'tax attorney', 'corporate lawyer', 'financial advisor', 'systems architect', 'investment analyst'. Makes the debate domain-specific rather than generic."),
    current_leaning: z.string().optional().describe("What the user is currently leaning toward. The Skeptic will specifically attack this leaning to counter confirmation bias. Example: 'I think we should go with the S-Corp election'."),
    additive_context: z.string().optional().describe("OPTIONAL supplementary perspectives from reframe/diverge tools. This is ADDITIVE — the debate is still about the original question, not about this context. Include compressed one-line summaries of reframes or ideas, not raw dumps. The models will treat this as background intelligence, not as the topic being debated."),
  },
  async ({ context, question, domain, current_leaning, additive_context }) => {
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
${evidencePack ? `\nVERIFIED EVIDENCE (from web search):\n${evidencePack}\n` : ""}
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

const REFRAME_DESCRIPTION = `Problem reframing tool. Takes a stuck situation and shows it from 5-6 completely different angles using thinking operators (invert incentives, flip the customer, shift scale, dissolve the problem, etc.). Does NOT solve the problem — changes how you see it so new solutions become visible.

USE THIS TOOL WHEN:
- User is stuck and keeps circling the same ideas
- User says "I don't know what to do", "I'm going in circles", "help me think about this differently", "what am I not seeing"
- Before using diverge — reframing first produces much better idea generation
- The problem feels unsolvable (it might be the wrong problem)
- User has tried obvious solutions and they haven't worked

DO NOT USE WHEN:
- User already knows the problem clearly and needs ideas (use diverge)
- User has ideas and needs them critiqued (use debate)
- Simple questions with clear answers

PIPELINE: For complex problems, STRONGLY RECOMMEND using the "think" tool instead (runs all phases in one call). If using reframe alone, follow up with diverge → debate for best results.

Returns evidence + instructions. You (Claude) generate the reframes, then help the user pick which feel most promising and feed those into diverge.`;

server.tool(
  "reframe",
  REFRAME_DESCRIPTION,
  {
    situation: z.string().describe("The stuck situation, problem, or challenge. Include all relevant context — what you've tried, what's not working, what constraints exist. The model only knows what you send it."),
    focus: z.string().optional().describe("Optional: what specifically to reframe. E.g., 'our pricing model' or 'why customers churn after month 2'. If omitted, the whole situation is reframed."),
  },
  async ({ situation, focus }) => {
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

const DIVERGE_DESCRIPTION = `Divergent idea generator. Takes a problem and produces 20 ideas in 3 waves: obvious (5), adjacent (10), and wild (5) — then surfaces the 3-5 non-obvious ideas most worth testing. Uses cross-domain analogy, constraint inversion, and forced volume to escape predictable thinking.

USE THIS TOOL WHEN:
- User needs ideas, options, or creative solutions
- User says "give me ideas", "what could we try", "brainstorm this", "how else could we do this", "think outside the box"
- After using reframe — generating ideas from a reframed angle produces much better results
- User is choosing between limited options and needs more possibilities
- The obvious approaches have already been tried or dismissed

DO NOT USE WHEN:
- User needs a problem reframed, not solved (use reframe)
- User has ideas and needs them critiqued (use debate)
- Simple tasks with known solutions
- User wants a single recommendation, not a menu of options

PIPELINE: For complex problems, STRONGLY RECOMMEND using the "think" tool instead (runs all phases in one call). If using diverge alone, follow up with debate to stress-test the best idea.

Returns evidence + instructions. You (Claude) generate the 20 ideas and top picks, then help the user evaluate and optionally send the best one to debate.`;

server.tool(
  "diverge",
  DIVERGE_DESCRIPTION,
  {
    situation: z.string().describe("The problem, challenge, or opportunity to generate ideas for. Include all relevant context — what's been tried, what constraints exist, what resources are available. The model only knows what you send it."),
    constraints: z.string().optional().describe("Real constraints to respect in Waves A and B (e.g., 'budget under $1000', 'must use existing team', 'B2B only'). Wave C will deliberately break these to find wild ideas."),
    avoid: z.string().optional().describe("Obvious solutions to explicitly skip (e.g., 'we already tried paid ads and SEO'). Forces the model away from defaults."),
    reframe: z.string().optional().describe("A specific reframe/angle to generate ideas from (often the output of the reframe tool). E.g., 'The real problem isn't customer acquisition, it's that customers don't trust us enough to start.' Focuses all 20 ideas through this lens."),
  },
  async ({ situation, constraints, avoid, reframe: reframeAngle }) => {
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

const THINK_DESCRIPTION = `Full thinking pipeline in one call. Runs all phases: reframe the problem, generate divergent ideas, then run a REAL multi-model adversarial debate (GPT Skeptic + Gemini Steelman) on the ORIGINAL question — enriched with compressed reframe/diverge perspectives.

IMPORTANT: The debate at the end is about the ORIGINAL question, NOT about the reframe or diverge outputs. Reframe and diverge are additive research that feeds into the debate — they don't replace the topic being debated.

USE THIS TOOL WHEN:
- User says "think about this", "help me figure this out", "I need to think through this", "what should I do about this"
- Complex problems that benefit from seeing it differently AND generating ideas AND stress-testing
- User wants the full pipeline without calling 3 separate tools
- Any time you would have recommended running all 3 tools in sequence

DO NOT USE WHEN:
- User specifically wants ONLY critique (use debate)
- User specifically wants ONLY reframing (use reframe)
- User specifically wants ONLY ideas (use diverge)
- Simple questions with clear answers

Cost: ~$0.25-0.30 per run (evidence + GPT + Gemini for the real adversarial debate). The reframe and diverge phases are free (Claude generates).

Returns: Claude's reframes and ideas PLUS a full multi-model debate transcript on the original question. You (Claude) MUST synthesize the debate using the constrained format.`;

server.tool(
  "think",
  THINK_DESCRIPTION,
  {
    situation: z.string().describe("The problem, challenge, or decision. Include all relevant context. The model only knows what you send it."),
    question: z.string().optional().describe("The specific question to answer. This is what gets DEBATED at the end — the original question. If omitted, a general analysis is performed."),
    domain: z.string().optional().describe("Domain expertise for the debate phase. Examples: 'tax attorney', 'systems architect'."),
    constraints: z.string().optional().describe("Real constraints to respect."),
    avoid: z.string().optional().describe("Obvious solutions to skip."),
    current_leaning: z.string().optional().describe("What you're currently leaning toward. The Skeptic will attack this."),
  },
  async ({ situation, question, domain, constraints, avoid, current_leaning }) => {
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

    // Round 2 cross-examination (if both succeeded)
    if (skepticR1.ok && steelmanR1.ok) {
      const crossExam = `ORIGINAL QUESTION (re-stated to prevent drift): ${focusQuestion}

ORIGINAL CONTEXT:
${situation}
${evidencePack ? `\nVERIFIED EVIDENCE:\n${evidencePack}\n` : ""}
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
