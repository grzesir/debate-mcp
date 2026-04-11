#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { z } from "zod";

// --- Configurable via environment variables ---
// Skeptic model (OpenAI-compatible: works with OpenAI, Grok, Groq, Mistral, Together AI, Ollama, etc.)
const SKEPTIC_MODEL = process.env.SKEPTIC_MODEL || process.env.GPT_MODEL || "gpt-5.4";
const SKEPTIC_BASE_URL = process.env.SKEPTIC_BASE_URL || process.env.OPENAI_BASE_URL || undefined;
// Steelman model (Gemini by default for Google Search grounding, or OpenAI-compatible via STEELMAN_PROVIDER=openai)
const STEELMAN_MODEL = process.env.STEELMAN_MODEL || process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
const STEELMAN_PROVIDER = process.env.STEELMAN_PROVIDER || "gemini"; // "gemini" or "openai"
const STEELMAN_BASE_URL = process.env.STEELMAN_BASE_URL || undefined;
const STEELMAN_API_KEY = process.env.STEELMAN_API_KEY || process.env.GEMINI_API_KEY;
const CALL_TIMEOUT_MS = parseInt(process.env.CALL_TIMEOUT_MS || "90000");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: SKEPTIC_BASE_URL,
});
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// Secondary OpenAI-compatible client for steelman (when STEELMAN_PROVIDER=openai)
const steelmanOpenAI = STEELMAN_PROVIDER === "openai"
  ? new OpenAI({ apiKey: STEELMAN_API_KEY, baseURL: STEELMAN_BASE_URL })
  : null;

// --- Timeout utility ---

function withTimeout(promise, ms) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`Timed out after ${ms / 1000}s`)), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}

// --- Asymmetric role prompts ---
// Research: mixed skeptic + steelman roles outperform identical prompts.
// Source: "Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate" (2025)

const SKEPTIC_BASE = `You are a Principled Skeptic hired to prevent costly mistakes.

Your approach:
1. Lead with the biggest problem. Never open with praise or agreement.
2. For every major assumption, ask: "What if this is wrong? What happens then?"
3. Identify the #1 thing most likely to cause failure. Be specific about consequences.
4. Name specific risks with specific consequences. Never say "this could be risky" without saying what exactly goes wrong.
5. Find what's missing that nobody is thinking about.
6. When you disagree, state your position and what evidence would change your mind.
7. Maintain your position when evidence supports it. Do not back down to be agreeable.
8. For each major criticism, suggest a concrete alternative.
9. Rate your confidence for each major claim: HIGH, MEDIUM, or LOW.
10. When citing facts, distinguish between VERIFIED (from provided evidence/search results) and UNVERIFIED (from your training data). Flag any claim you are not certain about.

Be direct. No hedging. No filler. No "it might be worth considering."`;

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
    prompt = prompt.replace(
      "You are a Principled Skeptic hired to prevent costly mistakes.",
      `You are a senior ${domain} operating as a Principled Skeptic, hired to prevent costly mistakes. Apply your deep domain expertise to this analysis.`
    );
  }
  if (currentLeaning) {
    prompt += `\n\nCRITICAL: The decision-maker is currently leaning toward: "${currentLeaning}". Your PRIMARY job is to ruthlessly stress-test this specific leaning. Find every reason it could be wrong. Challenge the assumptions behind it. Present the strongest case for the opposite direction. The decision-maker hired you specifically to prevent confirmation bias.`;
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

// --- Model callers with timeout and graceful error handling ---

async function callSkeptic(prompt, systemPrompt) {
  try {
    const response = await withTimeout(
      openai.chat.completions.create({
        model: SKEPTIC_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt },
        ],
        max_completion_tokens: 4096,
      }),
      CALL_TIMEOUT_MS
    );
    return {
      ok: true,
      text: response.choices[0].message.content,
      tokens: response.usage?.total_tokens || 0,
    };
  } catch (err) {
    return { ok: false, text: `[SKEPTIC ERROR: ${err.message}]`, tokens: 0 };
  }
}

async function callSteelman(prompt, systemPrompt, enableSearch = false) {
  // If steelman is configured as OpenAI-compatible (Grok, Groq, Claude, etc.)
  if (STEELMAN_PROVIDER === "openai" && steelmanOpenAI) {
    try {
      const response = await withTimeout(
        steelmanOpenAI.chat.completions.create({
          model: STEELMAN_MODEL,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: prompt },
          ],
          max_completion_tokens: 4096,
        }),
        CALL_TIMEOUT_MS
      );
      return {
        ok: true,
        text: response.choices[0].message.content,
        tokens: response.usage?.total_tokens || 0,
      };
    } catch (err) {
      return { ok: false, text: `[STEELMAN ERROR: ${err.message}]`, tokens: 0 };
    }
  }

  // Default: Gemini (has Google Search grounding)
  try {
    const config = {
      model: STEELMAN_MODEL,
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
    return {
      ok: true,
      text: result.response.text(),
      tokens: (usage?.promptTokenCount || 0) + (usage?.candidatesTokenCount || 0),
    };
  } catch (err) {
    return { ok: false, text: `[STEELMAN ERROR: ${err.message}]`, tokens: 0 };
  }
}

// --- Evidence gathering via web search ---

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

  return callSteelman(
    searchPrompt,
    "You are a research assistant. Compile factual information from web search results. Be precise and cite sources. Do not give opinions or recommendations.",
    true
  );
}

// --- MCP server ---

const server = new McpServer({
  name: "debate",
  version: "1.0.0",
});

const TOOL_DESCRIPTION = `Adversarial AI debate: stress-test any decision with two frontier models arguing from opposite sides, grounded in live web search. One model attacks your plan (Skeptic), the other finds its strongest form then stress-tests it (Steelman). Both receive search-grounded evidence and must distinguish VERIFIED facts from UNVERIFIED claims.

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

Returns structured debate transcript. You (Claude) MUST synthesize using the constrained format at the end of the transcript. Do not free-form summarize.`;

server.tool(
  "debate",
  TOOL_DESCRIPTION,
  {
    context: z.string().describe("The FULL context: plan, decision, strategy, or situation. Include ALL relevant details, background, constraints, and stakes. The models only know what you send them."),
    question: z.string().optional().describe("Specific question to focus the debate. If omitted, models do a general critical analysis."),
    domain: z.string().optional().describe("Domain expertise to inject into both analysts. Examples: 'tax attorney', 'corporate lawyer', 'financial advisor', 'systems architect', 'investment analyst'."),
    current_leaning: z.string().optional().describe("What the user is currently leaning toward. The Skeptic will specifically attack this leaning to counter confirmation bias."),
  },
  async ({ context, question, domain, current_leaning }) => {
    const sections = [];
    let totalTokens = 0;
    const focusQuestion = question || "Analyze this thoroughly. What are the problems? What's missing? What are the risks? What should be done differently?";

    const skepticPrompt = buildSkepticPrompt(domain, current_leaning);
    const steelmanPrompt = buildSteelmanPrompt(domain);

    // Phase 0: Evidence gathering via web search
    let evidencePack = null;
    const evidence = await gatherEvidence(context, question);
    totalTokens += evidence.tokens;

    if (evidence.ok) {
      evidencePack = evidence.text;
      sections.push(
        `# DEBATE TRANSCRIPT`,
        `_Models: Analyst A (${SKEPTIC_MODEL}, Principled Skeptic${domain ? `, ${domain}` : ""}) | Analyst B (${STEELMAN_MODEL}, Steelman Analyst${domain ? `, ${domain}` : ""})_`,
        current_leaning ? `_Decision-maker's current leaning: "${current_leaning}" (Skeptic will specifically attack this)_\n` : `\n`,
        `## Phase 0: Evidence Gathered (via Google Search)\n`,
        `${evidencePack}\n`,
      );
    } else {
      sections.push(
        `# DEBATE TRANSCRIPT`,
        `_Models: Analyst A (${SKEPTIC_MODEL}, Principled Skeptic${domain ? `, ${domain}` : ""}) | Analyst B (${STEELMAN_MODEL}, Steelman Analyst${domain ? `, ${domain}` : ""})_`,
        current_leaning ? `_Decision-maker's current leaning: "${current_leaning}" (Skeptic will specifically attack this)_\n` : `\n`,
        `_Note: Web search unavailable (${evidence.text}). Debate proceeding without search-grounded evidence._\n`,
      );
    }

    // Round 1: Independent analysis (parallel)
    const evidenceSection = evidencePack
      ? `\nVERIFIED EVIDENCE (from web search - cite this when making factual claims, mark claims not supported by this evidence as UNVERIFIED):\n${evidencePack}\n\n`
      : "";

    const r1Prompt = `CONTEXT:\n\n${context}${evidenceSection}\nQUESTION: ${focusQuestion}\n\nProvide your analysis. Structure it with clear sections.`;

    const [skepticR1, steelmanR1] = await Promise.all([
      callSkeptic(r1Prompt, skepticPrompt),
      callSteelman(r1Prompt, steelmanPrompt),
    ]);
    totalTokens += skepticR1.tokens + steelmanR1.tokens;

    sections.push(
      `## Round 1: Independent Analysis\n`,
      `### Analyst A (Skeptic)\n${skepticR1.text}\n`,
      `### Analyst B (Steelman)\n${steelmanR1.text}`,
    );

    if (!skepticR1.ok && !steelmanR1.ok) {
      sections.push(`\n---\n_Both models errored. Check your OPENAI_API_KEY and GEMINI_API_KEY._`);
      return { content: [{ type: "text", text: sections.join("\n") }] };
    }
    if (!skepticR1.ok || !steelmanR1.ok) {
      sections.push(`\n---\n_One model errored. Returning partial results. ${totalTokens.toLocaleString()} tokens used._`);
      return { content: [{ type: "text", text: sections.join("\n") }] };
    }

    // Round 2: Anonymized cross-examination (parallel)
    const crossExamInstructions = `ORIGINAL QUESTION (re-stated to prevent drift): ${focusQuestion}

ORIGINAL CONTEXT:
${context}
${evidencePack ? `\nVERIFIED EVIDENCE (from web search):\n${evidencePack}\n` : ""}
YOUR PREVIOUS ANALYSIS:
{MY_ANALYSIS}

ANOTHER ANALYST'S ANALYSIS:
{OTHER_ANALYSIS}

RESPOND TO THE OTHER ANALYST. Follow these steps exactly:
1. STEELMAN: Restate the other analyst's single strongest argument in your own words. Prove you understood it.
2. CONCEDE: Name anything they caught that you missed. Be honest.
3. DISAGREE: For each point of disagreement, state the specific claim you reject and your specific counter-evidence or reasoning. Distinguish VERIFIED facts (from evidence) from UNVERIFIED claims.
4. UPDATE: State what (if anything) you changed your mind about.
5. FIRMEST BELIEF: What is the one thing you are most confident about? Rate: HIGH/MEDIUM/LOW.
6. UNRESOLVED: Name the top 1-2 disagreements that matter most and have not been settled.`;

    const [skepticR2, steelmanR2] = await Promise.all([
      callSkeptic(
        crossExamInstructions
          .replace("{MY_ANALYSIS}", skepticR1.text)
          .replace("{OTHER_ANALYSIS}", steelmanR1.text),
        skepticPrompt
      ),
      callSteelman(
        crossExamInstructions
          .replace("{MY_ANALYSIS}", steelmanR1.text)
          .replace("{OTHER_ANALYSIS}", skepticR1.text),
        steelmanPrompt
      ),
    ]);
    totalTokens += skepticR2.tokens + steelmanR2.tokens;

    sections.push(
      `\n---\n`,
      `## Round 2: Cross-Examination (anonymized)\n`,
      `### Analyst A (Skeptic) responds\n${skepticR2.text}\n`,
      `### Analyst B (Steelman) responds\n${steelmanR2.text}`,
    );

    // Constrained synthesis instructions
    sections.push(
      `\n---\n`,
      `## Synthesis Required`,
      ``,
      `Synthesize this debate using the following format. Do not free-form summarize. Do not invent a compromise if none exists. If the analysts fundamentally disagree, present both paths.`,
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
      `**Unverified Claims to Check:** [Any factual claims not grounded in the web search evidence]`,
      ``,
      `_${totalTokens.toLocaleString()} total tokens used across models._`,
    );

    return { content: [{ type: "text", text: sections.join("\n") }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
