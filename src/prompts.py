"""
Prompt builders for the two LLM reasoning steps.

Kept separate from graph.py so the prompt engineering is easy to read and tune
without touching orchestration logic.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Step 0: Article relevance scoring (which scraped article matches the field?)
# ---------------------------------------------------------------------------
RELEVANCE_SYSTEM = (
    "You are a financial-news relevance classifier for an Indian "
    "wealth-management firm. You are given a market TOPIC (a set of focus "
    "keywords/themes for one sector) and a SINGLE news article (headline + "
    "summary). Your only job is to rate how relevant that article is to that "
    "topic, on an integer scale from 0 to 100.\n\n"
    "Definition of relevance: an article is relevant when its PRIMARY subject "
    "is the topic's sector/theme, or a market driver that directly moves that "
    "sector (e.g. crude-oil prices for an Energy topic; RBI rate decisions for "
    "a Banking topic). A passing, incidental mention does NOT make it relevant.\n\n"
    "Scoring rubric:\n"
    "  90-100 : directly and primarily about the topic.\n"
    "  60-89  : substantially about the topic or its main driver.\n"
    "  30-59  : partially related / topic is secondary.\n"
    "  1-29   : only an incidental or tangential mention.\n"
    "  0      : unrelated (other sectors, sports, politics, gadgets, lifestyle).\n\n"
    "Rules:\n"
    "- Judge ONLY topic match, not the article's importance or quality.\n"
    "- If the article is off-topic or non-market, the score MUST be 0.\n"
    "- Do not use outside knowledge beyond the headline and summary given.\n"
    "- Output ONLY valid JSON, no prose, no markdown fences."
)


def build_relevance_prompt(article: dict, news_focus: str) -> str:
    headline = str(article.get("headline") or article.get("title") or "").strip()
    summary = str(article.get("summary") or article.get("description") or "").strip()
    return f"""TOPIC (focus keywords/themes): {news_focus}

ARTICLE
  Headline: {headline}
  Summary:  {summary or '(no summary provided)'}

Return JSON with EXACTLY this shape:
{{
  "relevance_score": <integer 0-100>,
  "reason": "<=12 words justifying the score>"
}}

Output ONLY the JSON object, no prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Step 1: Impact analysis (event -> affected sectors, incl. second-order)
# ---------------------------------------------------------------------------
IMPACT_SYSTEM = (
    "You are a sell-side market analyst for an Indian wealth-management firm. "
    "You reason precisely about how a market event propagates across sectors, "
    "including SECOND-ORDER effects (e.g. an oil-price shock raises fuel costs "
    "and hurts airlines, paints, logistics; a rate hike pressures NBFCs and "
    "real estate). You only output valid JSON."
)


def build_impact_prompt(event: dict, known_sectors: List[str]) -> str:
    return f"""A market event has occurred. Analyse which sectors it materially affects.

EVENT
  Headline: {event.get('headline')}
  Summary:  {event.get('summary')}
  Field:    {event.get('field')}

The firm classifies holdings using these sectors (use these exact labels where
they apply; you may also name a closely-related one if clearly relevant):
{', '.join(known_sectors)}

Return JSON with EXACTLY this shape:
{{
  "event_direction": "risk-off" | "risk-on" | "mixed",
  "affected_sectors": [
    {{
      "sector": "<sector label>",
      "impact": "negative" | "positive",
      "order": "direct" | "second-order",
      "rationale": "<one concise sentence>"
    }}
  ],
  "one_line_summary": "<plain-English summary of the event's market impact>"
}}

Rules:
- Include BOTH direct and second-order sectors. Aim for 3-7 sectors total.
- Be specific about the transmission mechanism in each rationale.
- Output ONLY the JSON object, no prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Step 2: Per-investor recommendation (constrained to product_universe)
# ---------------------------------------------------------------------------
RECO_SYSTEM = (
    "You are an assistant to a Relationship Manager (RM) at an Indian wealth "
    "firm. You produce concise, RM-facing recommendations. You may ONLY "
    "recommend funds from the provided product universe — never invent funds or "
    "use outside knowledge. You always output valid JSON."
)


def build_reco_prompt(
    event: dict,
    impact: dict,
    affected_investor: dict,
    product_universe: str,
) -> str:
    holdings_lines = "\n".join(
        f"    - {h['fund_id']} {h['fund_name']}: {h['weight_pct']}% "
        f"(₹{h['current_value_inr']:,.0f}), exposed via {', '.join(h['matched_sectors'])}"
        for h in affected_investor["impacted_holdings"]
    )

    return f"""Generate ONE recommendation for this investor in response to the event.

EVENT: {event.get('headline')}
IMPACT: {impact.get('one_line_summary')}
AFFECTED SECTORS: {', '.join(s['sector'] + ' (' + s['impact'] + ', ' + s['order'] + ')' for s in impact.get('affected_sectors', []))}

INVESTOR
  Name: {affected_investor['name']}  |  Risk profile: {affected_investor['risk_profile']}
  Age: {affected_investor['age']}  |  Life stage: {affected_investor['life_stage']}
  Portfolio value: ₹{affected_investor['portfolio_value_inr']:,.0f}
  Total exposure to affected sectors: {affected_investor['total_exposure_pct']}%
  Impacted holdings:
{holdings_lines}

PRODUCT UNIVERSE — the ONLY funds you may recommend (fund_id | name | category | risk | sectors):
{product_universe}

Return JSON with EXACTLY this shape:
{{
  "investor_id": "{affected_investor['investor_id']}",
  "headline": "<=12 word RM-facing summary of the situation for this investor>",
  "severity": "high" | "medium" | "low",
  "rationale": "<1-2 sentences: why this investor is affected and the recommended direction>",
  "actions": [
    {{
      "type": "trim" | "add" | "switch" | "hold" | "review",
      "fund_id": "<MUST be a fund_id from the product universe above>",
      "reason": "<short reason tied to the event and this investor's profile>"
    }}
  ]
}}

Rules:
- severity should reflect exposure size AND how negatively the event hits them.
- Recommend 1-3 concrete actions. Every fund_id MUST come from the product
  universe list above. Do NOT reference any fund not in that list.
- Tailor to the risk profile (e.g. Conservative -> favour debt/hedges;
  Aggressive -> may rotate within equity).
- Output ONLY the JSON object, no prose, no markdown fences."""
