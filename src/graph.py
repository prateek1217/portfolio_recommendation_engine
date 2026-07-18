"""
LangGraph agent — orchestrates the market-signal -> recommendation pipeline.

Graph topology (linear, with an internal fan-out over investors):

    fetch_event -> analyze_impact -> match_investors -> recommend -> validate -> format

Each node updates a shared AgentState (a TypedDict). The two reasoning nodes
(analyze_impact, recommend) call the NVIDIA Nemotron LLM via llm_client; the
match/validate/format nodes are deterministic Python.

Design choices:
- LangGraph gives an explicit, inspectable state machine and streamable steps,
  which suits an "agentic" pipeline better than an ad-hoc loop.
- The product-universe constraint is enforced in `validate` (code), not just the
  prompt — see validate.py.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# Allow importing the repo-root llm_client.py both as a module and as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import NvidiaLLMClient  # noqa: E402

from . import config, data, prompts  # noqa: E402
from .agentql_client import AgentQLNewsClient  # noqa: E402
from .validate import validate_all, validate_recommendation  # noqa: E402


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # inputs
    field_name: str
    news_url: Optional[str]
    use_mock: Optional[bool]
    # working data
    event: dict
    impact: dict
    affected_investors: List[dict]
    recommendations: List[dict]
    violations: Dict[str, List[str]]
    # trace for UI / debugging
    trace: List[str]


# ---------------------------------------------------------------------------
# Robust JSON extraction (Nemotron may wrap JSON in prose or fences)
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("empty LLM response")
    # Strip markdown fences.
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # Fast path.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class Nodes:
    """Bundles the LLM + clients so nodes share one instance per run."""

    def __init__(self, llm: Optional[NvidiaLLMClient] = None,
                 news: Optional[AgentQLNewsClient] = None):
        self._llm = llm
        self._news = news

    # lazy so importing/constructing the graph never requires API keys
    @property
    def llm(self) -> NvidiaLLMClient:
        if self._llm is None:
            self._llm = NvidiaLLMClient()
        return self._llm

    def news(self, use_mock: Optional[bool]) -> AgentQLNewsClient:
        if self._news is None:
            # Share our LLM so article-relevance scoring reuses one client.
            # Only pass a live LLM (skip in mock mode / when no key) so building
            # the news client never forces an NVIDIA key.
            llm = None
            if not use_mock:
                try:
                    llm = self.llm
                except Exception:
                    llm = None
            self._news = AgentQLNewsClient(use_mock=use_mock, llm=llm)
        return self._news

    # -- node 1 -----------------------------------------------------------
    def fetch_event(self, state: AgentState) -> AgentState:
        field_name = state["field_name"]
        focus = config.MARKET_FIELDS.get(field_name, {}).get("news_focus", field_name)
        event = self.news(state.get("use_mock")).fetch_event(
            field_name, focus, url=state.get("news_url")
        )
        trace = state.get("trace", []) + [
            f"Fetched event via {event.get('source')}: {event.get('headline')}"
        ]
        return {"event": event, "trace": trace}

    # -- node 2 -----------------------------------------------------------
    def analyze_impact(self, state: AgentState) -> AgentState:
        known = sorted({s for meta in config.MARKET_FIELDS.values()
                        for s in meta["sectors"]})
        prompt = prompts.build_impact_prompt(state["event"], known)
        raw = self.llm.generate(
            prompt, system_prompt=prompts.IMPACT_SYSTEM,
            temperature=0.3, max_tokens=1500,
        )
        impact = _extract_json(raw)
        sectors = [s["sector"] for s in impact.get("affected_sectors", [])]
        trace = state.get("trace", []) + [
            f"Impact: {impact.get('event_direction')} | sectors: {', '.join(sectors)}"
        ]
        return {"impact": impact, "trace": trace}

    # -- node 3 (deterministic) ------------------------------------------
    def match_investors(self, state: AgentState) -> AgentState:
        sectors = [s["sector"] for s in state["impact"].get("affected_sectors", [])]
        affected = data.match_investors_to_sectors(sectors)
        trace = state.get("trace", []) + [
            f"Matched {len(affected)} affected investors."
        ]
        return {"affected_investors": affected, "trace": trace}

    # -- node 4 -----------------------------------------------------------
    def _recommend_one(self, event: dict, impact: dict, inv: dict,
                       universe: str) -> dict:
        """Generate + shape one investor's recommendation (runs in a thread)."""
        prompt = prompts.build_reco_prompt(event, impact, inv, universe)
        try:
            raw = self.llm.generate(
                prompt, system_prompt=prompts.RECO_SYSTEM,
                temperature=0.4, max_tokens=1200,
            )
            rec = _extract_json(raw)
        except Exception as e:
            rec = {
                "investor_id": inv["investor_id"],
                "headline": "Could not generate recommendation",
                "severity": "low",
                "rationale": f"LLM error: {e}",
                "actions": [],
            }
        # carry forward context the UI needs
        rec["investor_id"] = inv["investor_id"]
        rec["name"] = inv["name"]
        rec["risk_profile"] = inv["risk_profile"]
        rec["total_exposure_pct"] = inv["total_exposure_pct"]
        rec["matched_sectors"] = inv["matched_sectors"]
        rec["impacted_holdings"] = inv["impacted_holdings"]
        return rec

    def recommend(self, state: AgentState) -> AgentState:
        universe = data.catalogue_for_prompt()
        investors = state["affected_investors"]
        event, impact = state["event"], state["impact"]

        # Per-investor calls are independent → fan out concurrently. This turns
        # ~12s x N sequential into roughly ceil(N / workers) x 12s. Results are
        # re-sorted to the original (exposure-desc) order afterwards.
        recs: List[dict] = [None] * len(investors)  # type: ignore
        if investors:
            workers = min(config.RECO_MAX_WORKERS, len(investors))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._recommend_one, event, impact, inv, universe): i
                    for i, inv in enumerate(investors)
                }
                for fut in as_completed(futures):
                    recs[futures[fut]] = fut.result()

        trace = state.get("trace", []) + [
            f"Generated {len(recs)} recommendations "
            f"({min(config.RECO_MAX_WORKERS, len(investors) or 1)} in parallel)."
        ]
        return {"recommendations": recs, "trace": trace}

    # -- node 5 (deterministic guardrail) --------------------------------
    def validate(self, state: AgentState) -> AgentState:
        recs, violations = validate_all(state["recommendations"])
        note = (
            f"Guardrail dropped {sum(len(v) for v in violations.values())} "
            f"out-of-universe fund(s)."
            if violations else "Guardrail: all recommended funds in universe."
        )
        trace = state.get("trace", []) + [note]
        return {"recommendations": recs, "violations": violations, "trace": trace}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(nodes: Optional[Nodes] = None):
    nodes = nodes or Nodes()
    g = StateGraph(AgentState)
    g.add_node("fetch_event", nodes.fetch_event)
    g.add_node("analyze_impact", nodes.analyze_impact)
    g.add_node("match_investors", nodes.match_investors)
    g.add_node("recommend", nodes.recommend)
    g.add_node("validate", nodes.validate)

    g.set_entry_point("fetch_event")
    g.add_edge("fetch_event", "analyze_impact")
    g.add_edge("analyze_impact", "match_investors")
    g.add_edge("match_investors", "recommend")
    g.add_edge("recommend", "validate")
    g.add_edge("validate", END)
    return g.compile()


def run_pipeline(
    field_name: str,
    news_url: Optional[str] = None,
    use_mock: Optional[bool] = None,
    nodes: Optional[Nodes] = None,
) -> AgentState:
    """Convenience entry point used by the CLI and the web app."""
    graph = build_graph(nodes)
    initial: AgentState = {
        "field_name": field_name,
        "news_url": news_url,
        "use_mock": use_mock,
        "trace": [],
    }
    return graph.invoke(initial)


# ---------------------------------------------------------------------------
# Streaming entry points (used by the two-button / live-updating UI)
# ---------------------------------------------------------------------------
def fetch_event_and_impact(
    field_name: str,
    news_url: Optional[str] = None,
    use_mock: Optional[bool] = None,
    nodes: Optional[Nodes] = None,
) -> dict:
    """
    Stage 1 (the first button): pull the news event, analyse its sector impact,
    and pre-compute which investors are affected — WITHOUT generating any
    recommendations yet. Returns everything the UI needs to render the event
    card immediately and know how many investor calls are coming.
    """
    nodes = nodes or Nodes()
    s1 = nodes.fetch_event({"field_name": field_name, "news_url": news_url,
                            "use_mock": use_mock, "trace": []})
    s2 = nodes.analyze_impact({**s1, "field_name": field_name})
    s3 = nodes.match_investors({**s1, **s2})
    return {
        "event": s1["event"],
        "impact": s2["impact"],
        "affected_investors": s3["affected_investors"],
        "trace": s3.get("trace", []),
    }


def stream_recommendations(
    event: dict,
    impact: dict,
    affected_investors: List[dict],
    nodes: Optional[Nodes] = None,
):
    """
    Stage 2 (the second button): generate recommendations concurrently and
    YIELD each one the moment it is ready, already passed through the
    product-universe guardrail. Lets the UI paint cards as they arrive instead
    of waiting for the whole batch.
    """
    nodes = nodes or Nodes()
    universe = data.catalogue_for_prompt()
    if not affected_investors:
        return
    workers = min(config.RECO_MAX_WORKERS, len(affected_investors))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(nodes._recommend_one, event, impact, inv, universe)
            for inv in affected_investors
        ]
        for fut in as_completed(futures):
            rec = fut.result()
            # Apply the hard guardrail per-card before it leaves the server.
            validate_recommendation(rec)
            yield rec


if __name__ == "__main__":
    # Smoke test in mock mode (no keys needed for news; LLM key still required).
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="Energy & Oil")
    ap.add_argument("--mock", action="store_true", help="use cached news")
    args = ap.parse_args()

    result = run_pipeline(args.field, use_mock=args.mock or None)
    print("\n=== TRACE ===")
    for t in result.get("trace", []):
        print("  •", t)
    print("\n=== EVENT ===")
    print(" ", result["event"]["headline"])
    print("\n=== RECOMMENDATIONS ===")
    for r in result.get("recommendations", []):
        print(f"\n[{r.get('severity','?').upper()}] {r['name']} "
              f"({r['risk_profile']}, {r['total_exposure_pct']}% exposed)")
        print("  ", r.get("headline"))
        for a in r.get("actions", []):
            print(f"    -> {a.get('type','?').upper()} {a.get('fund_id')} "
                  f"({a.get('fund_name','')}): {a.get('reason','')}")
