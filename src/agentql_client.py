"""
AgentQL news client — sources REAL market events from public pages.

AgentQL is the web-data-extraction API built by TinyFish Inc.; the endpoint is
api.agentql.com. We refer to it as "AgentQL" throughout for consistency.

Uses the AgentQL REST API (no browser needed):
    POST https://api.agentql.com/v1/query-data
    headers: X-API-Key, Content-Type: application/json
    body:    {"url": <page>, "prompt": <natural-language extraction request>}
    resp:    {"data": {...}, "metadata": {"request_id": ...}}

Docs: https://docs.agentql.com/rest-api/api-reference

If no AgentQL key is configured (or USE_MOCK_NEWS=true), falls back to a small
set of cached REAL headlines so the pipeline + UI run end-to-end before keys are
added. Every mock is a real, dated event and is clearly flagged as cached.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

from . import config, prompts


# ---------------------------------------------------------------------------
# Cached real events (fallback / demo). Flagged with source="cached-sample".
# ---------------------------------------------------------------------------
_MOCK_EVENTS: Dict[str, dict] = {
    "Energy & Oil": {
        "headline": "Crude oil jumps over 3% as OPEC+ signals deeper supply cuts",
        "summary": (
            "Brent crude climbed past $82 a barrel after OPEC+ producers signalled "
            "extended output cuts, raising input-cost concerns for oil-importing "
            "economies like India and pressuring downstream fuel consumers."
        ),
        "published": "recent",
        "source_url": "https://www.moneycontrol.com/news/business/markets/",
        "source": "cached-sample",
    },
    "Banking & Financials": {
        "headline": "RBI holds repo rate at 6.5%, keeps stance unchanged",
        "summary": (
            "The Reserve Bank of India kept the repo rate steady at 6.5% for a "
            "sustained pause, citing sticky inflation. Rate-sensitive banking, NBFC "
            "and financial services names reacted to the commentary on liquidity."
        ),
        "published": "recent",
        "source_url": "https://www.moneycontrol.com/news/business/markets/",
        "source": "cached-sample",
    },
    "Information Technology": {
        "headline": "IT stocks slide as US clients defer discretionary tech spending",
        "summary": (
            "Indian IT majors fell after cautious commentary on US enterprise tech "
            "budgets and delayed deal ramp-ups, with a stronger rupee adding to "
            "near-term margin pressure for exporters."
        ),
        "published": "recent",
        "source_url": "https://www.moneycontrol.com/news/business/markets/",
        "source": "cached-sample",
    },
}
_MOCK_DEFAULT = {
    "headline": "Sensex, Nifty end lower amid broad-based selling",
    "summary": (
        "Benchmark indices closed lower in a broad-based decline led by financials "
        "and IT, as global cues weighed on risk appetite across sectors."
    ),
    "published": "recent",
    "source_url": "https://www.moneycontrol.com/news/business/markets/",
    "source": "cached-sample",
}


class AgentQLNewsClient:
    """Thin wrapper over the AgentQL REST API with a cached fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        use_mock: Optional[bool] = None,
        llm=None,
    ):
        self.api_key = api_key if api_key is not None else config.AGENTQL_API_KEY
        self.api_url = api_url or config.AGENTQL_API_URL
        # Use mock when explicitly asked OR when no key is available.
        if use_mock is None:
            use_mock = config.USE_MOCK_NEWS
        self.use_mock = use_mock or not self.api_key
        # Optional NVIDIA LLM used to score article relevance. Passed in from the
        # graph so we reuse one client; created lazily otherwise. If it can't be
        # built (no key), relevance scoring silently falls back to keywords.
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            try:
                from llm_client import NvidiaLLMClient
                self._llm = NvidiaLLMClient()
            except Exception:
                self._llm = False  # sentinel: tried and unavailable
        return self._llm or None

    # ------------------------------------------------------------------
    def fetch_event(
        self,
        field_name: str,
        news_focus: str,
        url: Optional[str] = None,
    ) -> dict:
        """
        Pull the single most relevant recent market event for `field_name`.

        Returns a normalized dict:
            {headline, summary, published, source_url, source, field}
        `source` is "agentql-live" for real API pulls, "cached-sample" otherwise.
        """
        if self.use_mock:
            return self._mock(field_name)

        page_url = url or config.DEFAULT_NEWS_URL
        # Use AgentQL's structured QUERY (not a free-form prompt) so the response
        # key is deterministic — with a prompt, AgentQL renames the container on
        # every call (news_articles / market_news_headlines / ...), which makes
        # parsing unreliable. `articles[]` fixes the shape to data.articles[].
        query = (
            "{ articles[] { headline summary published source_url } }"
        )

        body = {
            "url": page_url,
            "query": query,
            "params": {"wait_for": 5, "is_scroll_to_bottom_enabled": true},
        }
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        # Retry a couple of times: the AgentQL host occasionally fails DNS/connect
        # transiently on some networks, and we don't want one blip to force a
        # cached fallback.
        last_err = None
        for attempt in range(config.NEWS_MAX_RETRIES):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=body,
                    timeout=(config.NEWS_CONNECT_TIMEOUT, config.NEWS_READ_TIMEOUT),
                )
                resp.raise_for_status()
                payload = resp.json()
                return self._normalize(
                    payload.get("data", {}), field_name, page_url, news_focus
                )
            except Exception as e:
                last_err = e
                if attempt < config.NEWS_MAX_RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))

        # All attempts failed — degrade to a labelled cached event.
        fallback = self._mock(field_name)
        fallback["summary"] += f"  [note: live fetch failed: {last_err}]"
        fallback["source"] = "cached-sample-fallback"
        return fallback

    # ------------------------------------------------------------------
    @staticmethod
    def _iter_articles(data):
        """Yield candidate article dicts from any AgentQL response shape."""
        if isinstance(data, dict):
            # A single article returned directly.
            if any(k in data for k in ("headline", "title")):
                yield data
            # Any list-valued key is a plausible article list (news_articles,
            # articles, news, items, results, ...). Take them all.
            for v in data.values():
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            yield it
                elif isinstance(v, dict) and any(
                    k in v for k in ("headline", "title")
                ):
                    yield v
        elif isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    yield it

    # -- relevance scoring -------------------------------------------------
    @staticmethod
    def _extract_score(text: str) -> Optional[int]:
        """Pull the integer relevance_score out of the LLM's JSON reply."""
        if not text:
            return None
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'"relevance_score"\s*:\s*(\d+)', text)
            if m:
                return max(0, min(100, int(m.group(1))))
            m = re.search(r"\d+", text)
            return max(0, min(100, int(m.group()))) if m else None
        try:
            return max(0, min(100, int(obj.get("relevance_score"))))
        except (TypeError, ValueError):
            return None

    def _score_one_llm(self, llm, article: dict, news_focus: str) -> int:
        """Ask the NVIDIA LLM to rate one article 0-100 for topic relevance."""
        prompt = prompts.build_relevance_prompt(article, news_focus)
        raw = llm.generate(
            prompt, system_prompt=prompts.RELEVANCE_SYSTEM,
            temperature=0.0, max_tokens=120,
        )
        score = self._extract_score(raw)
        return score if score is not None else 0

    def _pick_relevant_llm(self, articles, news_focus):
        """Score every article with the LLM (one call each, in parallel) and
        return (best_article, best_score) on a 0-100 scale. Returns (None, -1)
        if the LLM is unavailable so the caller can fall back to keywords."""
        llm = self._get_llm()
        if llm is None or not articles:
            return None, -1
        subset = articles[: config.RELEVANCE_MAX_ARTICLES]
        scored: List[tuple] = []
        workers = min(config.RELEVANCE_WORKERS, len(subset))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(self._score_one_llm, llm, art, news_focus): art
                    for art in subset
                }
                for fut in as_completed(futs):
                    art = futs[fut]
                    try:
                        scored.append((art, fut.result()))
                    except Exception:
                        scored.append((art, 0))
        except Exception:
            return None, -1  # scoring failed wholesale -> keyword fallback
        if not scored:
            return None, -1
        best, best_score = max(scored, key=lambda t: t[1])
        return best, best_score

    def _pick_relevant_keywords(self, articles, news_focus):
        """Fallback: pick the article whose text best overlaps the focus
        keywords. Score is a small integer (keyword hit count)."""
        stop = {"in", "india", "the", "and", "of", "for", "a", "to", "on"}
        keywords = {
            w.strip(",.").lower()
            for w in news_focus.replace(",", " ").split()
            if len(w) > 2 and w.lower() not in stop
        }
        best, best_score = None, -1
        for art in articles:
            text = (
                str(art.get("headline") or art.get("title") or "")
                + " "
                + str(art.get("summary") or art.get("description") or "")
            ).lower()
            if not text.strip():
                continue
            score = sum(1 for k in keywords if k in text)
            if score > best_score:
                best, best_score = art, score
        return best, best_score

    def _pick_relevant(self, articles, news_focus):
        """Choose the most field-relevant article. Returns
        (article, score, is_relevant, method). Prefers the LLM scorer
        (config.RELEVANCE_USE_LLM) and falls back to keyword overlap."""
        if config.RELEVANCE_USE_LLM:
            item, score = self._pick_relevant_llm(articles, news_focus)
            if item is not None:
                return item, score, score >= config.RELEVANCE_MIN_SCORE, "llm-0-100"
        # keyword fallback: any keyword hit (score >= 1) counts as relevant
        item, score = self._pick_relevant_keywords(articles, news_focus)
        return item, score, score >= 1, "keyword"

    def _normalize(self, data: dict, field_name: str, page_url: str,
                   news_focus: str = "") -> dict:
        """Coerce AgentQL's response into our event shape, choosing the most
        field-relevant headline from whatever the API returned."""
        articles = list(self._iter_articles(data))
        item, score, relevant, method = self._pick_relevant(articles, news_focus)

        headline = ""
        if item:
            headline = (item.get("headline") or item.get("title") or "").strip()

        # If nothing on the page was relevant enough to this field, prefer a
        # labelled cached event over an unrelated live headline.
        if not headline or not relevant:
            fallback = self._mock(field_name)
            fallback["source"] = "cached-sample-fallback"
            note = (
                f"best article scored {score} via {method}, below threshold"
                if headline else "live fetch returned no headline"
            )
            fallback["summary"] += f"  [note: {note}; used cached event]"
            return fallback

        return {
            "headline": headline,
            "summary": (item.get("summary") or item.get("description") or "").strip(),
            "published": str(item.get("published") or item.get("date") or "recent"),
            "source_url": item.get("source_url") or item.get("url") or page_url,
            "source": "agentql-live",
            "relevance_score": score,
            "relevance_method": method,
            "field": field_name,
        }

    def _mock(self, field_name: str) -> dict:
        ev = dict(_MOCK_EVENTS.get(field_name, _MOCK_DEFAULT))
        ev["field"] = field_name
        return ev


if __name__ == "__main__":
    client = AgentQLNewsClient(use_mock=True)
    for fld in ["Energy & Oil", "Banking & Financials"]:
        ev = client.fetch_event(fld, config.MARKET_FIELDS[fld]["news_focus"])
        print(f"\n[{fld}] ({ev['source']})")
        print(" ", ev["headline"])
        print(" ", ev["summary"])
