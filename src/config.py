"""
Central configuration + path resolution for the Portfolio Recommendation Agent.

All tunables (file paths, sector taxonomy, feature flags) live here so the rest
of the codebase never hardcodes paths or magic strings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_ROOT / "Task"

PORTFOLIOS_CSV = TASK_DIR / "sample_portfolios.csv"
CATALOGUE_CSV = TASK_DIR / "product_catalogue.csv"

# ---------------------------------------------------------------------------
# AgentQL (by TinyFish) — market news sourcing
# ---------------------------------------------------------------------------
AGENTQL_API_KEY = os.getenv("AGENTQL_API_KEY", "")
AGENTQL_API_URL = os.getenv("AGENTQL_API_URL", "https://api.agentql.com/v1/query-data")

# When true (or when no AgentQL key is present) the news client returns cached
# sample events instead of calling the live API. Lets the pipeline/UI run before
# keys are added.
USE_MOCK_NEWS = os.getenv("USE_MOCK_NEWS", "false").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# Agent tuning
# ---------------------------------------------------------------------------
# Per-investor recommendation calls are independent and run concurrently.
# Higher = faster wall-clock, but watch the LLM provider's rate limits.
RECO_MAX_WORKERS = int(os.getenv("RECO_MAX_WORKERS", "12"))

# Article relevance scoring (which scraped article best matches the field).
#   RELEVANCE_USE_LLM : True -> score each article with the NVIDIA LLM (one call
#                       per article, run in parallel); False -> fast keyword
#                       overlap. LLM path auto-falls-back to keywords on error.
#   RELEVANCE_MAX_ARTICLES : cap how many scraped articles we score (protects
#                       against huge pages / rate limits — the first N are used).
#   RELEVANCE_WORKERS : concurrency for the per-article scoring calls.
#   RELEVANCE_MIN_SCORE : below this the page is treated as having nothing
#                       relevant, and we fall back to a cached event.
RELEVANCE_USE_LLM = os.getenv("RELEVANCE_USE_LLM", "true").strip().lower() == "true"
RELEVANCE_MAX_ARTICLES = int(os.getenv("RELEVANCE_MAX_ARTICLES", "20"))
RELEVANCE_WORKERS = int(os.getenv("RELEVANCE_WORKERS", "8"))
RELEVANCE_MIN_SCORE = int(os.getenv("RELEVANCE_MIN_SCORE", "30"))

# AgentQL network timeouts.
#   connect: fail fast if the host can't be reached (flaky DNS on some networks)
#   read:    how long to wait for AgentQL to scrape + return. Generous by design
#            (markets pages are heavy) — no artificial short cap on the fetch.
NEWS_CONNECT_TIMEOUT = int(os.getenv("NEWS_CONNECT_TIMEOUT", "15"))
NEWS_READ_TIMEOUT = int(os.getenv("NEWS_READ_TIMEOUT", "180"))

# How many times to attempt the live news call before using a cached event.
# The AgentQL host occasionally fails to resolve/connect transiently.
NEWS_MAX_RETRIES = int(os.getenv("NEWS_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Sector taxonomy
# ---------------------------------------------------------------------------
# The canonical sectors that appear in product_catalogue.csv's `primary_sectors`
# column, grouped into the "fields" a user can pick from the UI dropdown. Each
# field maps to a news search focus and to the underlying catalogue sectors.
#
# NOTE: matching between an event's affected sectors and a fund's primary_sectors
# is done case-insensitively and by substring at runtime (see data.py), so this
# list only needs to drive the UI + news query — not exhaustive string matching.
MARKET_FIELDS = {
    "Energy & Oil": {
        "sectors": ["Energy", "Power"],
        "news_focus": "crude oil prices, energy sector, OPEC, fuel, power utilities in India",
    },
    "Banking & Financials": {
        "sectors": ["Financial Services", "Private Banks", "NBFCs", "Insurance"],
        "news_focus": "RBI policy, interest rates, banks, NBFCs, financial services in India",
    },
    "Information Technology": {
        "sectors": ["IT", "IT Services", "Software"],
        "news_focus": "IT sector, software exporters, TCS Infosys, US tech spending, rupee",
    },
    "Healthcare & Pharma": {
        "sectors": ["Healthcare", "Pharmaceuticals", "Hospitals"],
        "news_focus": "pharma sector, healthcare, drug pricing, USFDA, hospitals in India",
    },
    "Metals & Materials": {
        "sectors": ["Metals", "Materials"],
        "news_focus": "metals, steel, commodity prices, mining, materials sector in India",
    },
    "Automobiles": {
        "sectors": ["Automobiles", "Consumer Discretionary"],
        "news_focus": "auto sector, vehicle sales, EV, automobiles in India",
    },
    "Infrastructure & Capital Goods": {
        "sectors": ["Industrials", "Capital Goods", "Construction"],
        "news_focus": "infrastructure, capital goods, construction, industrials in India",
    },
    "FMCG & Consumption": {
        "sectors": ["FMCG", "Consumer Discretionary"],
        "news_focus": "FMCG, consumption, rural demand, consumer goods in India",
    },
    "Broad Market / Index": {
        "sectors": ["Financial Services", "IT", "Energy", "FMCG"],
        "news_focus": "Nifty 50 Sensex index move, broad Indian stock market",
    },
}

# Default news source AgentQL will scrape (public, no auth). Economic Times'
# markets page renders a clean list of real market headlines that AgentQL
# extracts reliably; Moneycontrol's markets page is JS-heavy and returns junk.
DEFAULT_NEWS_URL = "https://economictimes.indiatimes.com/markets/stocks/news"
