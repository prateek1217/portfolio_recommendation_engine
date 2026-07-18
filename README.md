<img width="512" height="397" alt="image" src="https://github.com/user-attachments/assets/ba8d8f4a-61f1-4bc9-8d12-3a28cd51e976" />

<img width="863" height="301" alt="image" src="https://github.com/user-attachments/assets/7e96cb05-7425-4d0c-8dbb-992c97441c30" />





# Market Signal → Portfolio Recommendation Agent



An agent that takes a **real market event**, works out which sectors it hits
(directly *and* via second-order effects), finds which investors are materially
affected, and produces a **one-glance, RM-facing recommendation per investor** —
constrained to only the funds in the given product universe.

---

## What it does (end to end)

```
 AgentQL                NVIDIA Nemotron      NVIDIA Nemotron     deterministic      NVIDIA Nemotron     code guardrail
 ──────────────────     ───────────────      ───────────────     ─────────────      ───────────────     ──────────────
  scrape all news   →   score EACH        →  analyse impact:  →  match affected  →  recommend per   →   drop any fund
  articles for the      article 0-100        affected sectors    investors vs.      investor, only      not in the
  chosen field          for relevance,       incl. SECOND-       their holdings     from product        product universe
                        keep the best        ORDER (oil →        (weights)          universe            → RM alerts (UI)
                        (else cached)        autos/airlines)
```

Three of the steps run on **NVIDIA Nemotron** (`llm_client.py`) — relevance
scoring, impact analysis, and the per-investor recommendation; the matching and
the product-universe enforcement are **deterministic Python** so the hard money
constraint never depends on the model behaving.

---

## Architecture

| Layer | File | Responsibility |
|---|---|---|
| **Orchestration** | `src/graph.py` | LangGraph state machine: `fetch_event → analyze_impact → match_investors → recommend → validate`. Shared `AgentState`, streamable, inspectable. |
| **Event sourcing (tool)** | `src/agentql_client.py` | AgentQL (by TinyFish) REST client (`POST /v1/query-data`, `X-API-Key`). Scrapes all articles on the chosen field's news page, then **scores each with the NVIDIA LLM (0–100) for relevance** and keeps the best. Falls back to keyword overlap if the LLM is off, and to cached-real-events if the page has nothing relevant or no key is set. |
| **Reasoning (LLM)** | `src/prompts.py` + `llm_client.py` | Three prompt sets: article relevance scoring, impact analysis (sectors + second-order), and per-investor recommendation. All on NVIDIA Nemotron via the provided `NvidiaLLMClient`. |
| **Data (pre-ingested)** | `src/data.py` | Loads `sample_portfolios.csv` + `product_catalogue.csv`, builds fund→sectors lookup, computes per-investor sector exposure, matches investors to affected sectors. |
| **Guardrail (hard constraint)** | `src/validate.py` | Every recommended `fund_id` must exist in `product_catalogue.csv`. Out-of-universe funds are dropped and recorded; canonical fund names are backfilled. |
| **UI (backend)** | `src/app.py` | FastAPI. Serves the `frontend/` SPA and exposes a **two-step, streaming** API: `POST /api/fetch` (news + impact + affected investors) then `POST /api/stream` (Server-Sent Events — one RM alert card per investor, the moment it's ready). |
| **UI (frontend)** | `frontend/` | `index.html` + `styles.css` + `app.js`. Field dropdown and **two buttons**: **1 Fetch news** shows what AgentQL returned; **2 Analyze** opens the SSE stream and paints each recommendation card live as it arrives. |
| **CLI** | `main.py` | Runs end-to-end on ≥2 events for the deliverable. |

### Why these choices
- **LangGraph** — the task is a fixed pipeline with one fan-out over investors.
  A state graph makes each step explicit, testable, and streamable, and is the
  natural "agentic" framing without over-engineering a free-roaming agent.
- **LLM for impact, code for constraints** — second-order reasoning
  (oil → airlines via fuel) is exactly where an LLM adds value; the "never
  recommend outside the universe" rule is exactly where you must *not* trust an
  LLM, so it is enforced in `validate.py`.
- **AgentQL (by TinyFish)** — one structured query extracts articles from any
  public markets page, so we aren't tied to one news vendor's schema.
- **LLM relevance scoring over keyword matching** — AgentQL returns *every*
  article on the page; picking the one that matches the chosen field needs
  judgement (an oil story is relevant to "Energy" even without the word
  "energy"). Each article is scored 0–100 by the LLM and the best is kept, with
  a keyword-overlap fallback for speed/offline. See `RELEVANCE_*` in `config.py`.

### LangGraph pipeline

A single `StateGraph` where every node reads/writes a shared `AgentState`
(`TypedDict`). Linear, with one internal fan-out over investors in `recommend`:

```
START → fetch_event → analyze_impact → match_investors → recommend → validate → END
        AgentQL +       NVIDIA LLM       data.py           NVIDIA LLM   guardrail
        LLM relevance    (impact)         (code)            (× N, ‖)     (code)
```

`fetch_event` + `analyze_impact` + `match_investors` back the "Fetch news"
button (`/api/fetch`); `recommend` + `validate` back "Analyze & recommend",
streamed per-card over SSE (`/api/stream`). An editable diagram of both the
request flow and this graph is in **`pipeline_flow.excalidraw`**
(open at [excalidraw.com](https://excalidraw.com) → *Open*).

---

## Project structure

```
.
├── llm_client.py            # provided NVIDIA Nemotron client (unmodified)
├── main.py                  # CLI: run the pipeline on ≥2 events
├── run.py                   # launches the FastAPI web app
├── requirements.txt
├── .env.example             # copy to .env and add your keys
├── Task/
│   ├── sample_portfolios.csv   # investors + their holdings (pre-ingested)
│   └── product_catalogue.csv   # the ONLY funds allowed in recommendations
├── frontend/                # web UI (served as static files by app.py)
│   ├── index.html           #   field dropdown + two buttons
│   ├── styles.css
│   └── app.js               #   two-step flow + live SSE streaming
├── src/
│   ├── config.py            # paths, market fields, feature flags, tunables
│   ├── agentql_client.py    # news scraping + LLM relevance scoring
│   ├── prompts.py           # relevance / impact / recommendation prompts
│   ├── graph.py             # LangGraph pipeline + streaming entry points
│   ├── data.py              # CSV loading, exposure, investor↔sector matching
│   ├── validate.py          # product-universe guardrail
│   └── app.py               # FastAPI backend (/api/fetch, /api/stream SSE)
└── pipeline_flow.excalidraw # editable end-to-end + LangGraph diagram
```

---

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
```

`.env` (see `.env.example` for the full list):
```
NVIDIA_API_KEY=...             # required (relevance, impact, recommendation)
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
AGENTQL_API_KEY=...            # optional; without it, cached real events are used
USE_MOCK_NEWS=false            # set true to force cached events (no live scrape)
RELEVANCE_USE_LLM=true         # false = fast keyword matching instead of LLM
RECO_MAX_WORKERS=12            # concurrent per-investor recommendation calls
PORT=8000
```

> **Note:** `NVIDIA_API_KEY` is required for the reasoning steps. Without
> `AGENTQL_API_KEY` (or with `USE_MOCK_NEWS=true`) the app still runs end to end
> using cached real events, clearly flagged `cached-sample` in the UI/output.

## Run

**Web UI** (dropdown + two-step, live-streaming flow):
```bash
python run.py
# open http://127.0.0.1:8000
```

The UI has two buttons:

1. **Fetch news (AgentQL)** — calls `/api/fetch`, shows exactly what AgentQL
   pulled (headline, source, other candidate headlines) plus the impact sectors
   and how many investors are affected. The news call has **no artificial
   timeout** (heavy pages can take a while); an on-screen timer shows progress.
2. **Analyze & recommend** — opens an SSE stream to `/api/stream`. Per-investor
   recommendations are generated **concurrently** (`ThreadPoolExecutor`,
   `RECO_MAX_WORKERS`) and each RM alert card **appears the instant it's ready**,
   rather than waiting for the whole batch.

**CLI on 2 real events** (the deliverable):
```bash
python main.py                                   # Energy & Oil + Banking & Financials
python main.py --fields "Information Technology" "Energy & Oil"
python main.py --mock                            # cached news, no AgentQL key needed
```

Without an AgentQL key, news comes from **cached real events** (clearly flagged
`cached-sample` in the output/UI); the LLM reasoning and guardrail run for real.

---

Developed by **Prateek Khandelwal**
