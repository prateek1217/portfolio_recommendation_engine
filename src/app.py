"""
FastAPI backend for the Market Signal -> Portfolio Recommendation Agent.

Two-step, streaming API (matches the frontend/ two-button flow):
    GET  /                 -> the SPA (frontend/index.html)
    GET  /static/*         -> frontend assets (css/js)
    GET  /api/fields       -> selectable market fields for the dropdown
    POST /api/fetch        -> STEP 1: pull the news event + impact + affected
                              investors (no recommendations yet)
    POST /api/stream       -> STEP 2: Server-Sent Events; emits one
                              recommendation per investor as it completes

Run:  python run.py   (or  python -m uvicorn src.app:app --port 8000)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .graph import Nodes, fetch_event_and_impact, stream_recommendations

app = FastAPI(title="Pravar.AI — Market Signal → Portfolio Recommendation Agent")

FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Reuse one Nodes bundle (and thus one LLM client) across requests.
_NODES = Nodes()


class FetchRequest(BaseModel):
    field: str
    news_url: Optional[str] = None
    use_mock: Optional[bool] = None


class StreamRequest(BaseModel):
    event: dict
    impact: dict
    affected_investors: List[dict]


@app.get("/api/fields")
def get_fields():
    return {
        "fields": [
            {"name": name, "sectors": meta["sectors"]}
            for name, meta in config.MARKET_FIELDS.items()
        ],
        "mock_default": config.USE_MOCK_NEWS or not config.AGENTQL_API_KEY,
    }


@app.post("/api/fetch")
def api_fetch(req: FetchRequest):
    """STEP 1 — news + impact + affected investors (fast to render)."""
    result = fetch_event_and_impact(
        field_name=req.field,
        news_url=req.news_url,
        use_mock=req.use_mock,
        nodes=_NODES,
    )
    # A few other headlines seen on the page, for transparency in the UI.
    return {
        "field": req.field,
        "event": result["event"],
        "impact": result["impact"],
        "affected_investors": result["affected_investors"],
        "candidates": result.get("candidates", []),
        "trace": result.get("trace", []),
    }


@app.post("/api/stream")
def api_stream(req: StreamRequest):
    """STEP 2 — stream each recommendation as Server-Sent Events."""

    def gen():
        trace = [
            f"Streaming recommendations for {len(req.affected_investors)} investors "
            f"({min(config.RECO_MAX_WORKERS, max(len(req.affected_investors),1))} in parallel)."
        ]
        for rec in stream_recommendations(
            req.event, req.impact, req.affected_investors, nodes=_NODES
        ):
            yield _sse({"type": "rec", "rec": rec})
        trace.append("Guardrail applied per card; all funds constrained to universe.")
        yield _sse({"type": "done", "trace": trace})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
