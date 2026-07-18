"""
Launcher for the Market Signal -> Portfolio Recommendation Agent web app.

Usage:
    python run.py            # starts the web UI on $PORT (default 8000)
"""

import uvicorn

from src import config

if __name__ == "__main__":
    print(f"Starting Pravar.AI agent UI on http://127.0.0.1:{config.PORT}")
    uvicorn.run("src.app:app", host="127.0.0.1", port=config.PORT, reload=False)
