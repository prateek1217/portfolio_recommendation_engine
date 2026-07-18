"""
CLI end-to-end runner — satisfies the deliverable:
"runs end-to-end on at least 2 real market events against the portfolio dataset."

Usage:
    python main.py                       # runs the 2 default events
    python main.py --fields "Energy & Oil" "Banking & Financials"
    python main.py --mock                # use cached sample events (no AgentQL key)

Requires an NVIDIA API key in .env for the reasoning steps. Pass --mock to avoid
needing an AgentQL key (news is served from cached real events).
"""

from __future__ import annotations

import argparse

from src.graph import Nodes, run_pipeline


def print_result(field: str, result: dict) -> None:
    ev = result.get("event", {})
    im = result.get("impact", {})
    recs = result.get("recommendations", [])

    print("\n" + "=" * 74)
    print(f"MARKET FIELD: {field}")
    print("=" * 74)
    print(f"EVENT  ({ev.get('source')}): {ev.get('headline')}")
    print(f"       {ev.get('summary')}")
    if ev.get("source_url"):
        print(f"       source: {ev.get('source_url')}")
    print(f"\nIMPACT: {im.get('one_line_summary','')}")
    for s in im.get("affected_sectors", []):
        print(f"   - {s.get('sector')}: {s.get('impact')} ({s.get('order')}) "
              f"— {s.get('rationale')}")

    order = {"high": 0, "medium": 1, "low": 2}
    recs = sorted(recs, key=lambda r: (order.get((r.get('severity') or 'low').lower(), 3),
                                       -r.get("total_exposure_pct", 0)))
    print(f"\nRM ALERTS — {len(recs)} affected investor(s):")
    for r in recs:
        print(f"\n  [{(r.get('severity') or '?').upper():6}] {r.get('name')} "
              f"({r.get('risk_profile')}, {r.get('total_exposure_pct')}% exposed "
              f"via {', '.join(r.get('matched_sectors', []))})")
        print(f"          {r.get('headline','')}")
        print(f"          {r.get('rationale','')}")
        for a in r.get("actions", []):
            print(f"            -> {(a.get('type') or '?').upper():6} "
                  f"{a.get('fund_id')} {a.get('fund_name','')}: {a.get('reason','')}")
        if not r.get("actions"):
            print("            (no in-universe action generated)")

    viol = result.get("violations", {})
    if viol:
        print(f"\n  GUARDRAIL removed out-of-universe funds: {viol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+",
                    default=["Energy & Oil", "Banking & Financials"],
                    help="market fields to run (>=2 for the deliverable)")
    ap.add_argument("--mock", action="store_true",
                    help="use cached sample news (no AgentQL key needed)")
    args = ap.parse_args()

    nodes = Nodes()  # share one LLM client across all events
    for field in args.fields:
        result = run_pipeline(field, use_mock=args.mock or None, nodes=nodes)
        print_result(field, result)


if __name__ == "__main__":
    main()
