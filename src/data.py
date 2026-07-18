"""
Data layer — loads the two PRE-INGESTED datasets and derives the lookups the
agent needs. Per the assignment, portfolios + product_universe are treated as
already ingested; this module is the single source of truth for both.

Nothing here calls an LLM or the network. It is pure, deterministic data prep.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

from . import config


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
@dataclass
class Fund:
    """One row of product_catalogue.csv — a recommendable instrument."""

    fund_id: str
    fund_name: str
    amc: str
    category: str
    sub_category: str
    risk_grade: str
    expense_ratio_pct: str
    benchmark: str
    primary_sectors: List[str]

    def sector_str(self) -> str:
        return ", ".join(self.primary_sectors)


@dataclass
class Holding:
    """One fund position inside an investor's portfolio."""

    fund_id: str
    fund_name: str
    current_value_inr: float
    units: float
    weight_pct: float


@dataclass
class Investor:
    """One investor and all of their holdings."""

    investor_id: str
    name: str
    risk_profile: str
    age: int
    life_stage: str
    portfolio_value_inr: float
    holdings: List[Holding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders (cached — CSVs are static / pre-ingested)
# ---------------------------------------------------------------------------
def _split_sectors(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


@lru_cache(maxsize=1)
def load_catalogue() -> Dict[str, Fund]:
    """Return {fund_id: Fund} for the entire product_universe."""
    funds: Dict[str, Fund] = {}
    with open(config.CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            funds[row["fund_id"]] = Fund(
                fund_id=row["fund_id"],
                fund_name=row["fund_name"],
                amc=row["amc"],
                category=row["category"],
                sub_category=row["sub_category"],
                risk_grade=row["risk_grade"],
                expense_ratio_pct=row["expense_ratio_pct"],
                benchmark=row["benchmark"],
                primary_sectors=_split_sectors(row["primary_sectors"]),
            )
    return funds


@lru_cache(maxsize=1)
def load_investors() -> Dict[str, Investor]:
    """Return {investor_id: Investor} with holdings attached."""
    investors: Dict[str, Investor] = {}
    with open(config.PORTFOLIOS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inv_id = row["investor_id"]
            if inv_id not in investors:
                investors[inv_id] = Investor(
                    investor_id=inv_id,
                    name=row["name"],
                    risk_profile=row["risk_profile"],
                    age=int(row["age"]),
                    life_stage=row["life_stage"],
                    portfolio_value_inr=float(row["portfolio_value_inr"]),
                )
            investors[inv_id].holdings.append(
                Holding(
                    fund_id=row["fund_id"],
                    fund_name=row["fund_name"],
                    current_value_inr=float(row["current_value_inr"]),
                    units=float(row["units"]),
                    weight_pct=float(row["weight_pct"]),
                )
            )
    return investors


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def fund_sectors(fund_id: str) -> List[str]:
    fund = load_catalogue().get(fund_id)
    return fund.primary_sectors if fund else []


def investor_sector_exposure(inv: Investor) -> Dict[str, float]:
    """
    Aggregate an investor's weight_pct across the sectors their funds touch.

    A fund's weight is attributed in full to each of its primary_sectors (a fund
    tagged "Energy, IT" contributes its whole weight to both). This intentionally
    over-counts overlapping sectors — it is a materiality signal for "is this
    investor exposed to sector X", not a precise decomposition.
    """
    exposure: Dict[str, float] = {}
    for h in inv.holdings:
        for sector in fund_sectors(h.fund_id):
            exposure[sector] = exposure.get(sector, 0.0) + h.weight_pct
    return dict(sorted(exposure.items(), key=lambda kv: -kv[1]))


def match_investors_to_sectors(
    affected_sectors: List[str],
) -> List[dict]:
    """
    Deterministically find investors exposed to any of `affected_sectors`.

    Matching is case-insensitive substring in both directions so that an event
    sector like "Banking" matches catalogue sectors "Private Banks" /
    "Financial Services", and vice versa.

    Returns a list of dicts (JSON-ready) sorted by total exposure desc:
        {investor_id, name, risk_profile, age, life_stage, portfolio_value_inr,
         total_exposure_pct, matched_sectors, impacted_holdings:[...]}
    """
    affected_norm = [s.lower().strip() for s in affected_sectors if s.strip()]
    results = []

    for inv in load_investors().values():
        matched_sectors = set()
        impacted_holdings = []

        for h in inv.holdings:
            hsectors = fund_sectors(h.fund_id)
            hit = []
            for hs in hsectors:
                hs_l = hs.lower()
                for a in affected_norm:
                    if a and (a in hs_l or hs_l in a):
                        hit.append(hs)
                        matched_sectors.add(hs)
                        break
            if hit:
                impacted_holdings.append(
                    {
                        "fund_id": h.fund_id,
                        "fund_name": h.fund_name,
                        "weight_pct": h.weight_pct,
                        "current_value_inr": h.current_value_inr,
                        "matched_sectors": hit,
                    }
                )

        if impacted_holdings:
            total_exposure = round(
                sum(h["weight_pct"] for h in impacted_holdings), 2
            )
            results.append(
                {
                    "investor_id": inv.investor_id,
                    "name": inv.name,
                    "risk_profile": inv.risk_profile,
                    "age": inv.age,
                    "life_stage": inv.life_stage,
                    "portfolio_value_inr": inv.portfolio_value_inr,
                    "total_exposure_pct": total_exposure,
                    "matched_sectors": sorted(matched_sectors),
                    "impacted_holdings": sorted(
                        impacted_holdings, key=lambda x: -x["weight_pct"]
                    ),
                }
            )

    return sorted(results, key=lambda r: -r["total_exposure_pct"])


def catalogue_for_prompt() -> str:
    """
    Compact text rendering of the FULL product_universe for the LLM prompt.
    This is the ONLY set of instruments the agent may recommend from.
    """
    lines = []
    for f in load_catalogue().values():
        lines.append(
            f"{f.fund_id} | {f.fund_name} | {f.category}/{f.sub_category} "
            f"| risk={f.risk_grade} | sectors={f.sector_str()}"
        )
    return "\n".join(lines)


def valid_fund_ids() -> set:
    return set(load_catalogue().keys())


if __name__ == "__main__":
    inv = load_investors()
    cat = load_catalogue()
    print(f"Loaded {len(inv)} investors, {len(cat)} funds.")
    sample = next(iter(inv.values()))
    print(f"\nSample investor: {sample.name} ({sample.risk_profile})")
    print("Exposure:", investor_sector_exposure(sample))
    print("\nMatch test (Energy):")
    for r in match_investors_to_sectors(["Energy"])[:3]:
        print(f"  {r['name']}: {r['total_exposure_pct']}% via {r['matched_sectors']}")
