"""
Product-universe guardrail — the HARD CONSTRAINT enforced in code.

The assignment forbids recommending any instrument outside product_catalogue.csv.
We never trust the LLM alone for this: after the model proposes recommendations,
every fund_id is checked against the catalogue here. Anything outside the
universe is dropped and recorded so the failure is visible, not silent.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .data import load_catalogue, valid_fund_ids


def validate_recommendation(rec: dict) -> Tuple[dict, List[str]]:
    """
    Validate one investor recommendation dict in place.

    Expects rec["actions"] to be a list of dicts each carrying a "fund_id".
    Returns (clean_rec, violations) where violations lists any fund_ids that
    were removed because they are not in the product_universe.
    """
    valid = valid_fund_ids()
    catalogue = load_catalogue()

    clean_actions = []
    violations: List[str] = []

    for action in rec.get("actions", []):
        fid = (action.get("fund_id") or "").strip()
        if fid in valid:
            # Backfill the canonical fund name from the catalogue so the UI never
            # shows a hallucinated name even if the model got the name wrong.
            action["fund_name"] = catalogue[fid].fund_name
            action["_valid"] = True
            clean_actions.append(action)
        else:
            violations.append(fid or "<empty>")

    rec["actions"] = clean_actions
    if violations:
        rec["_violations"] = violations
    return rec, violations


def validate_all(recommendations: List[dict]) -> Tuple[List[dict], Dict[str, List[str]]]:
    """
    Validate a list of recommendations.

    Returns (clean_recommendations, violations_by_investor).
    Recommendations whose every action was invalid are still returned (with an
    empty actions list) so the RM at least sees who is affected.
    """
    all_violations: Dict[str, List[str]] = {}
    for rec in recommendations:
        _, violations = validate_recommendation(rec)
        if violations:
            all_violations[rec.get("investor_id", "?")] = violations
    return recommendations, all_violations
