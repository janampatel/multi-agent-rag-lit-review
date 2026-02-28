"""
agents/screening_council.py — 3-screener voting council for paper relevance.

Screeners:
  1. RecencyScreener    — heuristic: year >= min_year (no LLM)
  2. EmpiricalScreener  — heuristic: keyword scan for experiments (no LLM)
  3. MethodologyScreener— LLM-based: reuses cached ScreeningAgent logic

A paper passes if >= 2 of 3 screeners approve (majority vote).
Only 1 LLM screener means the council costs the same as the old single screener.
"""

import re
from typing import List, Dict
from agents.screening import ScreeningAgent


class RecencyScreener:
    """
    Heuristic: pass papers published in min_year or later.
    Falls back to True (pass) if year metadata is missing.
    """

    def __init__(self, min_year: int = 2019):
        self.min_year = min_year

    def approve(self, doc: Dict, query: str) -> bool:
        year_str = doc.get("metadata", {}).get("year", "")
        match = re.search(r"\b(19|20)\d{2}\b", str(year_str))
        if match:
            year = int(match.group(0))
            return year >= self.min_year
        return True  # Can't determine year → give benefit of the doubt


class EmpiricalScreener:
    """
    Heuristic: pass papers that contain evidence of experiments or evaluations.
    Looks for empirical keywords in the content.
    Falls back to True if content is short/missing.
    """

    KEYWORDS = {
        "experiment", "experiments", "evaluation", "evaluations",
        "baseline", "benchmark", "dataset", "accuracy", "performance",
        "results", "ablation", "metric", "f1", "precision", "recall",
        "table", "figure", "comparison"
    }

    def approve(self, doc: Dict, query: str) -> bool:
        content = doc.get("content", "").lower()
        if len(content) < 100:
            return True  # Too short to judge
        words = set(re.findall(r"\b\w+\b", content))
        hits = len(words & self.KEYWORDS)
        return hits >= 2  # At least 2 empirical keywords


class MethodologyScreener:
    """
    LLM-based screener: reuses the existing ScreeningAgent which is cached.
    Screens papers in a single batch call.
    """

    def __init__(self):
        self._agent = ScreeningAgent()

    def approve_batch(self, docs: List[Dict], query: str) -> List[bool]:
        """
        Returns a list of booleans (one per doc) for methodology relevance.
        Uses the existing cached ScreeningAgent.
        """
        if not docs:
            return []
        screened = self._agent.screen(docs, query)
        # Build a set of content hashes for O(1) lookup
        passed = {d.get("content", ""): True for d in screened}
        return [d.get("content", "") in passed for d in docs]


class ScreeningCouncil:
    """
    Runs all 3 screeners and applies majority voting (>= 2 of 3 must approve).
    """

    def __init__(self, min_year: int = 2019):
        self.recency    = RecencyScreener(min_year=min_year)
        self.empirical  = EmpiricalScreener()
        self.methodology = MethodologyScreener()

    def vote(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        Applies 3 screeners and keeps papers that get at least 2 approvals.

        Args:
            papers: List of {content, metadata} dicts.
            query:  Research topic string.

        Returns:
            Filtered list of papers that passed the council vote.
        """
        if not papers:
            return []

        print(f"[Council] Voting on {len(papers)} papers...")

        # Run heuristic screeners first (no latency)
        recency_votes   = [self.recency.approve(p, query)   for p in papers]
        empirical_votes = [self.empirical.approve(p, query) for p in papers]

        # Run LLM screener once for the full batch (cached)
        method_votes = self.methodology.approve_batch(papers, query)

        passed = []
        for i, paper in enumerate(papers):
            votes = [
                recency_votes[i],
                empirical_votes[i],
                method_votes[i] if i < len(method_votes) else True,
            ]
            approvals = sum(votes)
            if approvals >= 2:
                passed.append(paper)

        print(
            f"[Council] Passed: {len(passed)}/{len(papers)} "
            f"(recency={sum(recency_votes)}, "
            f"empirical={sum(empirical_votes)}, "
            f"methodology={sum(method_votes)})"
        )
        return passed
