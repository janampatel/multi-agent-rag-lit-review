"""
agents/supervisor.py — Heuristic Supervisor for quality gating.

No LLM call — purely checks document count against a threshold.
This keeps the supervisor gate at near-zero latency.
"""

from typing import List, Dict


class SupervisorAgent:
    """
    Acts as a quality gate between retrieval and synthesis.
    Uses heuristics only — no LLM — to keep latency minimal.
    """

    def evaluate_quality(self, docs: List[Dict], threshold: int = 5) -> bool:
        """
        Returns True if we have enough documents to proceed to synthesis.

        Args:
            docs:      List of retrieved/screened document dicts.
            threshold: Minimum number of docs required (default 5).

        Returns:
            True  → proceed to synthesis
            False → retry retrieval with broadened queries
        """
        count = len(docs)
        if count >= threshold:
            print(f"[Supervisor] Quality gate PASSED — {count} docs (threshold={threshold})")
            return True
        print(f"[Supervisor] Quality gate FAILED — only {count} docs (need {threshold})")
        return False
