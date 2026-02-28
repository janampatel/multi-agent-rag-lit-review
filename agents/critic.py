"""
agents/critic.py — Heuristic Critic for draft review validation.

No LLM required. Checks whether every [N] citation ID in the draft
actually maps to one of the source documents. Unsupported citations
are flagged as issues.

This is the most common failure mode in RAG synthesis — the aggregator
invents citation IDs that don't correspond to real sources.
"""

import re
from typing import List, Dict


class CriticAgent:
    """
    Reads the draft review and checks factual claims against source documents.
    Uses heuristic citation-ID validation — no extra LLM call.
    """

    def critique(self, draft: str, source_docs: List[Dict]) -> List[Dict]:
        """
        Checks every [N] citation in the draft against the available sources.

        Args:
            draft:       The synthesized review text.
            source_docs: The list of source documents used for synthesis.

        Returns:
            List of issue dicts: [{"claim": "...", "reason": "..."}]
            Empty list means the draft is clean.
        """
        if not draft or not source_docs:
            return []

        total_sources = len(source_docs)
        # Find all [N] style citations in the draft
        cited_ids = set(int(m) for m in re.findall(r"\[(\d+)\]", draft))
        valid_ids  = set(range(1, total_sources + 1))

        issues = []
        for cid in sorted(cited_ids):
            if cid not in valid_ids:
                issues.append({
                    "claim":  f"Citation [{cid}] referenced in draft",
                    "reason": f"No source [{cid}] exists (only {total_sources} sources available)"
                })

        if issues:
            print(f"[Critic] Found {len(issues)} unsupported citation(s): "
                  f"{[i['claim'] for i in issues]}")
        else:
            print(f"[Critic] Draft clean — all citations valid ({total_sources} sources).")

        return issues
