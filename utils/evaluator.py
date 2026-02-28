"""
utils/evaluator.py — Lightweight RAG pipeline evaluation metrics.

No external dependencies (no RAGAS, no OpenAI).
Computes 4 metrics from the final AgentState.
"""

import re
from typing import List, Dict


class RAGEvaluator:
    """
    Evaluates the pipeline output using lightweight heuristic metrics.

      1. context_precision  = screened_docs / total_retrieved
         (What fraction of retrieved docs were kept as relevant?)

      2. context_recall     = screened_docs / (screened_docs + arxiv_docs)
         (How much of the final screening came from live ArXiv vs local store?)

      3. faithfulness       = fraction of [N] citations in the draft that map to
                               valid source IDs (same check as CriticAgent).

      4. Pipeline counts:   total_retrieved, total_screened, revision_rounds.
    """

    def _compute_faithfulness(self, review: str, source_docs: List[Dict]) -> float:
        """Fraction of [N] citation IDs that are valid source indices."""
        if not review or not source_docs:
            return 1.0
        cited = set(int(m) for m in re.findall(r"\[(\d+)\]", review))
        if not cited:
            return 1.0
        valid = set(range(1, len(source_docs) + 1))
        supported = cited & valid
        return round(len(supported) / len(cited), 3)

    def generate_report(self, state: Dict) -> Dict:
        """
        Generates an evaluation report from the final agent state.

        Args:
            state: The final AgentState dict after the workflow completes.

        Returns:
            Dict with metric keys and values.
        """
        total_retrieved = len(state.get("merged_docs",
                              state.get("retrieved_docs", [])))
        screened_docs   = state.get("screened_docs",
                          state.get("retrieved_docs", []))
        total_screened  = len(screened_docs)
        web_docs        = state.get("web_search_docs", [])
        total_arxiv     = len(web_docs)
        review          = state.get("draft_review",
                          state.get("final_response", ""))
        revision_rounds = state.get("revision_count", 0)

        context_precision = round(
            total_screened / max(total_retrieved, 1), 3
        )
        # context_recall: how much of the final pool came from local vs ArXiv
        total_pool = total_screened + total_arxiv
        context_recall = round(
            total_screened / max(total_pool, 1), 3
        )

        faithfulness = self._compute_faithfulness(review, screened_docs)

        report = {
            "context_precision":  context_precision,
            "context_recall":     context_recall,
            "faithfulness":       faithfulness,
            "total_retrieved":    total_retrieved,
            "total_screened":     total_screened,
            "total_arxiv_added":  total_arxiv,
            "revision_rounds":    revision_rounds,
        }
        return report

    def print_report(self, report: Dict):
        """Pretty-prints the evaluation report to stdout."""
        print("\n" + "=" * 45)
        print("       RAG PIPELINE EVALUATION REPORT")
        print("=" * 45)
        print(f"  Context Precision : {report['context_precision']:.1%}")
        print(f"  Context Recall    : {report['context_recall']:.1%}")
        print(f"  Faithfulness      : {report['faithfulness']:.1%}")
        print(f"  Total Retrieved   : {report['total_retrieved']}")
        print(f"  Total Screened    : {report['total_screened']}")
        print(f"  ArXiv Docs Added  : {report['total_arxiv_added']}")
        print(f"  Revision Rounds   : {report['revision_rounds']}")
        print("=" * 45 + "\n")
