
from typing import List, Dict

from sentence_transformers import CrossEncoder

# Singleton: load once per process, not per query
_reranker_model = None


def _get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Lazily loads the cross-encoder re-ranker model.
    Uses a singleton to avoid reloading the model on every call.

    Model choice: ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking — ideal for this retrieval task.
    - ~85 MB download, fast CPU inference (~10ms per pair).
    - Drop-in replaceable: swap with 'cross-encoder/ms-marco-electra-base'
      for higher accuracy at slightly higher latency.
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading cross-encoder re-ranker: {model_name} ...")
        _reranker_model = CrossEncoder(model_name, max_length=512)
    return _reranker_model


def rerank(query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Re-scores a list of retrieved document chunks using a cross-encoder (CE).

    Why cross-encoder over bi-encoder cosine similarity?
    - Bi-encoder (used in retrieval): encodes query and doc independently → fast but approximate.
    - Cross-encoder: encodes (query, doc) together → much more accurate relevance score,
      at the cost of O(N) inference per query.

    The standard pipeline is:
      Bi-encoder recall (retrieve top 50) → Cross-encoder precision (rerank to top K).

    Args:
        query:  The original user research question.
        docs:   List of retrieved doc dicts: [{content, metadata}, ...]
        top_k:  Maximum number of documents to return after reranking.

    Returns:
        Reranked list, highest-scoring documents first, capped at top_k.
    """
    if not docs:
        return []

    model = _get_reranker()

    # CE expects (query, passage) pairs
    pairs = [(query, doc.get("content", "")) for doc in docs]
    scores = model.predict(pairs)  # Returns numpy array of floats

    # Zip, sort descending by score, unzip
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    reranked = [doc for doc, _ in scored[:top_k]]
    print(f"Re-ranked {len(docs)} docs → kept top {len(reranked)}.")
    return reranked
