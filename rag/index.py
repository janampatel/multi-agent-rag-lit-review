
import hashlib
import json
import os
from typing import Dict, List

import faiss
import numpy as np


class VectorStore:
    """
    Production vector store backed by FAISS (IndexFlatIP on normalised vectors = cosine similarity)
    with a companion JSON file for document text and metadata.

    Why FAISS over ChromaDB here?
    - Zero pydantic / grpc dependencies → no compatibility issues across Python versions.
    - faiss.IndexFlatIP on L2-normalised vectors gives exact cosine similarity.
    - Swap to IndexIVFFlat or IndexHNSWFlat for approximate search at larger scale.

    Storage layout (inside persist_directory/):
        faiss.index   — FAISS binary index (vectors)
        metadata.json — list of {page_content, metadata, content_hash} records
                        (one entry per vector, same positional order as the index)
    """

    def __init__(self, persist_directory: str = "data/faiss_index"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self._index_path = os.path.join(persist_directory, "faiss.index")
        self._meta_path = os.path.join(persist_directory, "metadata.json")

        self._records: List[Dict] = []    # parallel to FAISS vectors
        self._index: faiss.Index | None = None
        self._dim: int | None = None

        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self):
        """Load the FAISS index and metadata records from disk if they exist."""
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._records = json.load(f)

        if os.path.exists(self._index_path) and self._records:
            self._index = faiss.read_index(self._index_path)
            self._dim = self._index.d
            print(f"VectorStore loaded — {self._index.ntotal} vectors, dim={self._dim}.")
        else:
            print("VectorStore: no existing index found. Starting fresh.")

    def _save(self):
        """Persist the FAISS index and metadata records to disk."""
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)

    def _init_index(self, dim: int):
        """Lazily initialise the FAISS index on first document insertion."""
        self._dim = dim
        # IndexFlatIP + L2-normalised vectors ≡ cosine similarity, exact (brute-force).
        # For >10k docs replace with: faiss.IndexHNSWFlat(dim, 32)
        self._index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Content hashing
    # ------------------------------------------------------------------

    def _content_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Adds documents to FAISS + metadata store with deduplication via content hashing.
        Re-running ingestion on the same source is a safe no-op.
        """
        if not documents:
            print("No documents to add.")
            return

        existing_hashes = {r["content_hash"] for r in self._records}

        new_texts, new_metas, new_hashes, new_vecs = [], [], [], []

        for doc, emb in zip(documents, embeddings):
            h = self._content_hash(doc["page_content"])
            if h in existing_hashes:
                continue  # duplicate — skip
            existing_hashes.add(h)
            new_texts.append(doc["page_content"])
            new_metas.append(doc.get("metadata", {}))
            new_hashes.append(h)
            new_vecs.append(emb)

        skipped = len(documents) - len(new_texts)

        if not new_vecs:
            print(f"Ingestion complete: 0 new chunks (all {skipped} were duplicates).")
            return

        # Build numpy matrix and L2-normalise for cosine similarity
        mat = np.array(new_vecs, dtype="float32")
        faiss.normalize_L2(mat)

        # Initialise index on first use
        if self._index is None:
            self._init_index(mat.shape[1])

        self._index.add(mat)

        # Append matching metadata records (same positional order as FAISS)
        for text, meta, h in zip(new_texts, new_metas, new_hashes):
            self._records.append({
                "page_content": text,
                "metadata": meta,
                "content_hash": h,
            })

        self._save()
        print(f"Ingestion complete: {len(new_vecs)} new chunks added, "
              f"{skipped} duplicates skipped. "
              f"Total in store: {self._index.ntotal}.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries FAISS for the top-k most similar vectors.
        Returns results in the ChromaDB-compatible dict structure so the
        Retriever layer requires zero changes.

        Scores are cosine similarities in [-1, 1]; distances = 1 - score.
        """
        if self._index is None or self._index.ntotal == 0:
            print("Warning: VectorStore is empty. Please ingest documents first.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        k = min(n_results, self._index.ntotal)

        # Normalise query vector (same space as stored vectors)
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)

        scores, indices = self._index.search(q, k)  # shape (1, k) each

        top_docs, top_metas, top_ids, top_dists = [], [], [], []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for padding
                continue
            rec = self._records[idx]
            top_docs.append(rec["page_content"])
            top_metas.append(rec["metadata"])
            top_ids.append(str(rank))
            top_dists.append(round(float(1.0 - score), 4))

        return {
            "ids": [top_ids],
            "documents": [top_docs],
            "metadatas": [top_metas],
            "distances": [top_dists],
        }
