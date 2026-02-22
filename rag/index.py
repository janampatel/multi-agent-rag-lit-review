
import hashlib
import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Production vector store backed by ChromaDB with persistent HNSW indexing.

    Key improvements over the previous JSON-based store:
    - O(log N) approximate nearest-neighbour search (HNSW) instead of O(N) brute force.
    - Native persistence — no manual JSON serialisation required.
    - Built-in deduplication via upsert: re-ingesting the same PDF is a no-op.
    - Cosine similarity configured at the collection level.
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        print(f"ChromaDB ready — collection '{self.COLLECTION_NAME}' "
              f"has {self.collection.count()} documents.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _content_hash(self, text: str) -> str:
        """Stable MD5 hash of content — used as the ChromaDB document ID."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Upserts documents into ChromaDB.
        Using upsert (not add) means re-running ingestion on the same source
        is completely safe — existing documents are updated in place, not duplicated.
        """
        if not documents:
            print("No documents to add.")
            return

        ids, texts, metas, embs = [], [], [], []
        for doc, emb in zip(documents, embeddings):
            content_hash = self._content_hash(doc["page_content"])
            # Ensure all metadata values are ChromaDB-compatible (str / int / float / bool)
            safe_meta = {k: str(v) for k, v in doc.get("metadata", {}).items()}

            ids.append(content_hash)
            texts.append(doc["page_content"])
            metas.append(safe_meta)
            embs.append(emb)

        before = self.collection.count()
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs,
        )
        after = self.collection.count()
        new_count = after - before
        skipped = len(documents) - new_count
        print(f"Ingestion complete: {new_count} new chunks added, "
              f"{skipped} duplicates skipped. "
              f"Total in store: {after}.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries ChromaDB with a query embedding.
        Returns results in the same dict structure as before so the Retriever
        layer requires zero changes.

        ChromaDB with cosine space returns distances in [0, 2] where 0 = identical.
        We keep that convention so callers can interpret distances naturally.
        """
        total = self.collection.count()
        if total == 0:
            print("Warning: VectorStore is empty. Please ingest documents first.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Can't ask for more results than we have
        k = min(n_results, total)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return results
