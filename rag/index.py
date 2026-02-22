
import json
import os
import math
import hashlib
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        self.db_path = os.path.join(persist_directory, "db.json")
        self.documents = []
        self._load_db()

    def _ensure_dir(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

    def _save_db(self):
        self._ensure_dir()
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)

    def _content_hash(self, text: str) -> str:
        """Generates an MD5 hash of the text to use as a unique document identifier."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Adds documents to the store with deduplication via content hashing.
        Re-running ingestion on the same PDF will not create duplicate entries.
        """
        # Build a set of hashes already in the store for O(1) lookup
        existing_hashes = {doc.get("content_hash") for doc in self.documents}

        new_count = 0
        skipped_count = 0

        for doc, emb in zip(documents, embeddings):
            content_hash = self._content_hash(doc["page_content"])

            if content_hash in existing_hashes:
                skipped_count += 1
                continue  # Duplicate — skip silently

            entry = {
                "page_content": doc["page_content"],
                "metadata": doc["metadata"],
                "embedding": emb,
                "content_hash": content_hash
            }
            self.documents.append(entry)
            existing_hashes.add(content_hash)
            new_count += 1

        self._save_db()
        print(f"Ingestion complete: {new_count} new chunks added, {skipped_count} duplicates skipped.")

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries the vector store using cosine similarity.
        Returns a dict structure compatible with ChromaDB's output format.
        """
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = []
        for doc in self.documents:
            doc_emb = doc["embedding"]
            score = self._cosine_similarity(query_embedding, doc_emb)
            results.append({"doc": doc, "score": score})

        # Sort descending by similarity score
        results.sort(key=lambda x: x["score"], reverse=True)
        top_k = results[:n_results]

        return {
            "ids": [[str(i) for i in range(len(top_k))]],
            "documents": [[r["doc"]["page_content"] for r in top_k]],
            "metadatas": [[r["doc"]["metadata"] for r in top_k]],
            "distances": [[round(1 - r["score"], 4) for r in top_k]]
        }

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude_v1 = math.sqrt(sum(a * a for a in v1))
        magnitude_v2 = math.sqrt(sum(a * a for a in v2))
        if magnitude_v1 * magnitude_v2 == 0:
            return 0.0
        return dot_product / (magnitude_v1 * magnitude_v2)
