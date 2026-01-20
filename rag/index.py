
import json
import os
import math
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        # Simulating a persistent store with a JSON file
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

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Adds documents to the in-memory store and persists to JSON.
        """
        for doc, emb in zip(documents, embeddings):
            entry = {
                "page_content": doc["page_content"],
                "metadata": doc["metadata"],
                "embedding": emb
            }
            self.documents.append(entry)
        
        self._save_db()
        print(f"Stored {len(documents)} documents in temporary JSON store at {self.db_path}")

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries the vector store using cosine similarity (pure python/math implementation).
        Returns a dict structure similar to ChromaDB's output for compatibility.
        """
        results = []
        
        for doc in self.documents:
            doc_emb = doc["embedding"]
            score = self._cosine_similarity(query_embedding, doc_emb)
            results.append({
                "doc": doc,
                "score": score
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        top_k = results[:n_results]
        
        # Format like ChromaDB output: 
        # {'documents': [[...]], 'metadatas': [[...]], 'ids': [[...]], 'distances': [[...]]}
        # Note: Chroma uses 'distances' (lower is better) or 'similarities' depending on config.
        # We'll just return structures the downstream code expects.
        
        return {
            "ids": [[str(i) for i in range(len(top_k))]], # Dummy IDs
            "documents": [[r["doc"]["page_content"] for r in top_k]],
            "metadatas": [[r["doc"]["metadata"] for r in top_k]],
            "distances": [[1 - r["score"] for r in top_k]] # Fake distance
        }

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude_v1 = math.sqrt(sum(a * a for a in v1))
        magnitude_v2 = math.sqrt(sum(a * a for a in v2))
        if magnitude_v1 * magnitude_v2 == 0:
            return 0.0
        return dot_product / (magnitude_v1 * magnitude_v2)
