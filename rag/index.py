
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="research_papers")

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Adds documents to ChromaDB.
        documents: List of dicts with 'page_content' and 'metadata'
        embeddings: List of embedding vectors
        """
        ids = [str(i) for i in range(len(documents))]
        metadatas = [doc["metadata"] for doc in documents]
        documents_text = [doc["page_content"] for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries the vector store.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
