
from .index import VectorStore
from .embed import embed_texts
import os

class Retriever:
    def __init__(self, persist_directory: str = None):
        # Auto-detect correct path based on current working directory
        if persist_directory is None:
            if os.path.exists("data/faiss_index"):
                persist_directory = "data/faiss_index"
            elif os.path.exists("../data/faiss_index"):
                persist_directory = "../data/faiss_index"
            else:
                persist_directory = "data/faiss_index"
        self.vector_store = VectorStore(persist_directory=persist_directory)

    def retrieve(self, query: str, k: int = 3):
        """
        End-to-end retrieval: Query -> Embed -> Search -> Return Results
        """
        print(f"Retrieving for: '{query}'")
        # 1. Embed Query
        query_embedding = embed_texts([query])[0]
        
        # 2. Search Vector DB
        results = self.vector_store.query(query_embedding, n_results=k)
        
        # 3. Format Output
        retrieved_docs = []
        # Handle Chroma's list-of-lists format
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                retrieved_docs.append({
                    "content": doc_text,
                    "metadata": meta
                })
        
        return retrieved_docs
