
from .index import VectorStore
from .embed import embed_texts

class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()

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
