
from sentence_transformers import SentenceTransformer
from typing import List

# Load model once using a singleton text pattern effectively
_model = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        print(f"Loading embedding model: {model_name}...")
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of texts using local SentenceTransformer.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()
