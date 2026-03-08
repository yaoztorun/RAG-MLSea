import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_query(query: str, model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0]