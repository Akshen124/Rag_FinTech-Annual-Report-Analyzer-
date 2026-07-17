import os
from typing import List
import numpy as np

GROQ_KEY = os.environ.get("GROQ_API_KEY")
GROQ_EMBEDDINGS_URL = os.environ.get("GROQ_EMBEDDINGS_URL", "https://api.groq.ai/v1/embeddings")


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.groq_key = GROQ_KEY
        if not self.groq_key:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.groq_key:
            import requests
            payload = {
                "model": os.environ.get("GROQ_EMBEDDING_MODEL", "embed-small"),
                "input": texts,
            }
            headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
            r = requests.post(GROQ_EMBEDDINGS_URL, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            # expect data["data"] similar to OpenAI: list of {"embedding": [...]}
            embeddings = [item.get("embedding") for item in data.get("data", [])]
            return embeddings
        else:
            vectors = self.model.encode(texts, show_progress_bar=False)
            return vectors.tolist()


def normalize(v):
    a = np.array(v)
    n = np.linalg.norm(a)
    if n == 0:
        return a.tolist()
    return (a / n).tolist()
