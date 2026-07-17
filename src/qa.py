import os
from .embeddings import Embedder
from .indexer import query as chroma_query
import requests


def get_top_docs(collection_name: str, texts: list, k: int = 4):
    # placeholder if needed
    return []


def answer_query(collection_name: str, query_text: str, persist_directory: str = "chroma_db") -> str:
    emb = Embedder()
    q_emb = emb.embed_texts([query_text])[0]

    results = chroma_query(collection_name, q_emb, n_results=6, persist_directory=persist_directory)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n---\n".join([f"Source: {m.get('source')}\n{d}" for d, m in zip(docs, metadatas)])

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq_url = os.environ.get("GROQ_CHAT_URL", "https://api.groq.ai/v1/chat/completions")
        payload = {
            "model": os.environ.get("GROQ_CHAT_MODEL", "gpt-3o-mini"),
            "messages": [{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query_text}\n\nAnswer concisely and cite sources by filename."}],
            "max_tokens": 512,
        }
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        r = requests.post(groq_url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        # accept multiple response shapes
        choice = None
        if "choices" in data and len(data["choices"]) > 0:
            c = data["choices"][0]
            if "message" in c:
                choice = c["message"].get("content")
            elif "text" in c:
                choice = c.get("text")
        if choice:
            return choice.strip()
        else:
            return "[Groq returned no text]"
    else:
        # fallback: return context excerpts + simple answer placeholder
        excerpt = "\n\n".join(docs[:4])
        return f"[Extractive answer — sources included]\n\nContext excerpts:\n{excerpt}\n\nQuestion: {query_text}\n\n(Set GROQ_API_KEY to enable generative Groq answers.)"
