import os
from typing import List, Dict
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise ImportError(
        "chromadb is required for the vector store.\n"
        "Install it into your active environment with: pip install chromadb\n"
        "Then run Streamlit using the same Python: python -m streamlit run app.py"
    ) from e


def get_client(persist_directory: str = "chroma_db"):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client


def add_documents(collection_name: str, docs: List[Dict], embeddings: List[List[float]], persist_directory: str = "chroma_db"):
    client = get_client(persist_directory=persist_directory)
    if collection_name in [c.name for c in client.list_collections()]:
        col = client.get_collection(collection_name)
    else:
        col = client.create_collection(name=collection_name)

    ids = [d["id"] for d in docs]
    metadatas = [{"source": d.get("source"), "chunk_index": d.get("chunk_index")} for d in docs]
    documents = [d["text"] for d in docs]
    col.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    client.persist()


def query(collection_name: str, query_embedding, n_results: int = 4, persist_directory: str = "chroma_db"):
    client = get_client(persist_directory=persist_directory)
    col = client.get_collection(collection_name)
    results = col.query(query_embeddings=[query_embedding], n_results=n_results)
    # returns dict with ids, documents, metadatas, distances
    return results
