import os
from typing import List, Dict
try:
    import fitz
except Exception as e:
    raise ImportError("PyMuPDF (pymupdf) is required to extract PDF text.\nInstall it with: pip install pymupdf or pip install -r requirements.txt") from e

try:
    from docx import Document
except Exception as e:
    raise ImportError("python-docx is required to extract DOCX text.\nInstall it with: pip install python-docx or pip install -r requirements.txt") from e


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    tokens = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({
            "id": f"chunk-{idx}",
            "text": chunk_text,
        })
        idx += 1
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def load_and_chunk(path: str, chunk_size: int = 1000, overlap: int = 200):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        text = extract_text_from_docx(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    # add source metadata
    for i, c in enumerate(chunks):
        c["source"] = os.path.basename(path)
        c["chunk_index"] = i
    return chunks
