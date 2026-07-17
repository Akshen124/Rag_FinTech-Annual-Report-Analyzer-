import os
import tempfile
import streamlit as st
from src.ingest import load_and_chunk
from src.embeddings import Embedder
from src.indexer import add_documents
from src.qa import answer_query


st.set_page_config(page_title="Annual Report Analyzer")


@st.cache_data
def process_file(saved_path: str, collection_name: str = "reports"):
    chunks = load_and_chunk(saved_path)
    texts = [c["text"] for c in chunks]
    emb = Embedder()
    embeddings = emb.embed_texts(texts)
    add_documents(collection_name, docs=chunks, embeddings=embeddings)
    return len(chunks)


st.title("FinTech Annual Report Analyzer (RAG)")

uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded:
    if st.button("Ingest files"):
        with st.spinner("Processing..."):
            for f in uploaded:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp.flush()
                    count = process_file(tmp.name)
                st.success(f"Ingested {f.name}: {count} chunks")

st.markdown("---")
q = st.text_input("Ask a question about the ingested reports")
if q:
    with st.spinner("Retrieving answer..."):
        ans = answer_query("reports", q)
        st.markdown(ans)
