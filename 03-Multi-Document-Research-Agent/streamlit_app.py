"""
streamlit_app.py
-----------------
Streamlit chat UI for the Multi-Document Research Agent.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.config import config
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="Multi-Document Research Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------- #
# Dark theme styling
# ---------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #131722; }
    .source-card {
        background-color: #1a1f2b; border-radius: 8px; padding: 10px 14px;
        margin-bottom: 8px; border-left: 3px solid #4f8bf9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models & vector store…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(lazy_gemini=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ..., "sources": ..., "elapsed": ...}

pipeline = get_pipeline()

# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.title(" Research Agent")
    st.caption(f"Model: `{config.gemini_model}`  ·  Vectors indexed: `{len(pipeline.vector_store)}`")

    st.subheader(" Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT — multiple files supported",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index uploaded files", use_container_width=True):
        documents_dir = Path(config.documents_dir)
        documents_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for uf in uploaded_files:
            dest = documents_dir / uf.name
            dest.write_bytes(uf.getbuffer())
            saved_paths.append(str(dest))
        with st.spinner("Chunking, embedding, and indexing…"):
            stats = pipeline.ingest_paths(saved_paths)
        st.success(f"Indexed {stats['num_chunks']} chunks from {stats['num_documents']} document(s).")

    st.divider()
    st.subheader(" Retrieval settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=15, value=config.top_k)
    hybrid_alpha = st.slider("Vector ↔ BM25 balance", min_value=0.0, max_value=1.0,
                              value=config.hybrid_alpha, step=0.05,
                              help="1.0 = pure vector search, 0.0 = pure keyword (BM25) search")
    pipeline.retriever.alpha = hybrid_alpha

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(" Reset chat", use_container_width=True):
            st.session_state.chat_history = []
            pipeline.reset_memory()
            st.rerun()
    with col_b:
        if st.button(" Clear DB", use_container_width=True):
            pipeline.clear_index()
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------- #
# Main chat window
# ---------------------------------------------------------------------- #
st.header("Multi-Document Research Agent")
st.caption("Ask questions across every document you've indexed. Answers are grounded with inline citations.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f" Sources & retrieved chunks · {turn.get('elapsed', 0):.2f}s"):
                for src in turn["sources"]:
                    st.markdown(
                        f"<div class='source-card'><b>{src['citation']} {src['filename']}</b> "
                        f"— chunk #{src['chunk_index']} · score {src['score']}</div>",
                        unsafe_allow_html=True,
                    )

user_question = st.chat_input("Ask a question about your documents…")
if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if len(pipeline.vector_store) == 0:
            answer = "No documents are indexed yet — upload some files from the sidebar first."
            sources, elapsed = [], 0.0
            st.markdown(answer)
        else:
            placeholder = st.empty()
            start = time.perf_counter()
            with st.spinner("Retrieving & generating…"):
                result = pipeline.query(user_question, top_k=top_k)
            elapsed = time.perf_counter() - start
            answer = result["answer"]
            sources = result["sources"]
            placeholder.markdown(answer)
            with st.expander(f" Sources & retrieved chunks · {elapsed:.2f}s"):
                for src in sources:
                    st.markdown(
                        f"<div class='source-card'><b>{src['citation']} {src['filename']}</b> "
                        f"— chunk #{src['chunk_index']} · score {src['score']}</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": sources, "elapsed": elapsed,
    })
