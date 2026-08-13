
import os
import tempfile
from pathlib import Path

import streamlit as st

from core.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Enterprise PDF Chatbot")

st.markdown(
    """
Ask questions about your uploaded documents using
Retrieval-Augmented Generation (RAG) powered by
Gemini 3.5 Flash.
"""
)

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = Path("data/raw") / uploaded_file.name

    with open(save_path,"wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully.")

question = st.chat_input("Ask a question...")

if question:

    with st.spinner("Searching..."):

        response = st.session_state.rag.ask(question)

    st.session_state.history.append(
        (
            question,
            response
        )
    )

for question,response in st.session_state.history:

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        st.write(response["answer"])

        with st.expander("Sources"):

            for source in response["sources"]:

                st.markdown(
                    f"""
**File:** {source['filename']}

**Page:** {source['page']}

**Similarity:** {source['score']:.3f}

---

{source['text']}
"""
                )
