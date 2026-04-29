"""
app.py — Streamlit UI for the applied-probability RAG tool.

Run with:
    streamlit run app.py

The app loads the vector store built by ingest.py and exposes an
interactive question/answer interface. Retrieved excerpts are shown
below the answer for transparency and verification.
"""

import streamlit as st
from rag_chain import build_chain

st.set_page_config(page_title="Applied Probability RAG", layout="wide")

st.title("Applied Probability — Paper Q&A")
st.caption(
    "Ask a question about the indexed papers. Answers are grounded in "
    "retrieved excerpts and cite their source."
)


@st.cache_resource(show_spinner="Loading retrieval index and model...")
def get_chain():
    """Build the chain once and cache it across Streamlit reruns."""
    return build_chain()


chain = get_chain()

question = st.text_area(
    "Your question:",
    placeholder=(
        "e.g., Under what conditions does the Markov chain in Smith (2019) "
        "admit a stationary distribution?"
    ),
    height=100,
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and reasoning..."):
        result = chain(question.strip())

    st.subheader("Answer")
    st.markdown(result["answer"])

    with st.expander(f"Retrieved excerpts ({len(result['sources'])})"):
        for i, doc in enumerate(result["sources"], 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{i}. {source}, p. {page}**")
            snippet = doc.page_content[:800]
            if len(doc.page_content) > 800:
                snippet += "..."
            st.text(snippet)
            st.divider()
