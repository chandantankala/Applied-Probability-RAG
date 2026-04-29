"""
rag_chain.py — Build the retrieval + Claude chain used by the Streamlit app.

Isolated from app.py so the RAG logic can be tested independently of the UI.
The chain returned by build_chain() is a callable that takes a question
and returns {"answer": str, "sources": [Document]}.
"""

import os
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "voyage-4-large"   # must match the model used in ingest.py
LLM_MODEL = "claude-sonnet-4-6"      # swap to claude-opus-4-7 for higher quality
TOP_K = 6                            # number of chunks retrieved per query


SYSTEM_PROMPT = """You are a research assistant for applied probability. \
Answer the user's question using ONLY the provided context excerpts from \
academic papers. Follow these rules strictly:

1. If the context does not contain enough information to answer, say so \
   explicitly. Do not speculate or draw on outside knowledge.
2. Cite the paper (by filename) and page number for every substantive claim, \
   like this: [paper.pdf, p. 3].
3. When stating theorems or definitions, preserve mathematical precision. \
   Do not paraphrase formal statements loosely.
4. If the question is ambiguous, ask a clarifying question instead of guessing.
"""

USER_PROMPT = """Context excerpts from the papers:

{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Format retrieved documents with source metadata for the prompt."""
    formatted = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        formatted.append(f"[Excerpt {i} | {source}, p. {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_chain():
    """Construct the retriever + LLM chain.

    Returns a callable: question (str) -> {"answer": str, "sources": [Document]}.
    Raises EnvironmentError if API keys are missing.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    if not os.getenv("VOYAGE_API_KEY"):
        raise EnvironmentError("VOYAGE_API_KEY not set.")

    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    answer_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    def run(question: str):
        docs = retriever.invoke(question)
        answer = answer_chain.invoke({"docs": docs, "question": question})
        return {"answer": answer, "sources": docs}

    return run
