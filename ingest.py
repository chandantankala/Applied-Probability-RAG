"""
ingest.py — Build a Chroma vector store from PDFs in ./papers/

Run once, or whenever you add/remove papers. The resulting index
is written to ./chroma_db/ and read by app.py at query time.

Usage:
    python ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PAPERS_DIR = Path("./papers")
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "voyage-4-large"

# Math-aware chunking: prefer to split at these boundaries before hitting
# the character limit. Order matters — earlier separators are tried first.
MATH_SEPARATORS = [
    "\n\nTheorem",
    "\n\nLemma",
    "\n\nProposition",
    "\n\nCorollary",
    "\n\nDefinition",
    "\n\nProof",
    "\n\nRemark",
    "\n\nExample",
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # sentence
    " ",
    "",
]


def load_all_pdfs(papers_dir: Path):
    """Load every PDF in papers_dir. Returns a list of page-level Document objects."""
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {papers_dir.resolve()}. "
            f"Put your .pdf files there and re-run."
        )
    docs = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        pages = PyPDFLoader(str(pdf_path)).load()
        # Carry the filename in metadata so we can cite sources later.
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        docs.extend(pages)
    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return docs


def main():
    if not os.getenv("VOYAGE_API_KEY"):
        raise EnvironmentError("VOYAGE_API_KEY not set. Check your .env file.")

    # 1. Load
    documents = load_all_pdfs(PAPERS_DIR)

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=MATH_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed and store
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store written to {CHROMA_DIR}.")
    print(f"Total vectors indexed: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()
