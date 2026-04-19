"""
RAG retrieval module.
Loads the FAISS vector store (builds it on first call if missing) and
returns the top-3 most relevant market context sentences for a given query.
"""

from pathlib import Path

RAG_STORE_PATH = str(Path(__file__).parent.parent / "rag_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vectorstore = None


def _get_store():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(RAG_STORE_PATH).exists():
        _vectorstore = FAISS.load_local(
            RAG_STORE_PATH, embedder, allow_dangerous_deserialization=True
        )
    else:
        from rag.setup import build_vector_store
        _vectorstore = build_vector_store(silent=True)

    return _vectorstore


def get_market_context(city: str, property_type: str, bedrooms: int) -> str:
    """
    Return top-3 FAISS similarity results for the given property query.
    Injected directly into the LLM advisory prompt.
    """
    query = f"{city} {property_type} {bedrooms}BHK market price trend investment"
    try:
        store = _get_store()
        docs = store.similarity_search(query, k=3)
        return "\n".join(f"• {doc.page_content}" for doc in docs)
    except Exception as exc:
        return f"Market context unavailable ({exc})."
