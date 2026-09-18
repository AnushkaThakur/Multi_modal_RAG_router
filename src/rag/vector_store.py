from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig


def _load_documents(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path.name)}))
    return docs


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step
    return chunks


def _split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunk_docs: list[Document] = []
    for doc in documents:
        for idx, chunk_text in enumerate(_split_text(doc.page_content, chunk_size, chunk_overlap)):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = idx
            chunk_docs.append(Document(page_content=chunk_text, metadata=metadata))
    return chunk_docs


def ingest_corpus(config: AppConfig) -> int:
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    documents = _load_documents(config.corpus_dir)
    if not documents:
        raise ValueError("No .md or .txt files found in data/corpus.")

    chunks = _split_documents(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    store = Chroma(
        collection_name="rag_router_docs",
        embedding_function=embeddings,
        persist_directory=str(config.vectorstore_dir),
    )
    store.reset_collection()
    store.add_documents(chunks)
    return len(chunks)


def build_retriever(config: AppConfig):
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    store = Chroma(
        collection_name="rag_router_docs",
        embedding_function=embeddings,
        persist_directory=str(config.vectorstore_dir),
    )
    return store.as_retriever(search_kwargs={"k": config.retrieval_top_k})
