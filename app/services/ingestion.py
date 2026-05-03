import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.utils.loaders import load_pdf
from app.core.embeddings import embedding_model
from app.config import VECTOR_DB_PATH


def ingest_pdfs(pdf_paths):

    documents = []

    for path in pdf_paths:
        docs = load_pdf(path)
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    db.save_local(VECTOR_DB_PATH)

    return True