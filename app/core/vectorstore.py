import os

from langchain_community.vectorstores import FAISS

from app.core.embeddings import embedding_model
from app.config import VECTOR_DB_PATH


def load_vector_db():

    if not os.path.exists(VECTOR_DB_PATH):
        return None

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )