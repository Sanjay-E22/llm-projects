from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_PATH


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH
)