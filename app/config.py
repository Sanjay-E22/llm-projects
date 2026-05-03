from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

VECTOR_DB_PATH = str(BASE_DIR / "vectorstore")
EMBEDDING_MODEL_PATH = "C:\\Users\\sanja\\llm\\llm-projects\\models\\embedding_model"

OLLAMA_MODEL = "mistral"