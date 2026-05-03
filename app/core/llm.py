from langchain_community.llms import Ollama
from app.config import OLLAMA_MODEL


llm = Ollama(
    model=OLLAMA_MODEL
)


# import os

# from langchain_community.llms import Ollama


# llm = Ollama(
#     model="phi3",
#     base_url=os.getenv(
#         "OLLAMA_HOST",
#         "http://localhost:11434"
#     )
# )