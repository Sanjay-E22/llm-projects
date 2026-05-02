from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
llm = Ollama(model="mistral")

@app.get("/")
def home():
    return {"message": "RAG API running"}

@app.post("/ask")
def ask(question: str):
    docs = db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer based only on this context:

    {context}

    Question: {question}
    """

    response = llm.invoke(prompt)

    return {"answer": response}