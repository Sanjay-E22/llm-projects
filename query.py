from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS DB
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Load LLM
llm = Ollama(model="mistral")

print("RAG system ready (type 'exit' to stop)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant.
    Answer ONLY from the provided context.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAI:", response, "\n")