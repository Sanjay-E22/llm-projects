from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. Load PDF
loader = PyPDFLoader("_Resume  .pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Convert to embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

# 5. Load LLM (Ollama)
llm = Ollama(model="mistral")

print("RAG Chatbot Ready (type 'exit' to stop)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # 6. Search relevant chunks
    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below.
    
    Context:
    {context}
    
    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAI:", response, "\n")