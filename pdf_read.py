from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
 


print("123")


# 1. Load and Split

loader = PyPDFLoader("my bb")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Store in Vector DB
db = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 3. Create the Retriever
retriever = db.as_retriever()

# 4. Generate
llm = ChatOpenAI(model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 5. Ask!
print(qa_chain.invoke("What are the key findings in this document?"))