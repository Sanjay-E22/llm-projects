import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="Multi PDF AI Assistant")

st.title("📚 Multi PDF RAG Assistant")

uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if "db" not in st.session_state:
    st.session_state.db = None

if uploaded_files:

    if st.button("Process PDFs"):

        all_docs = []

        with st.spinner("Reading PDFs..."):

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)

                os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(chunks, embeddings)

        st.session_state.db = db

        st.success("PDFs processed successfully!")

# Question Section
question = st.text_input("Ask question from all PDFs")

if question and st.session_state.db:

    llm = Ollama(model="phi3")

    docs = st.session_state.db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant.
Answer only from provided context.
If answer not found, say I don't know.

Context:
{context}

Question:
{question}
"""

    with st.spinner("Thinking..."):
        answer = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(answer)