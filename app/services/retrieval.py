from app.core.vectorstore import load_vector_db
from app.core.llm import llm


def ask_question(question):

    db = load_vector_db()

    docs = db.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt)