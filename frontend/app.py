import streamlit as st
import requests


BACKEND_URL = "http://localhost:8000"


st.title(
    "📄 AI PDF Assistant"
)

# Upload section
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    if st.button(
        "Process PDF"
    ):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        with st.spinner(
            "Processing PDF..."
        ):

            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files
            )

        if response.status_code == 200:

            st.success(
                "PDF indexed successfully!"
            )

        else:

            st.error(
                "Upload failed"
            )

# Question section
question = st.text_input(
    "Ask question"
)


if st.button(
    "Ask"
):

    if question:

        with st.spinner(
            "Thinking..."
        ):

            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "question": question
                }
            )

            answer = response.json()

            st.write(
                answer["answer"]
            )