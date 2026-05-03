# import os

# from fastapi import APIRouter, UploadFile, File

from app.schemas.request import QuestionRequest
# from app.services.ingestion import ingest_pdfs
from app.services.retrieval import ask_question

from typing import List
from fastapi import APIRouter, UploadFile, File
import os

from app.services.ingestion import ingest_pdfs

router = APIRouter()


@router.post("/upload")
async def upload_pdfs(
    file: UploadFile = File(...)
):

    os.makedirs("data", exist_ok=True)

    saved_paths = []

    # for file in files:

    file_path = os.path.join(
        "data",
        file.filename
    )

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    saved_paths.append(file_path)

    ingest_pdfs(saved_paths)

    return {
        "status": "success",
        "files_uploaded": len(saved_paths)
    }
@router.post("/ask")
def ask(
    request: QuestionRequest
):

    answer = ask_question(
        request.question
    )

    return {
        "answer": answer
    }