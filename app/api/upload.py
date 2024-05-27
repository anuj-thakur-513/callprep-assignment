from fastapi import APIRouter, UploadFile, File
import PyPDF2
from app.core.embeddings import generate_embedding, store_embedding

upload = APIRouter()

@upload.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}

    # Read PDF content
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    
    # Generate and store embedding
    embedding = generate_embedding(text)
    store_embedding(str(file.filename), embedding)

    return {"message": "File uploaded successfully"}
