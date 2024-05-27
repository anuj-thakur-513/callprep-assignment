from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import upload
from app.api.search import search

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(upload)
app.include_router(search)
