import logging
from fastapi import APIRouter, Query, HTTPException
from app.core.embeddings import search_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

search = APIRouter()

@search.get("/docs")
async def search_documents(q: str = Query(...)):
    logger.info(f"Logging Received query: {q}")
    try:
        # enhanced_query = enhance_query_with_llm(q)
        results = search_embeddings(q)
        if results is None:
            return {"documents": None}
        return {"documents": results}
    except Exception as e:
        logger.error(f" Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
