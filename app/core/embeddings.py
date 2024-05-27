import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv("./.env"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def index_exists(index_name):
    indexes = pinecone.list_indexes()
    logger.info(f"LOGGING Existing indexes: {indexes}")
    for index in indexes:
        if index["name"] == index_name:
            return True
    return False

# Check if the index exists, if not, create it
index_name = "semantic-search-index"
if not index_exists(index_name):
    pinecone.create_index(
        index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

pinecone_index = pinecone.Index("semantic-search-index")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text: str):
    return model.encode(text).tolist()

def store_embedding(doc_id: str, embedding: list):
    pinecone_index.upsert([(doc_id, embedding)])

def search_embeddings(query: str):
    query_embedding = generate_embedding(query)
    logger.info(f"LOGGING query_embedding: {query_embedding}")
    results = pinecone_index.query(vector=query_embedding, top_k=2)
    if not results['matches']:
        return None
    return [match['id'] for match in results['matches']]

# def enhance_query_with_llm(query: str):
#     response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[
#         {"role": "user", "content": f"Enhance the following search query for better context relevance: {query}"}
#     ])
#     enhanced_query = response['choices'][0]['message']['content']
#     logger.info(f"LOGGING Enhanced query: {enhanced_query}")
#     return enhanced_query
