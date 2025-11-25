import os
from dotenv import load_dotenv

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")


PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")