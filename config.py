import os
from dotenv import load_dotenv
import logging

# -------------------

load_dotenv(override=True)

# Logging 
LOG_LEVEL =  os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Pinecone
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")








def validate_config():
    """Valida configurações críticas."""
    errors = []

    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY não configurada.")

    if not PINECONE_INDEX_NAME:
        errors.append("PINECONE_INDEX ou PINECONE_INDEX_NAME não configurada.")

    if errors:
        raise RuntimeError(f"Erro de configuração:\n{'\n'.join(errors)}")
    
    logger.info("✓ Configurações validadas com sucesso")
    logger.info(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")

if __name__ != "__main__":
    validate_config()