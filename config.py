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




# --- Variáveis de Ambiente ---
LOG_LEVEL             = os.getenv("LOG_LEVEL", "INFO").upper()
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL      = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX")
DEFAULT_NAMESPACE     = os.getenv("PINECONE_NAMESPACE", "default")
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K", "2"))
OPENAI_KEY            = os.getenv("OPENAI_API_KEY")

RERANK_METHOD_DEFAULT = os.getenv("RERANK_METHOD_DEFAULT", "none").lower()
RERANK_TOP_K_DEFAULT  = int(os.getenv("RERANK_TOP_K_DEFAULT", "0"))  # 0 or empty means “use top_k if not specified”
CROSS_ENCODER_MODEL   = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE     = int(os.getenv("RERANK_BATCH_SIZE", "16"))



def validate_config():
    """Valida configurações críticas."""
    errors = []

    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY não configurada.")

    if not PINECONE_INDEX_NAME:
        errors.append("PINECONE_INDEX ou PINECONE_INDEX_NAME não configurada.")

    if errors:
        error_msg = '\n'.join(errors)
        raise RuntimeError(f"Erro de configuração:\n{error_msg}")
    
    logger.info("✓ Configurações validadas com sucesso")
    logger.info(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")

if __name__ != "__main__":
    validate_config()