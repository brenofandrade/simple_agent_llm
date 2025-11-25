"""Configuração centralizada com defaults seguros para ambiente offline."""

from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv


load_dotenv(override=True)


# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# OpenAI / Ollama
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_DSUNIBLU") or os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX") or "local-mock-index"
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "2"))

# Reranking
RERANK_METHOD_DEFAULT = os.getenv("RERANK_METHOD_DEFAULT", "none").lower()
RERANK_TOP_K_DEFAULT = int(os.getenv("RERANK_TOP_K_DEFAULT", "0"))
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))


def validate_config() -> List[str]:
    """Valida configurações críticas, retornando avisos ao invés de erros.

    O objetivo é permitir que o código rode em ambientes offline (como os
    testes automatizados), mantendo o feedback visível para configuração
    futura.
    """

    warnings = []

    if not PINECONE_API_KEY:
        warnings.append("PINECONE_API_KEY não configurada; usando modo mock.")

    if not os.getenv("PINECONE_INDEX"):
        warnings.append("PINECONE_INDEX não configurada; usando índice mock.")

    if not OPENAI_KEY:
        warnings.append("OPENAI_API_KEY não configurada; classificações usarão heurísticas locais.")

    for warning in warnings:
        logger.warning(warning)

    if not warnings:
        logger.info("✓ Configurações validadas com sucesso")
        logger.info(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")

    return warnings


if __name__ != "__main__":
    validate_config()