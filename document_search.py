"""Document Search - Busca em documentos internos usando Pinecone."""

from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, openai_api_key, logger


@dataclass
class SearchResult:
    """Representa um resultado de busca."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str = None

    @property
    def formatted_source(self) -> str:
        """Retorna a fonte formatada."""
        if self.source:
            return self.source
        if "source" in self.metadata:
            return self.metadata["source"]
        if "title" in self.metadata:
            return self.metadata["title"]
        return "Documento Interno"


class DocumentSearcher:
    """Busca em documentos internos usando Pinecone."""

    def __init__(self, relevance_threshold: float = 0.7):
        """Inicializa o buscador de documentos.

        Args:
            relevance_threshold: Limiar mínimo de relevância (0-1)
        """
        self.relevance_threshold = relevance_threshold
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Inicializa Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        logger.info(f"DocumentSearcher inicializado com índice: {PINECONE_INDEX_NAME}")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Busca documentos relevantes para a query.

        Args:
            query: Pergunta/consulta do usuário
            top_k: Número máximo de resultados a retornar

        Returns:
            Lista de resultados relevantes
        """
        try:
            # Gera embedding da query
            query_embedding = self.embeddings.embed_query(query)

            # Busca no Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Processa resultados
            search_results = []
            for match in results.get("matches", []):
                score = match.get("score", 0.0)

                # Filtra por relevância
                if score >= self.relevance_threshold:
                    metadata = match.get("metadata", {})
                    content = metadata.get("text", metadata.get("content", ""))

                    search_results.append(SearchResult(
                        content=content,
                        metadata=metadata,
                        score=score,
                        source=metadata.get("source")
                    ))

            logger.info(f"Encontrados {len(search_results)} documentos relevantes (threshold: {self.relevance_threshold})")
            return search_results

        except Exception as e:
            logger.error(f"Erro ao buscar documentos: {e}")
            return []

    def has_relevant_results(self, results: List[SearchResult]) -> bool:
        """Verifica se há resultados suficientemente relevantes."""
        if not results:
            return False

        # Considera relevante se o melhor resultado tem score alto
        best_score = max(result.score for result in results)
        return best_score >= self.relevance_threshold

    def format_results_for_context(self, results: List[SearchResult], max_results: int = 3) -> str:
        """Formata resultados para usar como contexto na resposta.

        Args:
            results: Resultados da busca
            max_results: Número máximo de resultados a incluir

        Returns:
            String formatada com os resultados
        """
        if not results:
            return "Nenhum documento relevante encontrado."

        formatted = ["=== DOCUMENTOS ENCONTRADOS ===\n"]

        for i, result in enumerate(results[:max_results], 1):
            formatted.append(f"[Documento {i}] {result.formatted_source}")
            formatted.append(f"Relevância: {result.score:.2%}")
            formatted.append(f"Conteúdo:\n{result.content}\n")
            formatted.append("-" * 50)

        return "\n".join(formatted)
