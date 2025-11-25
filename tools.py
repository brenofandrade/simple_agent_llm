"""Tools - Ferramentas disponíveis para o agente."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_pinecone.vectorstores import PineconeVectorStore
from sentence_transformers import CrossEncoder
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    DEFAULT_NAMESPACE,
    OPENAI_KEY,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_K,
    CROSS_ENCODER_MODEL,
    RERANK_BATCH_SIZE,
    logger
)


@dataclass
class SearchResult:
    """Representa um resultado de busca."""
    content: str
    metadata: Dict[str, Any]
    score: float
    rerank_score: Optional[float] = None

    @property
    def formatted_source(self) -> str:
        """Retorna a fonte formatada."""
        meta = self.metadata
        source = (
            meta.get("source") or
            meta.get("file_path") or
            meta.get("filename") or
            meta.get("title") or
            "Documento Interno"
        )

        page = meta.get("page") or meta.get("page_number")
        if page:
            return f"{source} (página {page})"
        return source


class PineconeSearchTool:
    """Tool de busca híbrida no Pinecone."""

    def __init__(self, use_openai_embeddings: bool = True):
        """Inicializa a ferramenta de busca.

        Args:
            use_openai_embeddings: Se True, usa OpenAI; se False, usa Ollama
        """
        # Inicializa embeddings
        if use_openai_embeddings and OPENAI_KEY:
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
            logger.info("Usando OpenAI embeddings")
        else:
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            logger.info(f"Usando Ollama embeddings: {EMBEDDING_MODEL}")

        # Inicializa Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

        # Cross-encoder para reranking
        self.cross_encoder = None

        logger.info(f"✓ PineconeSearchTool inicializado - Index: {PINECONE_INDEX_NAME}")

    def _get_cross_encoder(self) -> CrossEncoder:
        """Inicializa cross-encoder lazy."""
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"Cross-encoder carregado: {CROSS_ENCODER_MODEL}")
        return self.cross_encoder

    def _generate_query_variants(self, query: str, n: int = 3) -> List[str]:
        """Gera variações da query para busca mais robusta.

        Args:
            query: Query original
            n: Número de variações

        Returns:
            Lista com query original e variações
        """
        variants = [query]

        # Adiciona variações simples
        q_lower = query.lower().strip()
        q_upper = query.upper().strip()

        if q_lower not in variants:
            variants.append(q_lower)
        if q_upper not in variants and len(variants) < n:
            variants.append(q_upper)

        return variants[:n]

    def _retrieve_with_variants(
        self,
        queries: List[str],
        k: int,
        namespace: str
    ) -> List[Document]:
        """Busca documentos usando múltiplas variações da query.

        Args:
            queries: Lista de queries
            k: Número de documentos por query
            namespace: Namespace do Pinecone

        Returns:
            Lista de documentos únicos
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "namespace": namespace}
        )

        collected: List[Document] = []
        seen = set()

        for q in queries:
            try:
                docs = retriever.get_relevant_documents(q)
                for doc in docs:
                    # Usa hash do conteúdo para evitar duplicatas
                    key = hash((doc.page_content or "").strip())
                    if key not in seen:
                        seen.add(key)
                        collected.append(doc)
            except Exception as e:
                logger.error(f"Erro ao buscar documentos para '{q}': {e}")
                continue

        logger.info(f"Coletados {len(collected)} documentos únicos de {len(queries)} queries")
        return collected

    def _rerank_by_embedding(
        self,
        query: str,
        docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """Reranking por similaridade de embeddings.

        Args:
            query: Query original
            docs: Documentos a reranquear
            top_k: Número de documentos a retornar

        Returns:
            Documentos reranqueados
        """
        if not docs:
            return docs

        # Gera embeddings
        query_vector = np.array(self.embeddings.embed_query(query), dtype=float)
        doc_texts = [d.page_content or "" for d in docs]
        doc_vectors = self.embeddings.embed_documents(doc_texts)
        doc_vectors = [np.array(vec, dtype=float) for vec in doc_vectors]

        # Normaliza vetores
        if np.linalg.norm(query_vector) != 0:
            query_vector = query_vector / np.linalg.norm(query_vector)

        normalized_docs = []
        for vec in doc_vectors:
            if np.linalg.norm(vec) != 0:
                normalized_docs.append(vec / np.linalg.norm(vec))
            else:
                normalized_docs.append(vec)

        # Calcula similaridade
        scores = [float(np.dot(query_vector, doc_vec)) for doc_vec in normalized_docs]

        # Ordena e retorna top_k
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, doc in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = round(score, 6)
            doc.metadata["rerank_method"] = "embedding"
            reranked.append(doc)

        return reranked

    def _rerank_by_cross_encoder(
        self,
        query: str,
        docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """Reranking usando cross-encoder.

        Args:
            query: Query original
            docs: Documentos a reranquear
            top_k: Número de documentos a retornar

        Returns:
            Documentos reranqueados
        """
        if not docs:
            return docs

        try:
            cross_model = self._get_cross_encoder()
            sentences = [(query, d.page_content) for d in docs]
            scores = cross_model.predict(sentences, batch_size=RERANK_BATCH_SIZE)

            # Ordena e retorna top_k
            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            reranked = []
            for score, doc in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_method"] = "cross_encoder"
                reranked.append(doc)

            return reranked

        except Exception as e:
            logger.error(f"Erro no cross-encoder: {e}. Usando embedding reranking")
            return self._rerank_by_embedding(query, docs, top_k)

    def search(
        self,
        query: str,
        k: int = None,
        namespace: str = None,
        rerank_method: str = "cross-encoder",
        rerank_top_k: int = None
    ) -> List[SearchResult]:
        """Busca híbrida com reranking no Pinecone.

        Args:
            query: Pergunta/consulta do usuário
            k: Número de documentos a buscar (padrão: RETRIEVAL_K)
            namespace: Namespace do Pinecone (padrão: DEFAULT_NAMESPACE)
            rerank_method: Método de reranking ("none", "embedding", "cross-encoder")
            rerank_top_k: Número final de documentos após reranking

        Returns:
            Lista de resultados ordenados por relevância
        """
        k = k or RETRIEVAL_K
        namespace = namespace or DEFAULT_NAMESPACE or "default"
        rerank_top_k = rerank_top_k or k

        logger.info(f"Buscando documentos: query='{query}', k={k}, namespace='{namespace}'")

        # Gera variações da query
        query_variants = self._generate_query_variants(query, n=3)

        # Busca com variações
        docs = self._retrieve_with_variants(query_variants, k=k, namespace=namespace)

        if not docs:
            logger.warning("Nenhum documento encontrado")
            return []

        # Reranking
        rerank_method = rerank_method.lower()
        if rerank_method == "cross-encoder":
            docs = self._rerank_by_cross_encoder(query, docs, rerank_top_k)
        elif rerank_method == "embedding":
            docs = self._rerank_by_embedding(query, docs, rerank_top_k)
        else:
            # Sem reranking, apenas limita
            docs = docs[:rerank_top_k]

        # Converte para SearchResult
        results = []
        for doc in docs:
            results.append(SearchResult(
                content=doc.page_content or "",
                metadata=doc.metadata or {},
                score=doc.metadata.get("score", 0.0),
                rerank_score=doc.metadata.get("rerank_score")
            ))

        logger.info(f"✓ Retornando {len(results)} resultados após reranking")
        return results

    def format_results_for_context(
        self,
        results: List[SearchResult],
        max_results: int = 3
    ) -> str:
        """Formata resultados para contexto do LLM.

        Args:
            results: Resultados da busca
            max_results: Número máximo de resultados a incluir

        Returns:
            String formatada
        """
        if not results:
            return "Nenhum documento relevante encontrado."

        formatted = ["=== DOCUMENTOS ENCONTRADOS ===\n"]

        for i, result in enumerate(results[:max_results], 1):
            formatted.append(f"[Documento {i}] {result.formatted_source}")
            if result.rerank_score:
                formatted.append(f"Relevância: {result.rerank_score:.2%}")
            formatted.append(f"Conteúdo:\n{result.content}\n")
            formatted.append("-" * 70)

        return "\n".join(formatted)


# Instância global da tool
_pinecone_tool: Optional[PineconeSearchTool] = None


def get_pinecone_tool() -> PineconeSearchTool:
    """Retorna instância singleton da PineconeSearchTool."""
    global _pinecone_tool
    if _pinecone_tool is None:
        _pinecone_tool = PineconeSearchTool(use_openai_embeddings=True)
    return _pinecone_tool
