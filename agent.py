"""Agente inteligente simplificado e seguro para ambiente offline.

O foco desta implementação é manter o fluxo conversacional funcional sem
dependências externas (OpenAI, Ollama ou Pinecone), permitindo que os testes
rodem sem acesso à rede.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from conversation_manager import ConversationManager
from router import QuestionClassifier
from tools import get_pinecone_tool
from config import logger


class IntelligentAgent:
    """Agente com roteamento básico e armazenamento de contexto."""

    def __init__(self, session_id: str = "default", use_openai_for_generation: bool = False):
        self.session_id = session_id
        self.conversation = ConversationManager(session_id)
        self.classifier = QuestionClassifier()
        self.pinecone_tool = get_pinecone_tool()

        logger.info(f"✓ IntelligentAgent inicializado (sessão: {session_id})")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def process_message(self, user_message: str) -> str:
        """Processa a mensagem do usuário e registra o turno."""

        history_messages = self._build_history_for_llm()
        result = self.process_question(user_message, history_messages)

        # Atualiza estado de clarificação
        if result["question_type"] == "clarification_needed":
            self.conversation.set_awaiting_clarification(topic=user_message)
        else:
            self.conversation.clear_clarification_state()

        self.conversation.add_turn(
            user_message=user_message,
            assistant_message=result["answer"],
            question_type=result["question_type"],
            metadata={"used_tool": result.get("used_tool", False)},
        )

        return result["answer"]

    def get_conversation_context(self) -> Dict[str, Any]:
        """Expose o contexto atual da conversa."""

        return self.conversation.get_context_info()

    # ------------------------------------------------------------------
    # Roteamento interno
    # ------------------------------------------------------------------
    def process_question(
        self,
        question: str,
        history: Optional[List[Any]] = None,
        k: int = 5,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processa uma pergunta e direciona para o handler adequado."""

        history = history or []
        question_type = self.classifier.classify(question, history)
        logger.info(f"Processando pergunta '{question}' como '{question_type}'")

        handlers = {
            "greetings": self._handle_greeting,
            "farewell": self._handle_farewell,
            "clarification_needed": self._handle_clarification,
            "internal_docs": lambda q, h: self._handle_internal_docs(q, h, k, namespace),
            "general_knowledge": self._handle_general_knowledge,
        }

        handler = handlers.get(question_type, self._handle_clarification)
        return handler(question, history)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _handle_greeting(self, question: str, history: List[Any]) -> Dict[str, Any]:
        answer = (
            "Olá! Sou seu assistente virtual. Como posso ajudar hoje? "
            "Se precisar de informações sobre políticas internas, é só dizer."
        )
        return {
            "answer": answer,
            "question_type": "greetings",
            "used_tool": False,
            "sources": None,
        }

    def _handle_farewell(self, question: str, history: List[Any]) -> Dict[str, Any]:
        answer = "Agradeço o contato! Se precisar de algo mais, estou à disposição."
        return {
            "answer": answer,
            "question_type": "farewell",
            "used_tool": False,
            "sources": None,
        }

    def _handle_clarification(self, question: str, history: List[Any]) -> Dict[str, Any]:
        answer = (
            "Para te ajudar melhor, poderia detalhar um pouco mais o que deseja? "
            "Se for sobre procedimentos internos, diga qual processo ou departamento."
        )
        return {
            "answer": answer,
            "question_type": "clarification_needed",
            "used_tool": False,
            "sources": None,
        }

    def _handle_general_knowledge(self, question: str, history: List[Any]) -> Dict[str, Any]:
        answer = (
            "Aqui vai um resumo rápido: "
            f"{question} é um tema importante. Posso trazer mais detalhes específicos se quiser."
        )
        return {
            "answer": answer,
            "question_type": "general_knowledge",
            "used_tool": False,
            "sources": None,
        }

    def _handle_internal_docs(
        self,
        question: str,
        history: List[Any],
        k: int,
        namespace: Optional[str],
    ) -> Dict[str, Any]:
        logger.info("Handler: internal_docs - usando ferramenta de busca simulada")

        try:
            search_results = self.pinecone_tool.search(
                query=question,
                k=k,
                namespace=namespace,
                rerank_method="none",
                rerank_top_k=min(3, k),
            )
        except Exception as exc:  # pragma: no cover - fallback de segurança
            logger.warning(f"Busca falhou ({exc}); retornando resposta padrão")
            search_results = []

        if not search_results:
            answer = (
                "Não encontrei documentos relevantes sobre esse assunto na base "
                "de conhecimento. Pode reformular a pergunta ou fornecer mais detalhes?"
            )
            return {
                "answer": answer,
                "question_type": "internal_docs",
                "used_tool": True,
                "tool_name": "pinecone_search",
                "sources": [],
                "num_docs_found": 0,
            }

        context = self.pinecone_tool.format_results_for_context(search_results, max_results=3)
        answer = (
            "Encontrei informações nos documentos internos e gerei um resumo:\n\n"
            f"{context}\n\nSe precisar de mais detalhes, posso aprofundar em algum dos itens."
        )

        sources = [
            {
                "rank": i + 1,
                "source": result.formatted_source,
                "score": result.score,
                "rerank_score": result.rerank_score,
                "content_preview": result.content[:500],
                "metadata": result.metadata,
            }
            for i, result in enumerate(search_results[:3])
        ]

        return {
            "answer": answer,
            "question_type": "internal_docs",
            "used_tool": True,
            "tool_name": "pinecone_search",
            "sources": sources,
            "num_docs_found": len(search_results),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _build_history_for_llm(self) -> List[Any]:
        """Converte o histórico da conversa em lista simples de mensagens.

        Mesmo sem LLM externo, manter o formato facilita futuras integrações.
        """

        return [
            {"role": "user", "content": turn.user_message}
            if idx % 2 == 0
            else {"role": "assistant", "content": turn.assistant_message}
            for idx, turn in enumerate(self.conversation.history)
        ]
