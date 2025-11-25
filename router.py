"""Router - Classifica perguntas usando heurísticas locais (offline-friendly)."""

from __future__ import annotations

from typing import Literal

from config import logger

QuestionType = Literal[
    "greetings",
    "farewell",
    "clarification_needed",
    "internal_docs",
    "general_knowledge",
]


class QuestionClassifier:
    """Classificador simples que não depende de chamadas externas."""

    GREETINGS = {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem?", "e aí"}
    FAREWELLS = {"tchau", "até logo", "ate logo", "obrigado", "valeu", "até mais", "ate mais"}

    def classify(self, question: str, conversation_history: list | None = None) -> QuestionType:
        """Classifica a pergunta do usuário usando regras determinísticas."""

        normalized = (question or "").strip().lower()

        if not normalized:
            logger.debug("Pergunta vazia: solicitando clarificação")
            return "clarification_needed"

        if any(greet in normalized for greet in self.GREETINGS):
            return "greetings"

        if any(farewell in normalized for farewell in self.FAREWELLS):
            return "farewell"

        internal_markers = [
            "empresa",
            "política",
            "politica",
            "manual",
            "procedimento",
            "unimed",
            "benefício",
            "beneficio",
            "colaborador",
        ]
        if any(marker in normalized for marker in internal_markers):
            return "internal_docs"

        if len(normalized) < 20:
            return "clarification_needed"

        return "general_knowledge"
