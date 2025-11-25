"""Conversation Manager - Gerencia o histórico e contexto da conversa."""

from __future__ import annotations
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Representa uma troca na conversa."""
    timestamp: datetime
    user_message: str
    assistant_message: str
    question_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Gerencia o contexto e histórico da conversa."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or self._generate_session_id()
        self.history: List[ConversationTurn] = []
        self.context: Dict[str, Any] = {
            "awaiting_clarification": False,
            "last_topic": None,
            "clarification_attempts": 0
        }

    @staticmethod
    def _generate_session_id() -> str:
        """Gera um ID único para a sessão."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        question_type: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Adiciona uma troca à conversa."""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_message=assistant_message,
            question_type=question_type,
            metadata=metadata or {}
        )
        self.history.append(turn)

    def get_last_turn(self) -> ConversationTurn | None:
        """Retorna a última troca da conversa."""
        return self.history[-1] if self.history else None

    def get_history_summary(self, last_n: int = 5) -> str:
        """Retorna um resumo do histórico recente."""
        if not self.history:
            return "Sem histórico anterior."

        recent = self.history[-last_n:]
        summary = []
        for turn in recent:
            summary.append(f"Usuário: {turn.user_message}")
            summary.append(f"Assistente: {turn.assistant_message}")
        return "\n".join(summary)

    def is_awaiting_clarification(self) -> bool:
        """Verifica se está aguardando clarificação."""
        return self.context.get("awaiting_clarification", False)

    def set_awaiting_clarification(self, topic: str = None) -> None:
        """Marca que está aguardando clarificação."""
        self.context["awaiting_clarification"] = True
        self.context["last_topic"] = topic
        self.context["clarification_attempts"] = self.context.get("clarification_attempts", 0) + 1

    def clear_clarification_state(self) -> None:
        """Limpa o estado de clarificação."""
        self.context["awaiting_clarification"] = False
        self.context["clarification_attempts"] = 0

    def get_context_info(self) -> Dict[str, Any]:
        """Retorna informações de contexto."""
        return {
            "session_id": self.session_id,
            "turn_count": len(self.history),
            "awaiting_clarification": self.is_awaiting_clarification(),
            "last_topic": self.context.get("last_topic"),
            "clarification_attempts": self.context.get("clarification_attempts", 0)
        }
