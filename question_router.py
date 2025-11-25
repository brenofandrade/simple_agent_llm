"""Question Router - Classifica o tipo de pergunta do usuário."""

from __future__ import annotations
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import logger

QuestionType = Literal["greeting", "clarification_needed", "internal_docs", "general_knowledge", "farewell"]


class QuestionRouter:
    """Classifica perguntas do usuário em diferentes categorias."""

    def __init__(self, model: ChatOpenAI):
        self.model = model

    def classify(self, question: str, conversation_history: list = None) -> QuestionType:
        """Classifica a pergunta do usuário.

        Args:
            question: Pergunta do usuário
            conversation_history: Histórico da conversa para contexto

        Returns:
            Tipo da pergunta classificada
        """
        system_prompt = """Você é um classificador de perguntas. Analise a pergunta do usuário e classifique em uma das seguintes categorias:

1. **greeting**: Saudações como "Olá", "Oi", "Bom dia", "Tudo bem?"
2. **farewell**: Despedidas como "Tchau", "Até logo", "Obrigado, é só isso"
3. **clarification_needed**: Perguntas vagas ou ambíguas que precisam de mais detalhes. Exemplos:
   - "Quero saber sobre reembolso" (não especifica se quer informação geral ou procedimento interno)
   - "Como funciona férias?" (não especifica contexto)
   - Perguntas muito curtas ou sem contexto suficiente
4. **internal_docs**: Perguntas sobre procedimentos, políticas, normas ou documentos internos específicos da empresa. Exemplos:
   - "Como a Unimed realiza reembolso para colaboradores em viagem?"
   - "Qual o procedimento de férias segundo a política da empresa?"
   - "O que diz o manual sobre horas extras?"
5. **general_knowledge**: Perguntas sobre conhecimento geral que não requerem documentos internos. Exemplos:
   - "O que é reembolso?"
   - "Explique o conceito de férias"
   - "Como funciona um banco de dados?"

IMPORTANTE:
- Se a pergunta menciona especificamente a empresa, políticas, manuais, normas → internal_docs
- Se a pergunta é vaga sem contexto suficiente → clarification_needed
- Se é conceitual/geral sem mencionar especificidades da empresa → general_knowledge

Responda APENAS com uma das seguintes palavras: greeting, farewell, clarification_needed, internal_docs, general_knowledge"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Pergunta do usuário: {question}")
        ]

        try:
            response = self.model.invoke(messages)
            classification = response.content.strip().lower()

            # Valida a resposta
            valid_types = ["greeting", "farewell", "clarification_needed", "internal_docs", "general_knowledge"]
            if classification not in valid_types:
                logger.warning(f"Classificação inválida: {classification}. Usando 'clarification_needed'")
                return "clarification_needed"

            logger.info(f"Pergunta classificada como: {classification}")
            return classification

        except Exception as e:
            logger.error(f"Erro ao classificar pergunta: {e}")
            return "clarification_needed"
