"""Router - Classifica perguntas usando OpenAI API."""

from __future__ import annotations
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import OPENAI_KEY, logger

QuestionType = Literal["greetings", "farewell", "clarification_needed", "internal_docs", "general_knowledge"]


class QuestionClassifier:
    """Classifica perguntas do usuário usando OpenAI."""

    def __init__(self):
        """Inicializa o classificador."""
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_KEY
        )
        logger.info("QuestionClassifier inicializado com gpt-4o-mini")

    def classify(self, question: str, conversation_history: list = None) -> QuestionType:
        """Classifica a pergunta do usuário.

        Args:
            question: Pergunta do usuário
            conversation_history: Histórico da conversa para contexto (opcional)

        Returns:
            Tipo da pergunta classificada
        """
        system_prompt = """Você é um classificador de perguntas preciso. Analise a pergunta do usuário e classifique em UMA das seguintes categorias:

1. **greetings**: Saudações iniciais
   - Exemplos: "Olá", "Oi", "Bom dia", "Tudo bem?", "E aí"

2. **farewell**: Despedidas
   - Exemplos: "Tchau", "Até logo", "Obrigado, é só isso", "Valeu, pode encerrar"

3. **clarification_needed**: Perguntas vagas ou ambíguas que precisam de mais detalhes
   - Exemplos: "Quero saber sobre reembolso" (não especifica contexto)
   - "Como funciona férias?" (vago, sem contexto)
   - Perguntas muito curtas sem contexto suficiente

4. **internal_docs**: Perguntas específicas sobre procedimentos, políticas, normas ou documentos internos da empresa
   - Exemplos:
     * "Como a Unimed realiza reembolso para colaboradores em viagem?"
     * "Qual o procedimento de férias segundo a política da empresa?"
     * "O que diz o manual sobre horas extras?"
     * "Quais são os benefícios oferecidos pela empresa?"
   - IMPORTANTE: Se menciona empresa, políticas, manuais, procedimentos internos → internal_docs

5. **general_knowledge**: Perguntas conceituais ou de conhecimento geral
   - Exemplos:
     * "O que é reembolso?"
     * "Explique o conceito de férias"
     * "Como funciona um banco de dados?"
     * "Qual a capital da França?"

REGRAS DE DECISÃO:
- Menciona empresa/políticas/manuais/procedimentos específicos → internal_docs
- Pergunta vaga sem contexto → clarification_needed
- Pergunta conceitual/geral → general_knowledge
- Saudação → greetings
- Despedida → farewell

Responda APENAS com uma das seguintes palavras exatas: greetings, farewell, clarification_needed, internal_docs, general_knowledge"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Pergunta do usuário: {question}")
        ]

        try:
            response = self.model.invoke(messages)
            classification = response.content.strip().lower()

            # Valida a resposta
            valid_types: list[QuestionType] = [
                "greetings", "farewell", "clarification_needed",
                "internal_docs", "general_knowledge"
            ]

            if classification not in valid_types:
                logger.warning(
                    f"Classificação inválida: '{classification}'. Usando 'clarification_needed'"
                )
                return "clarification_needed"

            logger.info(f"✓ Pergunta classificada como: {classification}")
            return classification

        except Exception as e:
            logger.error(f"Erro ao classificar pergunta: {e}")
            return "clarification_needed"
