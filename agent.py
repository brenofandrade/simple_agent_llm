"""Agent - Agente inteligente que usa tools baseado na classificação."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from router import QuestionClassifier, QuestionType
from tools import get_pinecone_tool, SearchResult
from config import (
    OPENAI_KEY,
    GENERATION_MODEL,
    OLLAMA_BASE_URL,
    logger
)


class IntelligentAgent:
    """Agente inteligente com roteamento e tools."""

    def __init__(
        self,
        session_id: str = "default",
        use_openai_for_generation: bool = False
    ):
        """Inicializa o agente.

        Args:
            session_id: ID da sessão
            use_openai_for_generation: Se True, usa OpenAI; se False, usa Ollama
        """
        self.session_id = session_id
        self.classifier = QuestionClassifier()
        self.pinecone_tool = get_pinecone_tool()

        # Modelo de geração
        if use_openai_for_generation and OPENAI_KEY:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                openai_api_key=OPENAI_KEY
            )
            logger.info("Usando OpenAI para geração")
        else:
            self.llm = ChatOllama(
                model=GENERATION_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.7
            )
            logger.info(f"Usando Ollama para geração: {GENERATION_MODEL}")

        # Prompts
        self._setup_prompts()

        logger.info(f"✓ IntelligentAgent inicializado - Session: {session_id}")

    def _setup_prompts(self):
        """Configura os prompts do agente."""
        # Prompt para internal_docs com contexto
        self.prompt_internal_docs = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é o Assistente interno da empresa para dúvidas de colaboradores.

INSTRUÇÕES:
1. Use SOMENTE as informações dos documentos fornecidos no Contexto
2. Seja preciso, objetivo e profissional
3. Sempre mencione a fonte quando disponível (Manual, Norma, etc.)
4. Se o contexto não tiver informação suficiente, seja honesto
5. Estruture a resposta de forma clara
6. Não inclua nomes de arquivos ou códigos de documentos
7. Não repita a pergunta do usuário

FORMATO DA RESPOSTA:
- Apresente as informações de forma estruturada
- Use citações curtas com Markdown (>) quando apropriado
- Seja cordial e ofereça ajuda adicional ao final

Se não houver informação suficiente no Contexto, use uma destas mensagens:
- "Não localizei informação suficiente no Contexto para responder com segurança. Pode reformular a pergunta?"
- "O Contexto traz menções relacionadas, mas sem detalhes suficientes. Recomendo validar com a área responsável."
"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("system", "# Contexto dos Documentos:\n{context}"),
            ("user", "# Pergunta:\n{input}")
        ])

        # Prompt para general_knowledge
        self.prompt_general = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um assistente útil e responsável.

INSTRUÇÕES:
1. Responda de forma clara, precisa e educativa
2. Use conhecimento geral, não invente fatos sobre empresas
3. Nunca forneça dados pessoais ou confidenciais
4. Não dê aconselhamento jurídico/financeiro específico
5. Se algo for especulativo, diga que não sabe
6. Seja objetivo mas completo
7. Mantenha tom profissional e amigável
"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        # Prompt para greetings
        self.prompt_greeting = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um assistente cordial e profissional.

Responda à saudação de forma breve e amigável.
Convide o usuário a fazer sua pergunta.
Seja natural e acolhedor."""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        # Prompt para farewell
        self.prompt_farewell = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um assistente cordial e profissional.

Responda à despedida de forma simpática.
Agradeça o contato e reforce que está disponível.
Seja breve e natural."""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        # Prompt para clarification
        self.prompt_clarification = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você precisa ajudar a clarificar a dúvida do usuário.

INSTRUÇÕES:
1. Faça 1-2 perguntas objetivas
2. Pergunte se ele busca informações gerais OU procedimentos internos
3. Seja cordial e direto
4. Mantenha as perguntas curtas

Exemplo:
Usuário: "Quero saber sobre reembolso"
Você: "Claro! Você quer informações gerais sobre reembolso ou o procedimento específico da empresa?"
"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

    def process_question(
        self,
        question: str,
        history: List[Any] = None,
        k: int = 5,
        namespace: str = None
    ) -> Dict[str, Any]:
        """Processa a pergunta do usuário.

        Args:
            question: Pergunta do usuário
            history: Histórico da conversa
            k: Número de documentos a buscar (para internal_docs)
            namespace: Namespace do Pinecone

        Returns:
            Dicionário com resposta e metadados
        """
        history = history or []

        logger.info(f"Processando pergunta: '{question}'")

        # Classifica a pergunta
        question_type = self.classifier.classify(question, history)

        # Processa baseado no tipo
        if question_type == "greetings":
            return self._handle_greeting(question, history)

        elif question_type == "farewell":
            return self._handle_farewell(question, history)

        elif question_type == "clarification_needed":
            return self._handle_clarification(question, history)

        elif question_type == "internal_docs":
            return self._handle_internal_docs(question, history, k, namespace)

        elif question_type == "general_knowledge":
            return self._handle_general_knowledge(question, history)

        else:
            # Fallback
            return self._handle_clarification(question, history)

    def _handle_greeting(self, question: str, history: List[Any]) -> Dict[str, Any]:
        """Trata saudações."""
        logger.info("Handler: greetings")

        ai_msg = (self.prompt_greeting | self.llm).invoke({
            "input": question,
            "history": history
        })

        return {
            "answer": ai_msg.content,
            "question_type": "greetings",
            "used_tool": False,
            "sources": None
        }

    def _handle_farewell(self, question: str, history: List[Any]) -> Dict[str, Any]:
        """Trata despedidas."""
        logger.info("Handler: farewell")

        ai_msg = (self.prompt_farewell | self.llm).invoke({
            "input": question,
            "history": history
        })

        return {
            "answer": ai_msg.content,
            "question_type": "farewell",
            "used_tool": False,
            "sources": None
        }

    def _handle_clarification(self, question: str, history: List[Any]) -> Dict[str, Any]:
        """Trata perguntas que precisam clarificação."""
        logger.info("Handler: clarification_needed")

        ai_msg = (self.prompt_clarification | self.llm).invoke({
            "input": question,
            "history": history
        })

        return {
            "answer": ai_msg.content,
            "question_type": "clarification_needed",
            "used_tool": False,
            "sources": None
        }

    def _handle_general_knowledge(self, question: str, history: List[Any]) -> Dict[str, Any]:
        """Trata perguntas de conhecimento geral."""
        logger.info("Handler: general_knowledge")

        ai_msg = (self.prompt_general | self.llm).invoke({
            "input": question,
            "history": history
        })

        return {
            "answer": ai_msg.content,
            "question_type": "general_knowledge",
            "used_tool": False,
            "sources": None
        }

    def _handle_internal_docs(
        self,
        question: str,
        history: List[Any],
        k: int,
        namespace: Optional[str]
    ) -> Dict[str, Any]:
        """Trata perguntas sobre documentos internos.

        Aqui o agente DECIDE usar a tool do Pinecone.
        """
        logger.info("Handler: internal_docs - Usando Pinecone tool")

        # AGENTE DECIDE: Para internal_docs, SEMPRE usa a tool
        search_results = self.pinecone_tool.search(
            query=question,
            k=k,
            namespace=namespace,
            rerank_method="cross-encoder",
            rerank_top_k=3
        )

        if not search_results:
            # Nenhum documento encontrado
            answer = (
                "Não encontrei documentos relevantes sobre esse assunto "
                "na base de conhecimento. Pode reformular a pergunta ou "
                "fornecer mais detalhes?"
            )
            return {
                "answer": answer,
                "question_type": "internal_docs",
                "used_tool": True,
                "tool_name": "pinecone_search",
                "sources": [],
                "num_docs_found": 0
            }

        # Formata contexto com os documentos
        context = self.pinecone_tool.format_results_for_context(search_results, max_results=3)

        # Gera resposta usando o contexto
        ai_msg = (self.prompt_internal_docs | self.llm).invoke({
            "input": question,
            "history": history,
            "context": context
        })

        # Serializa fontes para resposta
        sources = [
            {
                "rank": i + 1,
                "source": result.formatted_source,
                "score": result.score,
                "rerank_score": result.rerank_score,
                "content_preview": result.content[:500],
                "metadata": result.metadata
            }
            for i, result in enumerate(search_results[:3])
        ]

        return {
            "answer": ai_msg.content,
            "question_type": "internal_docs",
            "used_tool": True,
            "tool_name": "pinecone_search",
            "sources": sources,
            "num_docs_found": len(search_results)
        }
