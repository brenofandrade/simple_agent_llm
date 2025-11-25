"""Agent - Agente principal com roteamento inteligente de perguntas."""

from __future__ import annotations
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from question_router import QuestionRouter, QuestionType
from conversation_manager import ConversationManager
from document_search import DocumentSearcher
from config import openai_api_key, logger


class IntelligentAgent:
    """Agente inteligente com roteamento de perguntas."""

    def __init__(self, session_id: str = None):
        """Inicializa o agente.

        Args:
            session_id: ID da sessão (opcional)
        """
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )

        self.router = QuestionRouter(self.model)
        self.conversation = ConversationManager(session_id)
        self.document_searcher = DocumentSearcher(relevance_threshold=0.7)

        logger.info(f"IntelligentAgent inicializado - Session: {self.conversation.session_id}")

    def process_message(self, user_message: str) -> str:
        """Processa mensagem do usuário e retorna resposta apropriada.

        Args:
            user_message: Mensagem do usuário

        Returns:
            Resposta do agente
        """
        logger.info(f"Processando mensagem: {user_message}")

        # Classifica a pergunta
        question_type = self.router.classify(user_message, self.conversation.history)

        # Roteamento baseado no tipo
        if question_type == "greeting":
            response = self._handle_greeting(user_message)
        elif question_type == "farewell":
            response = self._handle_farewell(user_message)
        elif question_type == "clarification_needed":
            response = self._handle_clarification(user_message)
        elif question_type == "internal_docs":
            response = self._handle_internal_docs(user_message)
        elif question_type == "general_knowledge":
            response = self._handle_general_knowledge(user_message)
        else:
            response = self._handle_clarification(user_message)

        # Adiciona ao histórico
        self.conversation.add_turn(
            user_message=user_message,
            assistant_message=response,
            question_type=question_type
        )

        return response

    def _handle_greeting(self, message: str) -> str:
        """Trata saudações."""
        logger.info("Handler: greeting")

        greetings = [
            "Olá! Como posso ajudá-lo hoje?",
            "Olá, tudo bem! Em que posso auxiliar?",
            "Oi! Estou aqui para ajudar. O que você precisa?",
            "Olá! Seja bem-vindo. Como posso ser útil?"
        ]

        # Usa o modelo para gerar uma saudação natural
        system_prompt = """Você é um assistente cordial e profissional.
Responda à saudação do usuário de forma amigável e ofereça ajuda.
Seja breve e natural."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar saudação: {e}")
            return greetings[0]

    def _handle_farewell(self, message: str) -> str:
        """Trata despedidas."""
        logger.info("Handler: farewell")

        system_prompt = """Você é um assistente cordial e profissional.
Responda à despedida do usuário de forma amigável e profissional.
Deixe claro que está disponível para ajudar no futuro.
Seja breve e natural."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar despedida: {e}")
            return "Disponha! Sempre que precisar, estou aqui para ajudar. Tenha um ótimo dia!"

    def _handle_clarification(self, message: str) -> str:
        """Trata perguntas que precisam de clarificação."""
        logger.info("Handler: clarification_needed")

        # Marca que está aguardando clarificação
        self.conversation.set_awaiting_clarification(topic=message)

        system_prompt = """Você é um assistente que ajuda a clarificar dúvidas.
O usuário fez uma pergunta vaga ou ambígua. Você precisa fazer 1-2 perguntas para entender melhor o que ele precisa.

Especificamente, pergunte:
1. Se ele quer informações gerais sobre o assunto OU
2. Se ele quer saber sobre procedimentos/políticas/normas internas da empresa

Seja cordial e objetivo. Mantenha as perguntas curtas e diretas.

Exemplo:
Usuário: "Quero saber sobre reembolso"
Você: "Claro! O que você gostaria de saber sobre reembolso? Quer que eu forneça informações gerais sobre o assunto ou prefere saber sobre o procedimento específico da empresa?"
"""

        history_context = self.conversation.get_history_summary(last_n=3)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Contexto recente:\n{history_context}\n\nPergunta atual: {message}")
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar clarificação: {e}")
            return "Entendi que você tem uma dúvida, mas preciso de mais detalhes. Você quer informações gerais sobre o assunto ou sobre procedimentos específicos da empresa?"

    def _handle_internal_docs(self, message: str) -> str:
        """Trata perguntas sobre documentos internos."""
        logger.info("Handler: internal_docs")

        # Limpa estado de clarificação
        self.conversation.clear_clarification_state()

        # Primeiro informa que vai buscar
        searching_message = "Perfeito! Vou buscar na base de conhecimento disponível. Aguarde alguns instantes..."
        logger.info(searching_message)

        # Busca documentos relevantes
        search_results = self.document_searcher.search(message, top_k=5)

        # Verifica se encontrou resultados relevantes
        if not self.document_searcher.has_relevant_results(search_results):
            return self._handle_no_relevant_docs(message)

        # Formata contexto com os documentos encontrados
        docs_context = self.document_searcher.format_results_for_context(search_results, max_results=3)

        # Gera resposta usando os documentos
        system_prompt = """Você é um assistente especializado em responder perguntas usando documentos internos da empresa.

INSTRUÇÕES:
1. Use APENAS as informações dos documentos fornecidos para responder
2. Sempre mencione a fonte (Manual, Norma, etc.) quando disponível
3. Seja preciso e objetivo
4. Se os documentos não contiverem informação suficiente, seja honesto sobre isso
5. Estruture a resposta de forma clara e profissional
6. Ao final, pergunte se pode ajudar com mais alguma coisa

FORMATO DA RESPOSTA:
- Inicie mencionando a fonte se disponível
- Apresente as informações de forma estruturada
- Finalize oferecendo ajuda adicional
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"DOCUMENTOS ENCONTRADOS:\n{docs_context}\n\nPERGUNTA DO USUÁRIO:\n{message}")
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar resposta com documentos: {e}")
            return "Encontrei informações relevantes, mas tive um problema ao processar a resposta. Por favor, tente reformular sua pergunta."

    def _handle_no_relevant_docs(self, message: str) -> str:
        """Trata caso onde não encontrou documentos relevantes."""
        logger.info("Nenhum documento relevante encontrado")

        system_prompt = """Você é um assistente que informa ao usuário que não encontrou documentos relevantes.

INSTRUÇÕES:
1. Informe educadamente que não encontrou informações suficientes na base de conhecimento
2. Ofereça responder com conhecimento geral caso ele queira
3. Seja cordial e útil
4. Mantenha a resposta curta

Exemplo:
"Não encontrei informações específicas sobre isso em nossa base de documentos internos. Posso fornecer uma explicação geral sobre o assunto, se desejar. O que você prefere?"
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Pergunta do usuário: {message}")
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar resposta sem documentos: {e}")
            return "Não encontrei informações suficientes sobre isso em nossa base de conhecimento. Posso lhe fornecer uma explicação geral sobre o assunto, se desejar. O que você prefere?"

    def _handle_general_knowledge(self, message: str) -> str:
        """Trata perguntas de conhecimento geral."""
        logger.info("Handler: general_knowledge")

        # Limpa estado de clarificação
        self.conversation.clear_clarification_state()

        system_prompt = """Você é um assistente útil e conhecedor.
Responda à pergunta do usuário usando seu conhecimento geral.

INSTRUÇÕES:
1. Forneça uma resposta clara, precisa e educativa
2. Use exemplos quando apropriado
3. Seja objetivo mas completo
4. Ao final, pergunte se pode ajudar com mais alguma coisa
5. Mantenha um tom profissional e amigável
"""

        history_context = self.conversation.get_history_summary(last_n=3)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Contexto recente:\n{history_context}\n\nPergunta: {message}")
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erro ao gerar resposta de conhecimento geral: {e}")
            return "Desculpe, tive um problema ao processar sua pergunta. Pode tentar reformulá-la?"

    def get_conversation_context(self) -> Dict[str, Any]:
        """Retorna informações sobre o contexto da conversa."""
        return self.conversation.get_context_info()
