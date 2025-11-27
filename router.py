import json
import re
from typing import Dict

from config import openai_api_key
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
llm_router = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    timeout=15,
    api_key=SecretStr(openai_api_key) if openai_api_key else None,
)

ALLOWED_CLASSES = {
    "clarify",
    "general_knowledge",
    "internal_docs_with_rag",
    "greeting",
    "farewell",
}

FALLBACK_RESULT = {
    "classification": "clarify",
    "message": "Só para entender melhor: você quer saber isso de forma geral ou especificamente nas regras e documentos da Unimed?",
}


def route_question(query: str) -> Dict[str, str]:
    prompt = f"""
        Você é um roteador de intenções para um chatbot da Unimed.

        Sua tarefa é:
        1) Ler a mensagem do usuário.
        2) Classificar a intenção em EXATAMENTE UMA das classes:
        - clarify
        - general_knowledge
        - internal_docs_with_rag
        - greeting
        - farewell
        3) Gerar uma mensagem CURTA e NATURAL em português (pt-BR), que ajude a manter a
        conversa fluida e interativa.

        Definições das classes:

        - clarify
        Use quando:
            * a pergunta for ambígua entre conhecimento geral e contexto interno da Unimed, OU
            * faltar informação para entender o que o usuário quer, OU
            * você não tiver certeza se deve usar conhecimento geral ou documentos internos.
        A mensagem deve pedir esclarecimento de forma amigável.
        Ex.: "Você quer saber isso de forma geral ou especificamente nas regras e documentos da Unimed?"

        - general_knowledge
        Use quando:
            * a pergunta puder ser respondida apenas com conhecimento geral de mundo,
            * e não houver indicação clara de que o usuário quer algo específico da Unimed.

        - internal_docs_with_rag
        Use quando:
            * a pergunta estiver claramente relacionada a informações internas da Unimed
            (benefícios, planos, regras, processos, políticas, documentos internos, etc.),
            * ou ficar evidente que o usuário busca informações específicas da empresa.

        - greeting
        Use quando:
            * a mensagem for principalmente uma saudação
            (ex: "oi", "bom dia", "boa tarde", "olá", "e aí", etc.).

        - farewell
        Use quando:
            * a mensagem for principalmente uma despedida
            (ex: "tchau", "até mais", "obrigado, boa noite", etc.).

        REGRAS IMPORTANTES:
        - Se a pergunta puder ser tanto conhecimento geral quanto interno da empresa e isso for relevante
        (como "O que é sinistralidade?"), prefira a classe "clarify".
        - A resposta DEVE ser um JSON VÁLIDO, sem texto extra, no formato:

        {{
            "classification": "<clarify | general_knowledge | internal_docs_with_rag | greeting | farewell>",
            "message": "<mensagem em português>"
        }}

        - Não inclua comentários, explicações, markdown ou texto fora do JSON.

        Alguns exemplos:

        Pergunta: "O que é sinistralidade?"
        JSON:
        {{
        "classification": "clarify",
        "message": "Você quer saber o conceito de sinistralidade de forma geral ou especificamente aplicado aos planos da Unimed?"
        }}

        Pergunta: "Como são os benefícios da Unimed?"
        JSON:
        {{
        "classification": "internal_docs_with_rag",
        "message": "Posso verificar nos documentos internos da Unimed quais são os benefícios disponíveis. Sobre quais benefícios estaria mais interessado?"
        }}

        Pergunta: "Qual é a capital da França?"
        JSON:
        {{
        "classification": "general_knowledge",
        "message": "Essa é uma pergunta de conhecimento geral, posso responder com base no meu conhecimento de mundo."
        }}

        Pergunta: "Oi, tudo bem?"
        JSON:
        {{
        "classification": "greeting",
        "message": "Oi! Tudo bem, e com você? Como posso te ajudar hoje?"
        }}

        Pergunta: "Obrigado, até mais!"
        JSON:
        {{
        "classification": "farewell",
        "message": "Eu que agradeço! Até mais, se precisar é só chamar."
        }}

        Agora classifique a pergunta a seguir.

        Pergunta do usuário: \"\"\"{query}\"\"\"
        Responda apenas com o JSON solicitado.
    """

    response = llm_router.invoke(prompt)
    raw = response.content

    result = FALLBACK_RESULT.copy()
    if isinstance(raw, str):
        try:
            data = json.loads(raw)

            classification = str(data.get("classification", "")).strip().lower()
            message = str(data.get("message", "")).strip()

            classification = re.sub(r"[^a-z_]", "", classification)

            if classification in ALLOWED_CLASSES:
                result["classification"] = classification

            if message:
                result["message"] = message

        except Exception:
            # Se der erro de parse entra o fallback padrão
            pass

    return result
