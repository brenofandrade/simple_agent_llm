"""API Flask - Chat com roteamento inteligente e busca híbrida."""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from agent import IntelligentAgent
from config import logger

# --- Configurações ---
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
TTL_SETUP = int(os.getenv("TTL_SETUP", "1200"))
PORT = int(os.getenv("BACKEND_PORT", "8000"))
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# --- Memória em RAM ---
memory_store: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    """Retorna timestamp atual."""
    return time.time()


def _is_expired(entry: Dict[str, Any]) -> bool:
    """Verifica se entrada expirou."""
    return bool(entry) and entry.get("expires_at", 0) <= _now()


def _ensure_entry(session_id: str, ttl: int = TTL_SETUP) -> Dict[str, Any]:
    """Garante que entrada existe e não expirou."""
    if session_id not in memory_store or _is_expired(memory_store.get(session_id, {})):
        memory_store[session_id] = {
            "messages": [],
            "expires_at": _now() + ttl
        }
    return memory_store[session_id]


def load_memory(session_id: str) -> List[Any]:
    """Carrega histórico de mensagens."""
    entry = memory_store.get(session_id)
    if not entry:
        return []
    if _is_expired(entry):
        memory_store.pop(session_id, None)
        return []
    return list(entry.get("messages", []))


def update_memory(session_id: str, messages: List[Any]) -> List[Any]:
    """Atualiza memória com truncamento."""
    truncated = list(messages[-MAX_HISTORY:])
    memory_store[session_id] = {
        "messages": truncated,
        "expires_at": _now() + TTL_SETUP
    }
    return truncated


def clear_memory(session_id: str) -> bool:
    """Limpa memória de uma sessão."""
    return memory_store.pop(session_id, None) is not None


# --- Flask App ---
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)
app.url_map.strict_slashes = False


# --- Cache de Agentes por Sessão ---
agents_cache: Dict[str, IntelligentAgent] = {}


def get_agent(session_id: str) -> IntelligentAgent:
    """Retorna agente para sessão (com cache)."""
    if session_id not in agents_cache:
        agents_cache[session_id] = IntelligentAgent(
            session_id=session_id,
            use_openai_for_generation=False  # Usa Ollama por padrão
        )
        logger.info(f"Novo agente criado para sessão: {session_id}")
    return agents_cache[session_id]


# --- Endpoints ---

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de health check."""
    return jsonify({"status": "ok", "service": "chat-agent"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint principal de chat.

    Fluxo:
    1. Recebe pergunta do usuário
    2. Roteador classifica a pergunta (greetings, farewell, clarification_needed, internal_docs, general_knowledge)
    3. Agente decide se usa tool do Pinecone (quando internal_docs)
    4. Retorna resposta + metadados
    """
    start_time = time.perf_counter()

    try:
        payload = request.get_json(force=True, silent=False) or {}
        question = (payload.get("question") or payload.get("message") or "").strip()
        session_id = request.headers.get("X-Session-Id", "") or payload.get("session_id", "")

        # Validações
        if not session_id:
            return jsonify({"error": "Campo 'session_id' é obrigatório"}), 400
        if not question:
            return jsonify({"error": "Campo 'question' é obrigatório"}), 400

        # Carrega histórico
        history_messages = load_memory(session_id)

        # Parâmetros opcionais
        k = int(payload.get("k", 5))  # Número de documentos
        namespace = payload.get("namespace") or None

        # Obtém agente da sessão
        agent = get_agent(session_id)

        # Processa pergunta (agente decide se usa Pinecone tool)
        result = agent.process_question(
            question=question,
            history=history_messages,
            k=k,
            namespace=namespace
        )

        answer = result["answer"]
        question_type = result["question_type"]
        used_tool = result.get("used_tool", False)
        sources = result.get("sources")

        # Atualiza histórico
        history_messages.append(HumanMessage(content=question))
        history_messages.append(AIMessage(content=answer))
        update_memory(session_id, history_messages)

        # Calcula latência
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"✓ Latência /chat = {latency:.2f} ms")

        # Resposta
        response_body = {
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "used_tool": used_tool,
            "sources": sources,
            "latency_ms": round(latency, 2)
        }

        # Adiciona metadados se usou tool
        if used_tool and sources:
            response_body["tool_info"] = {
                "tool_name": result.get("tool_name", "pinecone_search"),
                "num_docs_found": result.get("num_docs_found", 0),
                "k": k,
                "namespace": namespace or "default"
            }

        return jsonify(response_body), 200

    except Exception as e:
        logger.exception("Erro no endpoint /chat")
        return jsonify({
            "error": "Erro interno ao processar a solicitação",
            "detail": str(e)
        }), 500


@app.route("/clear", methods=["POST"])
def clear():
    """Limpa memória de uma sessão."""
    try:
        session_id = request.headers.get("X-Session-Id")
        if not session_id:
            payload = request.get_json(silent=True) or {}
            session_id = payload.get("session_id")

        if not session_id:
            return jsonify({
                "error": "Informe o session_id no header X-Session-Id ou no corpo JSON"
            }), 400

        # Limpa memória
        removed = clear_memory(session_id)

        # Remove agente do cache
        if session_id in agents_cache:
            del agents_cache[session_id]
            logger.info(f"Agente removido do cache: {session_id}")

        if removed:
            return jsonify({
                "status": "ok",
                "message": f"Memória da sessão {session_id} foi limpa"
            }), 200

        return jsonify({
            "status": "not_found",
            "message": f"Nenhuma memória encontrada para {session_id}"
        }), 404

    except Exception as e:
        logger.exception("Erro no endpoint /clear")
        return jsonify({
            "error": "Erro interno ao limpar memória",
            "detail": str(e)
        }), 500


@app.route("/", methods=["GET"])
def index():
    """Endpoint raiz."""
    return jsonify({
        "service": "chat-agent-api",
        "version": "2.0",
        "endpoints": {
            "/health": "Health check",
            "/chat": "Chat com classificação e busca híbrida",
            "/clear": "Limpar memória de sessão"
        }
    }), 200


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Iniciando Chat Agent API")
    logger.info(f"   Porta: {PORT}")
    logger.info(f"   Debug: {DEBUG}")
    logger.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG
    )
