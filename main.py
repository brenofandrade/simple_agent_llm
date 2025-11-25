"""Entry point for the Intelligent Agent with Question Routing.

This script creates a Flask API that routes user questions intelligently
based on their type: greetings, clarifications, internal docs, or general knowledge.
"""

from __future__ import annotations

import argparse
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from agent import IntelligentAgent
from config import logger

# Flask App Setup
app = Flask(__name__)
CORS(app,
     resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)

# Store agents by session
agents: Dict[str, IntelligentAgent] = {}


def get_or_create_agent(session_id: str = None) -> IntelligentAgent:
    """Obtém ou cria um agente para a sessão."""
    if session_id and session_id in agents:
        return agents[session_id]

    agent = IntelligentAgent(session_id)
    agents[agent.conversation.session_id] = agent
    return agent


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "intelligent-agent",
        "active_sessions": len(agents)
    }), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal para conversar com o agente.

    Request body:
        {
            "message": "user message",
            "session_id": "optional session id"
        }

    Response:
        {
            "response": "agent response",
            "session_id": "session identifier",
            "question_type": "type of question",
            "context": {...}
        }
    """
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                "error": "Campo 'message' é obrigatório"
            }), 400

        user_message = data['message']
        session_id = data.get('session_id')

        # Obtém ou cria agente para a sessão
        agent = get_or_create_agent(session_id)

        # Processa mensagem
        response = agent.process_message(user_message)

        # Obtém contexto da conversa
        context = agent.get_conversation_context()

        return jsonify({
            "response": response,
            "session_id": agent.conversation.session_id,
            "question_type": agent.conversation.get_last_turn().question_type,
            "context": context
        }), 200

    except Exception as e:
        logger.error(f"Erro no endpoint /chat: {e}", exc_info=True)
        return jsonify({
            "error": "Erro ao processar mensagem",
            "details": str(e)
        }), 500


@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Obtém informações sobre uma sessão específica."""
    if session_id not in agents:
        return jsonify({
            "error": "Sessão não encontrada"
        }), 404

    agent = agents[session_id]
    context = agent.get_conversation_context()

    return jsonify({
        "session_id": session_id,
        "context": context,
        "history_length": len(agent.conversation.history)
    }), 200


@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Deleta uma sessão."""
    if session_id in agents:
        del agents[session_id]
        return jsonify({
            "message": "Sessão deletada com sucesso"
        }), 200

    return jsonify({
        "error": "Sessão não encontrada"
    }), 404


@app.route('/sessions', methods=['GET'])
def list_sessions():
    """Lista todas as sessões ativas."""
    sessions = []
    for session_id, agent in agents.items():
        context = agent.get_conversation_context()
        sessions.append({
            "session_id": session_id,
            "turn_count": context["turn_count"],
            "awaiting_clarification": context["awaiting_clarification"]
        })

    return jsonify({
        "sessions": sessions,
        "total": len(sessions)
    }), 200


def run_cli(question: str) -> None:
    """Executa o agente em modo CLI para testes."""
    agent = IntelligentAgent()

    print("\n" + "=" * 80)
    print("INTELLIGENT AGENT - Modo CLI")
    print("=" * 80)
    print(f"Session ID: {agent.conversation.session_id}")
    print("=" * 80 + "\n")

    response = agent.process_message(question)

    print(f"Usuário: {question}")
    print(f"Agente: {response}\n")

    context = agent.get_conversation_context()
    print(f"Tipo de pergunta: {context}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Intelligent Agent.")
    parser.add_argument(
        "question",
        nargs="?",
        help="Pergunta para testar o agente em modo CLI.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Inicia o servidor Flask API."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Porta para o servidor Flask (padrão: 5000)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.api:
        logger.info(f"Iniciando servidor Flask na porta {args.port}...")
        app.run(host="0.0.0.0", port=args.port, debug=True)
    elif args.question:
        run_cli(args.question)
    else:
        # Modo interativo
        logger.info("Modo interativo - Iniciando servidor Flask na porta 5000...")
        app.run(host="0.0.0.0", port=5000, debug=True)
