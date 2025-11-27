import json
import time
import re
from typing import Dict

from config import openai_api_key
from router import route_question

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI





def process_payload(payload: dict) -> str:
    """
    Processa a pergunta do usuário a partir do payload.

    Espera um campo "message" no JSON.
    """
    query = (payload.get("message") or "").strip()

    if not query:
        raise ValueError("Campo 'message' é obrigatório e não pode estar vazio.")

    return query


app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=3600,
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Espera JSON:
    {
        "message": "texto da pergunta",
        "session_id": "opcional"
    }
    """

    start_time = time.perf_counter()

    try:
        payload = request.get_json(force=True, silent=False) or {}
        question = process_payload(payload)

        router_result = route_question(question)

        latency = time.perf_counter() - start_time

        return (
            jsonify(
                {
                    "question": question,
                    "classification": router_result["classification"],
                    "router_message": router_result["message"],
                    "latency": latency,
                }
            ),
            200,
        )

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        # se quiser ver o stacktrace no console:
        import traceback

        print("Erro na rota /chat:\n", traceback.format_exc())

        return (
            jsonify(
                {
                    "error": "Erro interno ao processar a solicitação.",
                    "detalhes": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    # garante que está rodando na porta 8000, como o seu cliente está usando
    app.run(host="0.0.0.0", port=8000, debug=True)
