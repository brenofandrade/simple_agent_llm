"""Entry point for running a simple LangChain agent example.

The script wires together a chat model with a basic search tool backed by
static documents, then invokes the agent with a provided question.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from config import openai_api_key
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from flask import Flask, request, jsonify
from flask_cors import CORS 







@tool
def search_documents(query: str) -> List[Document]:
    """Search static documents for content related to the query."""

    query_lower = query.lower()
    matches = [
        document
        for document in STATIC_DOCUMENTS
        if query_lower in document.page_content.lower()
    ]
    return matches or STATIC_DOCUMENTS


def build_model() -> ChatOpenAI:
    """Create a chat model instance with basic configuration."""

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        timeout=30,
    )


def run_agent(question: str) -> None:
    """Create an agent, invoke it with the question and print the response."""

    model = build_model()
    agent = create_agent(model, [search_documents])

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample LangChain agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="O que é feriado religioso?",
        help="Pergunta a ser enviada ao agente.",
    )
    return parser.parse_args()



app = Flask(__name__)
CORS(app, 
     resources={r"/*":{"origins":"*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)



if __name__ == "__main__":
    args = parse_args()
    run_agent(args.question)
