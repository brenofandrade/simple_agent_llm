"""Script de teste para o Intelligent Agent."""

from __future__ import annotations
from agent import IntelligentAgent
from config import logger
import time


def print_separator(title: str = ""):
    """Imprime um separador visual."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def test_conversation_flow():
    """Testa um fluxo completo de conversa."""
    print_separator("TESTE: Fluxo Completo de Conversa")

    agent = IntelligentAgent()
    print(f"Session ID: {agent.conversation.session_id}\n")

    # Simulação do fluxo de exemplo
    test_messages = [
        "Olá",
        "Queria saber sobre reembolso.",
        "Eu queria saber como a Unimed realiza reembolso para os colaboradores quando estão em viagem a trabalho ou evento.",
        "Não, obrigado!"
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n[Turno {i}]")
        print(f"👤 Usuário: {message}")
        print("-" * 80)

        response = agent.process_message(message)

        print(f"🤖 Agente: {response}")

        # Mostra informações de contexto
        context = agent.get_conversation_context()
        last_turn = agent.conversation.get_last_turn()
        print(f"\n📊 Tipo: {last_turn.question_type}")

        if context["awaiting_clarification"]:
            print(f"⏳ Aguardando clarificação sobre: {context['last_topic']}")

        time.sleep(1)  # Pausa para simular leitura


def test_greeting_types():
    """Testa diferentes tipos de saudações."""
    print_separator("TESTE: Tipos de Saudação")

    agent = IntelligentAgent()

    greetings = [
        "Olá",
        "Oi",
        "Bom dia",
        "Tudo bem?",
        "E aí!"
    ]

    for greeting in greetings:
        print(f"\n👤 Usuário: {greeting}")
        response = agent.process_message(greeting)
        print(f"🤖 Agente: {response}")
        time.sleep(0.5)


def test_clarification():
    """Testa perguntas que precisam de clarificação."""
    print_separator("TESTE: Perguntas Vagas (Clarificação)")

    agent = IntelligentAgent()

    vague_questions = [
        "Me fala sobre férias",
        "Quero saber sobre horas extras",
        "Como funciona benefícios?",
        "Preciso de ajuda com documentos"
    ]

    for question in vague_questions:
        print(f"\n👤 Usuário: {question}")
        response = agent.process_message(question)
        print(f"🤖 Agente: {response}")

        context = agent.get_conversation_context()
        if context["awaiting_clarification"]:
            print(f"✅ Clarificação solicitada corretamente")
        else:
            print(f"❌ Deveria ter solicitado clarificação")

        time.sleep(0.5)


def test_general_knowledge():
    """Testa perguntas de conhecimento geral."""
    print_separator("TESTE: Conhecimento Geral")

    agent = IntelligentAgent()

    general_questions = [
        "O que é Python?",
        "Explique o que é inteligência artificial",
        "Como funciona um banco de dados?",
        "O que são férias remuneradas?"
    ]

    for question in general_questions:
        print(f"\n👤 Usuário: {question}")
        response = agent.process_message(question)
        print(f"🤖 Agente: {response[:200]}...")  # Primeiros 200 caracteres
        time.sleep(0.5)


def test_internal_docs():
    """Testa perguntas sobre documentos internos."""
    print_separator("TESTE: Busca em Documentos Internos")

    agent = IntelligentAgent()

    internal_questions = [
        "Como a Unimed realiza reembolso para colaboradores em viagem?",
        "Qual o procedimento da empresa para solicitar férias?",
        "O que diz o manual sobre horas extras?",
        "Quais são as políticas de home office da empresa?"
    ]

    for question in internal_questions:
        print(f"\n👤 Usuário: {question}")
        print("🔍 Buscando em documentos internos...")

        response = agent.process_message(question)

        # Mostra se encontrou documentos
        last_turn = agent.conversation.get_last_turn()
        if "não encontrei" in response.lower() or "não há" in response.lower():
            print("⚠️  Nenhum documento relevante encontrado")
        else:
            print("✅ Documentos encontrados e utilizados")

        print(f"🤖 Agente: {response[:300]}...")  # Primeiros 300 caracteres
        time.sleep(0.5)


def test_farewell():
    """Testa despedidas."""
    print_separator("TESTE: Despedidas")

    agent = IntelligentAgent()

    farewells = [
        "Obrigado, é só isso!",
        "Tchau",
        "Até logo",
        "Valeu, até mais!"
    ]

    for farewell in farewells:
        print(f"\n👤 Usuário: {farewell}")
        response = agent.process_message(farewell)
        print(f"🤖 Agente: {response}")
        time.sleep(0.5)


def test_session_management():
    """Testa gerenciamento de sessões."""
    print_separator("TESTE: Gerenciamento de Sessões")

    # Cria múltiplas sessões
    agent1 = IntelligentAgent()
    agent2 = IntelligentAgent()

    print(f"Agente 1 - Session: {agent1.conversation.session_id}")
    print(f"Agente 2 - Session: {agent2.conversation.session_id}")

    # Testa conversas independentes
    print("\n--- Conversa Agente 1 ---")
    agent1.process_message("Olá")
    agent1.process_message("Me fala sobre Python")

    print("\n--- Conversa Agente 2 ---")
    agent2.process_message("Oi")
    agent2.process_message("Preciso de ajuda com reembolso")

    # Verifica contextos independentes
    ctx1 = agent1.get_conversation_context()
    ctx2 = agent2.get_conversation_context()

    print(f"\nAgente 1: {ctx1['turn_count']} turnos")
    print(f"Agente 2: {ctx2['turn_count']} turnos")
    print("✅ Sessões independentes funcionando corretamente")


def run_all_tests():
    """Executa todos os testes."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "INTELLIGENT AGENT - TESTES" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")

    tests = [
        ("Fluxo Completo", test_conversation_flow),
        ("Saudações", test_greeting_types),
        ("Clarificação", test_clarification),
        ("Conhecimento Geral", test_general_knowledge),
        ("Documentos Internos", test_internal_docs),
        ("Despedidas", test_farewell),
        ("Gerenciamento de Sessões", test_session_management)
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n\n🧪 Executando teste: {test_name}")
            test_func()
            print(f"\n✅ Teste '{test_name}' concluído")
        except Exception as e:
            print(f"\n❌ Erro no teste '{test_name}': {e}")
            logger.error(f"Erro no teste {test_name}", exc_info=True)

        time.sleep(2)  # Pausa entre testes

    print_separator("TESTES CONCLUÍDOS")


def interactive_mode():
    """Modo interativo para testar o agente."""
    print_separator("MODO INTERATIVO")
    print("Digite 'sair' para encerrar\n")

    agent = IntelligentAgent()
    print(f"Session ID: {agent.conversation.session_id}\n")

    while True:
        try:
            user_input = input("👤 Você: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\n👋 Encerrando sessão...")
                break

            response = agent.process_message(user_input)
            print(f"\n🤖 Agente: {response}\n")

            # Mostra contexto se solicitado
            if user_input.lower() == "contexto":
                context = agent.get_conversation_context()
                print(f"\n📊 Contexto: {context}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Sessão interrompida pelo usuário")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")
            logger.error("Erro no modo interativo", exc_info=True)


if __name__ == "__main__":
    import sys

    print("\n🤖 Intelligent Agent - Sistema de Testes")
    print("=" * 80)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_all_tests()
        elif sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_mode()
        elif sys.argv[1] == "--flow":
            test_conversation_flow()
        elif sys.argv[1] == "--greeting":
            test_greeting_types()
        elif sys.argv[1] == "--clarification":
            test_clarification()
        elif sys.argv[1] == "--general":
            test_general_knowledge()
        elif sys.argv[1] == "--docs":
            test_internal_docs()
        elif sys.argv[1] == "--farewell":
            test_farewell()
        elif sys.argv[1] == "--session":
            test_session_management()
        else:
            print(f"Comando desconhecido: {sys.argv[1]}")
            print("\nComandos disponíveis:")
            print("  --all            Executa todos os testes")
            print("  --interactive    Modo interativo")
            print("  --flow           Teste de fluxo completo")
            print("  --greeting       Teste de saudações")
            print("  --clarification  Teste de clarificação")
            print("  --general        Teste de conhecimento geral")
            print("  --docs           Teste de documentos internos")
            print("  --farewell       Teste de despedidas")
            print("  --session        Teste de gerenciamento de sessões")
    else:
        print("\nUso: python test_agent.py [COMANDO]")
        print("\nComandos disponíveis:")
        print("  --all            Executa todos os testes")
        print("  --interactive    Modo interativo (-i)")
        print("  --flow           Teste de fluxo completo")
        print("  --greeting       Teste de saudações")
        print("  --clarification  Teste de clarificação")
        print("  --general        Teste de conhecimento geral")
        print("  --docs           Teste de documentos internos")
        print("  --farewell       Teste de despedidas")
        print("  --session        Teste de gerenciamento de sessões")
        print("\nIniciando modo interativo por padrão...\n")
        interactive_mode()
