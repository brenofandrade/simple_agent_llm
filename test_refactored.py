"""Script de teste para a aplicação refatorada."""

import sys

print("=" * 60)
print("Testando importações dos módulos refatorados...")
print("=" * 60)

try:
    import config
    print("✓ config.py importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar config.py: {e}")
    sys.exit(1)

try:
    import router
    print("✓ router.py importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar router.py: {e}")
    sys.exit(1)

try:
    import tools
    print("✓ tools.py importado com sucesso (Pinecone conectado)")
except Exception as e:
    print(f"✗ Erro ao importar tools.py: {e}")
    sys.exit(1)

try:
    import agent
    print("✓ agent.py importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar agent.py: {e}")
    sys.exit(1)

try:
    from main import app
    print("✓ main.py (Flask app) importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar main.py: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TODOS OS MÓDULOS IMPORTADOS COM SUCESSO!")
print("=" * 60)
print("\nArquitetura:")
print("1. router.py - Classificador de perguntas (OpenAI)")
print("2. tools.py - Pinecone Search Tool (busca híbrida)")
print("3. agent.py - Agente inteligente que usa tools")
print("4. main.py - Flask API na porta 8000")
print("\nPara iniciar:")
print("  python main.py")
print("\nPara testar:")
print("  curl -X POST http://localhost:8000/chat \\")
print("    -H 'Content-Type: application/json' \\")
print("    -H 'X-Session-Id: test123' \\")
print("    -d '{\"question\": \"Olá!\"}'")
print("=" * 60)
