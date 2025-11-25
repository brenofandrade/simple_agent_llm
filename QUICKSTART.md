# 🚀 Quick Start - Intelligent Agent

Guia rápido para começar a usar o sistema de roteamento inteligente de perguntas.

## 📦 Instalação

### 1. Clone o repositório (se necessário)
```bash
git clone <repository-url>
cd simple_agent_llm
```

### 2. Configure o ambiente Python
```bash
# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env com suas chaves
nano .env  # ou use seu editor preferido
```

Preencha com suas chaves:
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY_DSUNIBLU=pcsk_...
PINECONE_INDEX_NAME=seu-indice
LOG_LEVEL=INFO
```

## ▶️ Executando o Sistema

### Modo 1: API REST (Recomendado para produção)

```bash
# Inicia o servidor Flask na porta padrão (5000)
python main.py --api

# Ou especifique uma porta diferente
python main.py --api --port 8080
```

O servidor estará disponível em `http://localhost:5000`

### Modo 2: CLI (Testes rápidos)

```bash
# Teste com uma pergunta específica
python main.py "Olá, como você está?"

# Ou simplesmente execute para modo interativo
python main.py
```

### Modo 3: Script de Testes

```bash
# Modo interativo de testes
python test_agent.py --interactive

# Executa todos os testes automatizados
python test_agent.py --all

# Testes específicos
python test_agent.py --flow           # Fluxo completo
python test_agent.py --greeting       # Saudações
python test_agent.py --clarification  # Clarificações
python test_agent.py --docs          # Documentos internos
```

## 📡 Usando a API

### Enviar uma mensagem

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Olá, preciso de ajuda"
  }'
```

### Com sessão específica

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Como funciona reembolso?",
    "session_id": "session_123"
  }'
```

### Listar sessões ativas

```bash
curl http://localhost:5000/sessions
```

### Verificar saúde do serviço

```bash
curl http://localhost:5000/health
```

## 🧪 Testando o Sistema

### Teste Rápido no Terminal

```python
# Execute o Python interativo
python

# Importe e teste o agente
from agent import IntelligentAgent

agent = IntelligentAgent()
response = agent.process_message("Olá")
print(response)
```

### Teste com curl (API)

```bash
# 1. Inicie o servidor
python main.py --api

# 2. Em outro terminal, teste:

# Saudação
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá"}'

# Pergunta vaga (deve pedir clarificação)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quero saber sobre reembolso"}'

# Pergunta sobre documentos internos
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como a Unimed faz reembolso para viagens?"}'

# Conhecimento geral
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "O que é Python?"}'
```

## 🎯 Exemplos de Uso

### Exemplo 1: Conversação Simples (CLI)

```bash
$ python main.py "Olá"

================================================================================
INTELLIGENT AGENT - Modo CLI
================================================================================
Session ID: session_20250125_143022
================================================================================

Usuário: Olá
Agente: Olá! Como posso ajudá-lo hoje?

Tipo de pergunta: {'session_id': 'session_20250125_143022', 'turn_count': 1, ...}
```

### Exemplo 2: Fluxo Completo de Conversa (API)

```bash
# 1. Saudação
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá"}' | jq

# Resposta:
{
  "response": "Olá! Como posso ajudá-lo hoje?",
  "session_id": "session_20250125_143022",
  "question_type": "greeting",
  "context": {...}
}

# 2. Pergunta vaga (mesma sessão)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Quero saber sobre reembolso",
    "session_id": "session_20250125_143022"
  }' | jq

# Resposta:
{
  "response": "O que você gostaria de saber sobre reembolso? ...",
  "session_id": "session_20250125_143022",
  "question_type": "clarification_needed",
  "context": {
    "awaiting_clarification": true,
    ...
  }
}
```

### Exemplo 3: Teste Interativo

```bash
$ python test_agent.py --interactive

================================================================================
  MODO INTERATIVO
================================================================================
Digite 'sair' para encerrar

Session ID: session_20250125_143022

👤 Você: Olá
🤖 Agente: Olá! Como posso ajudá-lo hoje?

👤 Você: O que é Python?
🤖 Agente: Python é uma linguagem de programação de alto nível...

👤 Você: sair
👋 Encerrando sessão...
```

## 🔍 Verificando Logs

Os logs são exibidos no console por padrão:

```bash
2025-01-25 14:30:22 - INFO - ✓ Configurações validadas com sucesso
2025-01-25 14:30:22 - INFO -   - Pinecone Index: nome-do-indice
2025-01-25 14:30:23 - INFO - IntelligentAgent inicializado - Session: session_20250125_143022
2025-01-25 14:30:25 - INFO - Processando mensagem: Olá
2025-01-25 14:30:26 - INFO - Pergunta classificada como: greeting
2025-01-25 14:30:26 - INFO - Handler: greeting
```

Para alterar o nível de log, edite o `.env`:
```env
LOG_LEVEL=DEBUG  # Para mais detalhes
LOG_LEVEL=ERROR  # Para menos verbosidade
```

## ❗ Troubleshooting

### Erro: "OPENAI_API_KEY is not configured"
**Solução:** Verifique se o arquivo `.env` existe e contém a chave da OpenAI.

```bash
cat .env | grep OPENAI_API_KEY
```

### Erro: "PINECONE_API_KEY não configurada"
**Solução:** Configure as credenciais do Pinecone no `.env`.

### Erro: "ModuleNotFoundError"
**Solução:** Instale as dependências:
```bash
pip install -r requirements.txt
```

### Erro: "Port already in use"
**Solução:** Use uma porta diferente:
```bash
python main.py --api --port 8080
```

## 📚 Próximos Passos

1. **Leia a documentação completa:** `README_AGENT.md`
2. **Execute os testes automatizados:** `python test_agent.py --all`
3. **Explore a API:** Use Postman ou Insomnia para testar endpoints
4. **Customize os prompts:** Edite `agent.py` para ajustar as respostas
5. **Adicione documentos:** Configure seu índice Pinecone com documentos internos

## 🆘 Precisa de Ajuda?

- **Documentação completa:** `README_AGENT.md`
- **Exemplos de código:** `test_agent.py`
- **Configurações:** `config.py`

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique os logs do sistema
2. Consulte a documentação
3. Execute os testes para identificar problemas

---

**Pronto para começar!** 🎉

Execute `python main.py --api` para iniciar o servidor ou `python test_agent.py -i` para modo interativo.
