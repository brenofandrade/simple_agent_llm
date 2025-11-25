# Intelligent Agent - Sistema de Roteamento de Perguntas

Sistema inteligente que roteia perguntas do usuário baseado no tipo de pergunta, oferecendo respostas contextualizadas e apropriadas.

## 🎯 Funcionalidades

O agente classifica automaticamente as perguntas em 5 categorias:

1. **Saudações (greeting)**: Respostas cordiais e amigáveis
2. **Despedidas (farewell)**: Respostas de encerramento profissionais
3. **Clarificação (clarification_needed)**: Perguntas para entender melhor a intenção
4. **Documentos Internos (internal_docs)**: Busca em base de conhecimento via Pinecone
5. **Conhecimento Geral (general_knowledge)**: Respostas usando conhecimento do modelo

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (Flask API + CLI)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       agent.py                               │
│              (IntelligentAgent - Orquestrador)              │
└───┬──────────────┬──────────────┬──────────────────────┬────┘
    │              │              │                      │
    ▼              ▼              ▼                      ▼
┌─────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────────┐
│question │  │conversa  │  │document    │  │config.py        │
│_router  │  │tion_mgr  │  │_search     │  │(Configurações)  │
└─────────┘  └──────────┘  └────────────┘  └─────────────────┘
```

### Componentes

#### 1. **question_router.py**
- Classifica perguntas usando GPT-4o-mini
- Identifica intenção e contexto
- Retorna tipo de pergunta

#### 2. **conversation_manager.py**
- Gerencia histórico de conversas
- Mantém contexto entre mensagens
- Rastreia estado de clarificação

#### 3. **document_search.py**
- Integração com Pinecone
- Busca semântica em documentos
- Filtragem por relevância (threshold: 0.7)

#### 4. **agent.py**
- Orquestra todos os componentes
- Handlers específicos por tipo de pergunta
- Geração de respostas contextualizadas

#### 5. **main.py**
- API Flask com endpoints REST
- Modo CLI para testes
- Gerenciamento de sessões

## 📋 Fluxo de Interação

```
1. Usuário envia mensagem
         ↓
2. QuestionRouter classifica a pergunta
         ↓
3. Agent roteia para handler apropriado
         ↓
4. Handler processa e gera resposta
         ↓
5. ConversationManager salva no histórico
         ↓
6. Resposta retornada ao usuário
```

## 🚀 Como Usar

### Modo API (Recomendado)

```bash
# Iniciar servidor Flask
python main.py --api

# Ou especificar porta
python main.py --api --port 8080
```

### Endpoints Disponíveis

#### POST /chat
Enviar mensagem para o agente

**Request:**
```json
{
    "message": "Olá",
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "response": "Olá! Como posso ajudá-lo hoje?",
    "session_id": "session_20250125_143022",
    "question_type": "greeting",
    "context": {
        "turn_count": 1,
        "awaiting_clarification": false
    }
}
```

#### GET /health
Status do serviço

#### GET /sessions
Lista todas as sessões ativas

#### GET /session/<session_id>
Detalhes de uma sessão específica

#### DELETE /session/<session_id>
Remove uma sessão

### Modo CLI (Testes Rápidos)

```bash
# Testar pergunta específica
python main.py "Olá"
python main.py "Como fazer reembolso na Unimed?"
python main.py "O que é Python?"
```

## 📝 Exemplos de Uso

### Exemplo 1: Saudação
```
Usuário: Olá
Agente: Olá! Como posso ajudá-lo hoje?
```

### Exemplo 2: Clarificação
```
Usuário: Quero saber sobre reembolso
Agente: O que você gostaria de saber sobre reembolso?
        Quer informações gerais ou procedimentos da empresa?
```

### Exemplo 3: Documentos Internos
```
Usuário: Como a Unimed faz reembolso para colaboradores em viagem?
Agente: Vou buscar na base de conhecimento. Aguarde...

        Segundo o Manual 297 - Políticas de Reembolso:
        [Informações do documento]

        Posso ajudar com mais alguma coisa?
```

### Exemplo 4: Conhecimento Geral
```
Usuário: O que é reembolso?
Agente: Reembolso é o processo de devolução de valores...
        [Explicação conceitual]

        Posso ajudar com mais alguma coisa?
```

## 🔧 Configuração

### Variáveis de Ambiente (.env)

```env
# OpenAI
OPENAI_API_KEY=sua-chave-aqui

# Pinecone
PINECONE_API_KEY_DSUNIBLU=sua-chave-pinecone
PINECONE_INDEX_NAME=nome-do-indice

# Logging
LOG_LEVEL=INFO
```

### Dependências

```bash
pip install -r requirements.txt
```

Principais dependências:
- `langchain-openai`: Integração com OpenAI
- `pinecone-client`: Busca vetorial
- `flask`: API REST
- `flask-cors`: CORS support
- `python-dotenv`: Gerenciamento de variáveis de ambiente

## 🎨 Personalização

### Ajustar Threshold de Relevância

Em `document_search.py`:
```python
searcher = DocumentSearcher(relevance_threshold=0.75)  # Padrão: 0.7
```

### Modificar Prompts

Os prompts do sistema estão em `agent.py` nos métodos:
- `_handle_greeting()`
- `_handle_farewell()`
- `_handle_clarification()`
- `_handle_internal_docs()`
- `_handle_general_knowledge()`

### Adicionar Novos Tipos de Pergunta

1. Adicione o tipo em `question_router.py` no enum `QuestionType`
2. Atualize o prompt de classificação
3. Crie um novo handler em `agent.py`
4. Adicione no roteamento do `process_message()`

## 🧪 Testes

### Testar Classificação

```python
from question_router import QuestionRouter
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
router = QuestionRouter(model)

print(router.classify("Olá"))  # greeting
print(router.classify("O que é Python?"))  # general_knowledge
print(router.classify("Qual o procedimento da empresa para férias?"))  # internal_docs
```

### Testar Busca em Documentos

```python
from document_search import DocumentSearcher

searcher = DocumentSearcher()
results = searcher.search("reembolso viagem colaboradores")

for result in results:
    print(f"Score: {result.score:.2%}")
    print(f"Fonte: {result.formatted_source}")
    print(f"Conteúdo: {result.content[:200]}...")
```

## 📊 Logs

O sistema registra todas as operações importantes:

```
2025-01-25 14:30:22 - INFO - IntelligentAgent inicializado - Session: session_20250125_143022
2025-01-25 14:30:25 - INFO - Processando mensagem: Olá
2025-01-25 14:30:26 - INFO - Pergunta classificada como: greeting
2025-01-25 14:30:26 - INFO - Handler: greeting
```

Configurar nível de log em `.env`:
```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 🔒 Segurança

- Todas as chaves de API devem estar em `.env`
- Nunca commitar o arquivo `.env`
- Validação de configurações em `config.py`
- CORS configurado para aceitar todas origens (ajustar em produção)

## 🚧 Melhorias Futuras

- [ ] Suporte a múltiplos idiomas
- [ ] Cache de respostas frequentes
- [ ] Análise de sentimento
- [ ] Feedback de qualidade das respostas
- [ ] Dashboard de métricas
- [ ] Testes unitários e de integração
- [ ] Rate limiting na API
- [ ] Autenticação JWT

## 📞 Suporte

Para dúvidas ou problemas, verifique:
1. Logs do sistema
2. Configurações no `.env`
3. Conexão com Pinecone
4. Créditos da API OpenAI

## 📄 Licença

Este projeto é parte do sistema interno e deve seguir as políticas de uso da organização.
