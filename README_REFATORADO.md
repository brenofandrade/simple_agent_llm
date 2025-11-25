# Chat Agent API - Arquitetura Refatorada

Sistema de chat inteligente com roteamento automático e busca híbrida no Pinecone.

## 🏗️ Arquitetura

### Fluxo de Processamento

```
Usuário
  ↓
[Flask API - Porta 8000]
  ↓
[Router - Classifica pergunta via OpenAI]
  ↓
├─→ greetings          → Resposta direta
├─→ farewell           → Resposta direta
├─→ clarification      → Pede mais detalhes
├─→ general_knowledge  → Usa LLM diretamente
└─→ internal_docs      → Agente usa Pinecone Tool
                           ↓
                      [Busca Híbrida]
                      - Similaridade vetorial
                      - Reranking com cross-encoder
                           ↓
                      [Resposta com contexto]
```

### Componentes Principais

#### 1. **router.py** - Classificador de Perguntas
- Usa OpenAI GPT-4o-mini
- Classifica em 5 categorias:
  - `greetings` - Saudações
  - `farewell` - Despedidas
  - `clarification_needed` - Perguntas vagas
  - `internal_docs` - Documentos internos
  - `general_knowledge` - Conhecimento geral

#### 2. **tools.py** - Pinecone Search Tool
- **Busca Híbrida**:
  - Gera variações da query
  - Busca por similaridade vetorial
  - Reranking com cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Embeddings**: OpenAI ou Ollama (configurável)
- **Deduplicação** automática de resultados

#### 3. **agent.py** - Agente Inteligente
- Recebe classificação do router
- **Decide autonomamente** quando usar Pinecone tool
- Para `internal_docs`: sempre busca contexto
- Gera respostas contextualizadas

#### 4. **main.py** - Flask API
- Porta: **8000**
- Endpoints:
  - `POST /chat` - Processa perguntas
  - `POST /clear` - Limpa memória
  - `GET /health` - Health check
  - `GET /` - Info da API

## 🚀 Como Usar

### 1. Configurar Ambiente

Crie um arquivo `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY_DSUNIBLU=your-pinecone-key
PINECONE_INDEX=your-index-name
PINECONE_NAMESPACE=default

# Ollama (opcional)
OLLAMA_BASE_URL=http://localhost:11434
GENERATION_MODEL=llama3.2:latest
EMBEDDING_MODEL=mxbai-embed-large:latest

# Flask
BACKEND_PORT=8000
FLASK_DEBUG=0

# Outros
LOG_LEVEL=INFO
MAX_HISTORY=10
RETRIEVAL_K=5
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Iniciar Aplicação

```bash
python main.py
```

A API estará disponível em: `http://localhost:8000`

## 📡 Exemplos de Uso

### 1. Saudação

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user123" \
  -d '{
    "question": "Olá!"
  }'
```

**Resposta:**
```json
{
  "question": "Olá!",
  "answer": "Olá! Como posso ajudá-lo hoje?",
  "question_type": "greetings",
  "used_tool": false,
  "sources": null,
  "latency_ms": 234.56
}
```

### 2. Pergunta sobre Documentos Internos

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user123" \
  -d '{
    "question": "Qual o procedimento de reembolso para viagens?",
    "k": 5,
    "namespace": "default"
  }'
```

**Resposta:**
```json
{
  "question": "Qual o procedimento de reembolso para viagens?",
  "answer": "De acordo com o Manual de Políticas...",
  "question_type": "internal_docs",
  "used_tool": true,
  "sources": [
    {
      "rank": 1,
      "source": "Manual de Políticas (página 15)",
      "score": 0.89,
      "rerank_score": 0.95,
      "content_preview": "Para reembolso de viagens...",
      "metadata": {...}
    }
  ],
  "tool_info": {
    "tool_name": "pinecone_search",
    "num_docs_found": 3,
    "k": 5,
    "namespace": "default"
  },
  "latency_ms": 1523.45
}
```

### 3. Pergunta de Conhecimento Geral

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user123" \
  -d '{
    "question": "O que é machine learning?"
  }'
```

**Resposta:**
```json
{
  "question": "O que é machine learning?",
  "answer": "Machine Learning é um subcampo da inteligência artificial...",
  "question_type": "general_knowledge",
  "used_tool": false,
  "sources": null,
  "latency_ms": 567.89
}
```

### 4. Limpar Memória

```bash
curl -X POST http://localhost:8000/clear \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user123"
```

## 🔧 Parâmetros da API

### POST /chat

**Headers:**
- `X-Session-Id` (obrigatório): ID da sessão do usuário

**Body (JSON):**
- `question` (obrigatório): Pergunta do usuário
- `session_id` (opcional): Alternativa ao header
- `k` (opcional, padrão=5): Número de documentos a buscar
- `namespace` (opcional): Namespace do Pinecone

## ⚙️ Configurações Avançadas

### Trocar Modelo de Geração

No `main.py:89-92`, altere:

```python
IntelligentAgent(
    session_id=session_id,
    use_openai_for_generation=True  # True = OpenAI, False = Ollama
)
```

### Ajustar Reranking

No `agent.py:282-288`, altere o método:

```python
search_results = self.pinecone_tool.search(
    query=question,
    k=k,
    namespace=namespace,
    rerank_method="cross-encoder",  # "none", "embedding", "cross-encoder"
    rerank_top_k=3
)
```

## 🆕 Mudanças Principais

### ❌ Removido
- Flag `use_rag` - O agente decide automaticamente
- Código duplicado de prompts
- ConversationBufferMemory (conflitos de pacotes)

### ✅ Adicionado
- Arquitetura baseada em Tools
- Busca híbrida no Pinecone
- Reranking com cross-encoder
- Cache de agentes por sessão
- Classificação automática com OpenAI

## 📊 Estrutura de Arquivos

```
simple_agent_llm/
├── main.py              # Flask API (porta 8000)
├── agent.py             # Agente inteligente
├── router.py            # Classificador de perguntas
├── tools.py             # Pinecone search tool
├── config.py            # Configurações
├── requirements.txt     # Dependências
└── .env                 # Variáveis de ambiente
```

## 🐛 Debug

Para ativar logs detalhados:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📝 Notas

1. **Pinecone obrigatório**: Para `internal_docs`, o Pinecone deve estar configurado
2. **OpenAI obrigatória**: O router usa GPT-4o-mini para classificação
3. **Ollama opcional**: Pode usar para geração de respostas (economia de custos)
4. **Sessões**: Cada sessão mantém histórico independente (max 10 mensagens)
5. **TTL**: Sessões expiram após 1200s (20 minutos) de inatividade

## 🎯 Próximos Passos

- [ ] Adicionar telemetria e métricas
- [ ] Implementar cache de embeddings
- [ ] Adicionar suporte a múltiplos namespaces
- [ ] Interface web para testes
- [ ] Testes automatizados
