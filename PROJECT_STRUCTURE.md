# 📁 Estrutura do Projeto - Intelligent Agent

## 🌳 Árvore de Arquivos

```
simple_agent_llm/
│
├── 📄 main.py                      # Ponto de entrada (Flask API + CLI)
├── 🤖 agent.py                     # Agente principal com roteamento
├── 🧭 question_router.py           # Classificador de perguntas
├── 💬 conversation_manager.py      # Gerenciador de contexto e histórico
├── 🔍 document_search.py           # Busca em documentos (Pinecone)
├── ⚙️  config.py                    # Configurações e validação
│
├── 🧪 test_agent.py                # Suite completa de testes
│
├── 📚 README_AGENT.md              # Documentação completa
├── 🚀 QUICKSTART.md                # Guia rápido de início
├── 📋 PROJECT_STRUCTURE.md         # Este arquivo
│
├── 📦 requirements.txt             # Dependências Python
├── 🔐 .env.example                 # Exemplo de configuração
├── 🔐 .env                         # Configurações (não commitado)
│
└── 📁 __pycache__/                 # Cache do Python (gerado)
```

## 🔗 Fluxo de Dependências

```
main.py
  └─> agent.py
        ├─> question_router.py
        │     └─> config.py
        ├─> conversation_manager.py
        ├─> document_search.py
        │     └─> config.py
        └─> config.py
```

## 📦 Módulos Principais

### 1️⃣ **main.py** - Entry Point
**Responsabilidades:**
- Inicialização do servidor Flask
- Gerenciamento de rotas REST
- Modo CLI para testes
- Gerenciamento de sessões de usuário

**Endpoints:**
- `POST /chat` - Enviar mensagem
- `GET /health` - Health check
- `GET /sessions` - Listar sessões
- `GET /session/<id>` - Detalhes da sessão
- `DELETE /session/<id>` - Deletar sessão

**Modos de Execução:**
```bash
python main.py --api            # Servidor Flask
python main.py "pergunta"       # CLI teste rápido
python main.py                  # Modo interativo
```

---

### 2️⃣ **agent.py** - Intelligent Agent
**Responsabilidades:**
- Orquestração de todos os componentes
- Roteamento para handlers específicos
- Geração de respostas contextualizadas
- Integração com histórico de conversa

**Handlers:**
- `_handle_greeting()` - Saudações
- `_handle_farewell()` - Despedidas
- `_handle_clarification()` - Perguntas vagas
- `_handle_internal_docs()` - Busca em documentos
- `_handle_general_knowledge()` - Conhecimento geral
- `_handle_no_relevant_docs()` - Fallback sem documentos

**Classe Principal:**
```python
class IntelligentAgent:
    def __init__(self, session_id: str = None)
    def process_message(self, user_message: str) -> str
    def get_conversation_context(self) -> Dict[str, Any]
```

---

### 3️⃣ **question_router.py** - Question Classifier
**Responsabilidades:**
- Classificação inteligente de perguntas
- Análise de intenção do usuário
- Uso de GPT para categorização

**Tipos de Pergunta:**
- `greeting` - Saudações
- `farewell` - Despedidas
- `clarification_needed` - Requer mais informações
- `internal_docs` - Busca em documentos
- `general_knowledge` - Conhecimento geral

**Classe Principal:**
```python
class QuestionRouter:
    def __init__(self, model: ChatOpenAI)
    def classify(self, question: str, conversation_history: list = None) -> QuestionType
```

---

### 4️⃣ **conversation_manager.py** - Context Manager
**Responsabilidades:**
- Manutenção do histórico de conversa
- Gerenciamento de contexto entre mensagens
- Rastreamento de estado de clarificação
- Geração de resumos

**Classes:**
```python
@dataclass
class ConversationTurn:
    timestamp: datetime
    user_message: str
    assistant_message: str
    question_type: str
    metadata: Dict[str, Any]

class ConversationManager:
    def __init__(self, session_id: str = None)
    def add_turn(...)
    def get_last_turn() -> ConversationTurn | None
    def get_history_summary(last_n: int = 5) -> str
    def is_awaiting_clarification() -> bool
    def set_awaiting_clarification(topic: str = None)
    def clear_clarification_state()
    def get_context_info() -> Dict[str, Any]
```

---

### 5️⃣ **document_search.py** - Vector Search
**Responsabilidades:**
- Integração com Pinecone
- Busca semântica em documentos
- Filtragem por relevância
- Formatação de resultados

**Classes:**
```python
@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str = None

    @property
    def formatted_source(self) -> str

class DocumentSearcher:
    def __init__(self, relevance_threshold: float = 0.7)
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]
    def has_relevant_results(self, results: List[SearchResult]) -> bool
    def format_results_for_context(self, results: List[SearchResult], max_results: int = 3) -> str
```

**Configuração:**
- Threshold padrão: 0.7 (70% de relevância)
- Top K: 5 documentos
- Usa OpenAI Embeddings

---

### 6️⃣ **config.py** - Configuration
**Responsabilidades:**
- Carregamento de variáveis de ambiente
- Validação de configurações críticas
- Setup de logging
- Exposição de constantes

**Configurações:**
```python
# OpenAI
openai_api_key: str

# Pinecone
PINECONE_API_KEY: str
PINECONE_INDEX_NAME: str

# Logging
LOG_LEVEL: str
logger: Logger

# Funções
validate_config()  # Valida configurações críticas
```

---

### 7️⃣ **test_agent.py** - Test Suite
**Responsabilidades:**
- Testes automatizados
- Modo interativo
- Validação de fluxos
- Demonstração de funcionalidades

**Testes Disponíveis:**
```python
test_conversation_flow()      # Fluxo completo
test_greeting_types()         # Saudações
test_clarification()          # Clarificações
test_general_knowledge()      # Conhecimento geral
test_internal_docs()          # Documentos internos
test_farewell()               # Despedidas
test_session_management()     # Gerenciamento de sessões
run_all_tests()               # Todos os testes
interactive_mode()            # Modo interativo
```

**Modos de Execução:**
```bash
python test_agent.py --all            # Todos os testes
python test_agent.py --interactive    # Modo interativo
python test_agent.py --flow           # Teste específico
```

---

## 🔄 Fluxo de Execução

### Fluxo Principal (POST /chat)

```
1. Requisição HTTP POST /chat
   ↓
2. Flask recebe e valida dados
   ↓
3. get_or_create_agent(session_id)
   ↓
4. agent.process_message(user_message)
   ├─> 4.1. router.classify(message)
   │        └─> Retorna: QuestionType
   ├─> 4.2. Roteamento para handler apropriado
   │        ├─> _handle_greeting()
   │        ├─> _handle_farewell()
   │        ├─> _handle_clarification()
   │        ├─> _handle_internal_docs()
   │        │     ├─> document_searcher.search()
   │        │     │     ├─> Gera embeddings (OpenAI)
   │        │     │     ├─> Busca no Pinecone
   │        │     │     └─> Filtra por relevância
   │        │     └─> Gera resposta com documentos
   │        └─> _handle_general_knowledge()
   └─> 4.3. conversation.add_turn()
   ↓
5. Retorna resposta JSON
```

### Fluxo de Classificação

```
question_router.classify()
  ↓
1. Prepara prompt de sistema
   ↓
2. Adiciona pergunta do usuário
   ↓
3. Envia para GPT-4o-mini
   ↓
4. Valida resposta
   ↓
5. Retorna QuestionType
```

### Fluxo de Busca em Documentos

```
document_searcher.search()
  ↓
1. embeddings.embed_query(query)
   └─> OpenAI gera embedding
   ↓
2. pinecone_index.query(vector)
   └─> Busca vetores similares
   ↓
3. Filtra por threshold (0.7)
   ↓
4. Formata SearchResults
   ↓
5. Retorna List[SearchResult]
```

---

## 🔌 Integrações Externas

### OpenAI API
- **Usado em:**
  - `question_router.py` - Classificação de perguntas
  - `agent.py` - Geração de respostas
  - `document_search.py` - Embeddings
- **Modelo:** gpt-4o-mini
- **Temperatura:** 0.7

### Pinecone
- **Usado em:**
  - `document_search.py` - Busca vetorial
- **Recursos:**
  - Vector search
  - Metadata filtering
  - Similarity scoring

### Flask
- **Usado em:**
  - `main.py` - API REST
- **Recursos:**
  - Rotas HTTP
  - JSON serialization
  - CORS support

---

## 📊 Dados e Estado

### Estado de Sessão
Cada sessão mantém:
- `session_id`: Identificador único
- `history`: Lista de ConversationTurn
- `context`: Dicionário de contexto
  - `awaiting_clarification`: bool
  - `last_topic`: str
  - `clarification_attempts`: int

### Persistência
- **Sessões:** Em memória (Dict)
- **Histórico:** Em memória durante sessão
- **Documentos:** Pinecone (persistente)

**Nota:** Para persistência permanente, seria necessário:
- Redis/Database para sessões
- Sistema de logging para histórico
- Backup de configurações

---

## 🔒 Segurança

### Variáveis Sensíveis
- `OPENAI_API_KEY` - Nunca commitada
- `PINECONE_API_KEY` - Nunca commitada
- `.env` - Adicionado ao `.gitignore`

### Validações
- Configuração validada no startup
- Inputs sanitizados
- Error handling em todos os módulos

---

## 🚀 Performance

### Otimizações
- Embeddings cacheados (OpenAI)
- Pinecone com índices otimizados
- Histórico limitado (últimos N turnos)
- Timeout nas APIs

### Limitações Atuais
- Sessões em memória (não persiste restart)
- Sem rate limiting
- Sem cache de respostas
- Sem pool de conexões

---

## 📈 Métricas e Logs

### Logs Registrados
- Inicialização de componentes
- Classificação de perguntas
- Handlers executados
- Resultados de buscas
- Erros e exceções

### Níveis de Log
- `DEBUG`: Detalhes técnicos
- `INFO`: Operações normais
- `WARNING`: Situações anômalas
- `ERROR`: Erros que precisam atenção

---

## 🔧 Configuração e Customização

### Ajustes Comuns

**Threshold de Relevância:**
```python
# Em document_search.py
searcher = DocumentSearcher(relevance_threshold=0.75)
```

**Temperatura do Modelo:**
```python
# Em agent.py, __init__
self.model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,  # Ajuste aqui
    openai_api_key=openai_api_key
)
```

**Histórico de Conversa:**
```python
# Em agent.py, handlers
history_context = self.conversation.get_history_summary(last_n=10)  # Ajuste aqui
```

---

## 📝 Próximos Passos de Desenvolvimento

### Funcionalidades Sugeridas
1. [ ] Persistência de sessões (Redis/PostgreSQL)
2. [ ] Cache de respostas frequentes
3. [ ] Rate limiting na API
4. [ ] Autenticação JWT
5. [ ] Dashboard de métricas
6. [ ] Testes unitários com pytest
7. [ ] CI/CD pipeline
8. [ ] Containerização (Docker)
9. [ ] Suporte a múltiplos idiomas
10. [ ] Análise de sentimento

### Melhorias de Performance
1. [ ] Connection pooling
2. [ ] Async/await onde possível
3. [ ] Batch processing de embeddings
4. [ ] Caching inteligente
5. [ ] Load balancing

---

**Documentação criada em:** 2025-01-25
**Versão do Sistema:** 1.0.0
