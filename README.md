# simple_agent_llm

Pequeno agente baseado em LLM com roteamento de perguntas. Consulte o `QUICKSTART.md` para instruções detalhadas.

## Instalação rápida
Use o script de instalação para evitar o erro do pip (`AssertionError: len(weights) == expected_node_count`) observado em versões 25.x. Ele fixa o pip em uma versão estável antes de instalar as dependências e agora inclui o PyTorch, que corrige o erro `libtorch_global_deps.so` ausente em algumas instalações.

```bash
bash scripts/install.sh
```

Em seguida configure o arquivo `.env` conforme o exemplo e execute `python main.py --api` para iniciar a API.
