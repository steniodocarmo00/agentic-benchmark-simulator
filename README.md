# Agentic Benchmark Simulator

Simulador de enxame de agentes LLM para benchmarking sobre GSM8K.
Projeto acadêmico — Tema 8.

## Status
Passo 1: arquitetura definida e esqueleto de diretórios criado. Nenhum código de aplicação ainda.

## Stack
- **Linguagem:** Python 3.11+
- **Modelo:** Phi-4 via Ollama (auto-hospedado)
- **Mensageria:** AWS SQS (produção) / ElasticMQ (local)
- **Container:** Docker + docker-compose
- **Benchmark:** GSM8K

## Componentes
- `master/`     — Coordenador: monta RunPlan e enfileira tarefas
- `worker/`     — Consumidor das filas; chama o modelo
- `aggregator/` — Coleta resultados e aplica voto majoritário
- `inference/`  — Cliente Ollama (retry + backoff)
- `messaging/`  — Abstração sobre SQS / ElasticMQ
- `observability/` — Logs estruturados + métricas
- `prompts/`    — System prompts versionados (ISOLADO do código)

## Como Rodar
A ser definido nos próximos passos.
