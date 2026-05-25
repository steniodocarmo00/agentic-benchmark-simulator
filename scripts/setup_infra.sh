set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ELASTICMQ_URL="${ELASTICMQ_URL:-http://localhost:9324}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-phi4-mini}"
OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-swarm-ollama}"

QUEUES=("tasks" "results" "tasks-dlq" "results-dlq")

log()  { printf "\n\033[1;36m==>\033[0m %s\n" "$*"; }
ok()   { printf "    \033[1;32m✓\033[0m %s\n"  "$*"; }
warn() { printf "    \033[1;33m!\033[0m %s\n"  "$*"; }
die()  { printf "\n\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

log "0/5  Verificando pré-requisitos"

command -v curl  >/dev/null || die "curl não encontrado no PATH."
command -v docker >/dev/null || die "docker não encontrado no PATH."
docker info >/dev/null 2>&1 || die "Docker daemon não está respondendo. Inicie o Docker Desktop."

if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  die "Nem 'docker compose' (v2) nem 'docker-compose' (v1) foram encontrados."
fi
ok "Usando: $DC"

log "1/5  docker compose up -d"
$DC up -d
ok "Containers iniciados (em background)."

log "2/5  Aguardando ElasticMQ em $ELASTICMQ_URL"
for i in $(seq 1 30); do
  if curl -fs "$ELASTICMQ_URL/?Action=ListQueues" >/dev/null 2>&1; then
    ok "ElasticMQ respondeu (após ${i} tentativas)."
    break
  fi
  sleep 2
  if [[ $i -eq 30 ]]; then
    die "ElasticMQ não respondeu em 60s. Cheque: $DC logs elasticmq"
  fi
done

log "3/5  Aguardando Ollama em $OLLAMA_URL"
for i in $(seq 1 60); do
  if curl -fs "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    ok "Ollama respondeu (após ${i} tentativas)."
    break
  fi
  sleep 2
  if [[ $i -eq 60 ]]; then
    die "Ollama não respondeu em 120s. Cheque: $DC logs ollama"
  fi
done

log "4/5  Criando filas SQS no ElasticMQ (idempotente)"
for q in "${QUEUES[@]}"; do
  if curl -fs -X POST "$ELASTICMQ_URL/" \
        --data-urlencode "Action=CreateQueue" \
        --data-urlencode "QueueName=$q" >/dev/null 2>&1; then
    ok "fila '$q' pronta"
  else
    warn "fila '$q' falhou ao criar (pode já existir com atributos diferentes)"
  fi
done

log "5/5  Baixando modelo '$MODEL_NAME' (pode demorar — ~9 GB)"
if docker exec "$OLLAMA_CONTAINER" ollama list 2>/dev/null | grep -q "^$MODEL_NAME"; then
  ok "Modelo '$MODEL_NAME' já presente no volume. Pulando download."
else
  docker exec "$OLLAMA_CONTAINER" ollama pull "$MODEL_NAME"
  ok "Modelo '$MODEL_NAME' baixado."
fi

cat <<EOF

\033[1;32m Infraestrutura pronta.\033[0m

  ElasticMQ  : $ELASTICMQ_URL   (UI: http://localhost:9325)
  Ollama     : $OLLAMA_URL
  Filas      : ${QUEUES[*]}
  Modelo     : $MODEL_NAME

Smoke tests sugeridos:

  # listar filas
  curl '$ELASTICMQ_URL/?Action=ListQueues'

  # listar modelos disponíveis
  curl $OLLAMA_URL/api/tags

  # rodar uma inferência de 1 token (sanidade do Phi-4)
  curl $OLLAMA_URL/api/generate -d '{
    "model": "$MODEL_NAME",
    "prompt": "What is 2+2? Answer with a single number.",
    "stream": false
  }'

EOF
