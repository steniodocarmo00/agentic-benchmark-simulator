# Prompts — Contexto Versionado

Este diretório é **deliberadamente isolado** do código Python. Nenhum módulo
em `src/` deve conter system prompts hard-coded.

## Regras
1. Cada prompt é um arquivo `.md` com front-matter YAML.
2. O `id` segue o padrão `<estrategia-curta>.v<N>` (ex: `cot.v1`, `zs.v2`).
3. Bump de versão é **mandatório** em qualquer alteração de conteúdo — não
   sobrescrever in-place. O histórico do git + a versão no nome do arquivo
   garantem reproducibilidade dos experimentos.
4. O `hash` é calculado em build e injetado em cada mensagem da fila, de modo
   que cada linha de `results.jsonl` aponte para o prompt exato usado.

## Estrutura
- `v1/` — primeira geração de prompts. Subdiretório por versão facilita
  comparação A/B futura.
- `schema/` — JSON Schema do front-matter, validado no boot do Master.
