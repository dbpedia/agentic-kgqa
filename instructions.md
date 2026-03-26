# Agentic KGQA: Instructions for Claude Code

## Goal

Create a Question Answering agent that answers questions against the DBpedia knowledge graph, 2015-10 snapshot.

## Tools

- The DBpedia ontology used by the benchmark. You may adopt the same approach as in `agentic-kgqa.ipynb` (`lookup_term` function).
- The Redis entity linking service. You may adopt the same approach as in `src/entity_linking.py`.

## Benchmark

- The benchmark is available at `benchmark/`. The file contains the dataset from last year's text2sparql challenge.
- The evaluation will be carried out by the challenge organisers. We need to open an API endpoint. See `src/api_example.py` for an example.

## Important points

- DBpedia data is messy. Do not attempt to double-check whether a result is correct. Just focus on the translation into SPARQL.
- For now, focus on the English questions.

## Implemented features

### KGQA Agent (`src/agent.py`)
- 5-step pipeline: analyse question (LLM) -> entity linking (Redis) -> ontology lookup (embeddings) -> SPARQL generation (LLM) -> verify & revise
- **Versioned prompt system** (`PROMPT_VERSION`): prompt evolves based on eval failure analysis. Current version: **v5**.
  - v1: baseline — "prefer ontology properties over dbp:"
  - v2: explicit namespace guidance (dbo vs dbp), instructs LLM to pick properties from ontology lookup results by score rather than defaulting to dbo:. **Regressed to 6%** — over-corrected toward dbp: and lost DISTINCT.
  - v3: restores dbo: preference (use dbo when both exist), enforces SELECT DISTINCT, adds 3 diverse few-shot examples (dbo SELECT, dbp multi-hop, ASK).
  - v4: frequency-weighted ontology lookup — triple counts in results. **Regressed to 12%** — triple counts are anti-correlated with benchmark namespace preference.
  - v5: **hard dbo: preference + grouped dbo/dbp pairs**. Key insight from v2+v4: data-driven signals hurt namespace selection. Reverts to hard "always use dbo:" rule but keeps triple counts as informational context. Ontology results now show dbo/dbp pairs grouped together (e.g. "dbo:director / dbp:director — use dbo:"). Revision prompt explicitly prioritises dbo→dbp swap as fallback strategy. 5 few-shot examples retained from v4. See `docs/analysis.md` for full evolution analysis.
- **Self-correcting queries**: after generating SPARQL, the agent executes it against the local DBpedia 2015-10 endpoint (`http://localhost:7878/query`). If the result is empty (0 results or COUNT=0) or errors, the agent asks the LLM to revise the query (up to 2 retries). Revision prefers **simplification** (removing type constraints, then swapping dbo→dbp, then synonym properties) over adding UNIONs.
- Streaming mode via `answer_stream()` that yields step-by-step events including revision steps
- Full-URI output (no PREFIX shorthand) to avoid illegal SPARQL with special characters
- Heuristic fallback when Redis is unavailable
- **Model selector**: choose from mid-tier models across providers (Claude, GPT, DeepSeek, Qwen, Gemma) via UI dropdown. Model list defined in `MODELS` constant; all routed through OpenRouter.
- `max_tokens=2048` cap on all LLM calls to prevent runaway generation

### Ontology Lookup (`src/ontology_lookup.py`)
- Semantic search over DBpedia ontology using precomputed embeddings (gensim KeyedVectors + OpenAI text-embedding-3-small)
- **Covers both `dbo:` (ontology/) and `dbp:` (property/) namespaces** — 7,098 vectors total (3,572 dbo + 3,526 dbp)
- Balanced results: returns a mix of dbo and dbp hits so the LLM can pick the right namespace
- **Frequency-weighted annotations**: each result includes triple count from the actual dataset (`data/predicate_frequencies.json`). Enables data-informed namespace choices.
- `dbp:` properties extracted from `data/infobox_properties_en.ttl` (filtered to ≥1,000 occurrences to exclude noise)
- Rebuild frequencies: `python scripts/build_predicate_frequencies.py` (requires local SPARQL endpoint)
- Reindex embeddings: `python scripts/reindex_ontology.py`

### Evaluation (`src/evaluate.py`)
- **LLM-based SPARQL comparison** using Claude Sonnet 4.6 as judge (always, regardless of generation model)
- Compares generated vs expected queries on **four axes**: BGP nodes (entity URIs), BGP predicates (property URIs including namespace), inner-query operators (FILTER/VALUES/OPTIONAL/UNION), outer-query operators (SELECT/ASK/COUNT/ORDER BY/LIMIT/DISTINCT)
- The node/predicate split reveals whether failures are due to wrong entities vs wrong properties (the main bottleneck is dbo/dbp namespace mismatches)
- Variable names are ignored — only functional equivalence matters
- **Prompt version tracking**: each eval result records the `prompt_version` used, enabling comparison across prompt iterations
- Randomly samples N questions from the benchmark (`benchmark/questions_db25.yaml`, 100 questions)
- Streams per-question results and summary stats (overall accuracy + per-axis breakdowns)
- **"All models" comparison mode**: runs the same question sample across all 6 models, streams per-model results, and renders a grouped bar chart comparing accuracy across all axes. Per-model summary cards shown alongside the chart.
- **Full benchmark button**: runs all 100 questions in order (no shuffling) for a deterministic full evaluation
- **Auto-save**: every evaluation run is automatically saved to `data/evals/<timestamp>_<model>.json` with full per-question results, summaries, and metadata
- **Past runs browser**: saved results are listed in the Evaluate tab and can be reloaded instantly (single-model table or all-models chart) without re-running

### API (`src/api.py`)
- `GET /answer?question=...&dataset=...` — Challenge endpoint returning `{dataset, question, query}`
- `GET /stream?question=...&model=...` — SSE endpoint streaming each agent step in real time (optional model override)
- `GET /evaluate?n=10&model=...` — SSE endpoint streaming evaluation results against benchmark. Use `model=all` to compare all models.
- `GET /eval-results` — List saved evaluation results (metadata)
- `GET /eval-results/{id}` — Load a full saved evaluation result
- `GET /models` — Returns available models and default
- `GET /` — Interactive web UI with Ask/Evaluate/History tabs, model selector, past runs browser, bar chart comparison, and live visualisation
- **History tab**: line chart showing performance metrics across prompt versions, with a summary table. Loads from saved eval results automatically.

### Entity Linking (`src/entity_linking.py`)
- Redis-backed surface form lookup with redirect resolution (from NEF)
- Default port fallback (6379)
