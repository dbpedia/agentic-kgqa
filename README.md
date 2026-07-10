# Agentic Question Answering over DBpedia

A GSoC 2026 project that translates natural language questions into SPARQL queries against the DBpedia knowledge graph using an agentic LangGraph pipeline.

**Contributor:** Malla Siddharth Reddy  
**Organization:** DBpedia  
**Mentors:** Tommaso Soru, Ronit Banerjee, Gandharva Naveen, Abdulsobur  
**Blog:** https://mallasiddharthreddy.github.io/blogs/gsoc-26/

---

## How It Works

The pipeline consists of six LangGraph nodes that run in sequence:

1. **Planner** — Analyses the question and extracts entities, concepts, answer type, aggregator, join type, and whether a type filter is needed
2. **Entity Linker** — Maps entity mentions to DBpedia resource URIs using a Redis surface-form index with LLM disambiguation
3. **Ontology Explorer** — Looks up the top-15 most semantically similar dbo: properties for each concept using a Nomic Embed v1.5 cosine index, then enriches each candidate with rdfs:domain, rdfs:range, and rdfs:label from the DBpedia OWL ontology file via the Schema Introspector
4. **Query Builder** — Generates a SPARQL query using the linked entities and ontology candidates
5. **Query Executor** — Executes the query against the DBpedia endpoint with a class-safe deterministic dbo: to dbp: namespace swap as a first-level fallback
6. **Validator** — Inspects the result and either passes, retries the Query Builder with live agentic probe context, or gives up

The Validator node runs a two-stage agentic probe when a query fails: Stage 1 searches for dbp: properties matching the concept keyword on the subject entity, Stage 2 falls back to listing all dbp: properties for that entity. Probe results are injected as grounded context into the Query Builder on retry.

```
Planner -> Entity Linker -> Ontology Explorer -> Query Builder -> Query Executor -> Validator
                                                       ^                               |
                                               retry_query_builder <-- probe_context  |
                                               pass / give_up ----------------------> END
```

---

## Requirements

- Python 3.13
- Redis server running locally (for entity linking)
- OpenRouter API key (for LLM calls)
- DBpedia evaluation endpoint access

---

## Setup

**1. Install dependencies:**
```bash
pipenv install
```

**2. Set up environment variables:**

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**3. Build the dbo ontology index:**

Download the DBpedia OWL ontology file and place it at `data/dbpedia-20250806.owl.rdf`, then run:
```bash
pipenv run python scripts/build_ontology_index.py
```

This produces `data/nomic_embeddings_dbo.pt`, `data/nomic_uris_dbo.json`, and `data/nomic_labels_dbo.json`.

**4. Set up Redis entity linking:**

The entity linker uses a Redis database pre-loaded with DBpedia surface forms. Contact the project maintainers for access to the Redis dump, or refer to the DBpedia entity linking documentation.

---

## Running the Pipeline

**Single question (via the eval script):**
```bash
pipenv run python -m src.evaluate 1 deepseek/deepseek-v3.2 0 db26
```

**Single question (directly via Python):**
```bash
pipenv run python -c "
from src.agent import KGQAAgent
agent = KGQAAgent()
print(agent.answer('What is the birthplace of Keanu Reeves?'))
"
```

**Via the API (streaming UI):**
```bash
pipenv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Then open `http://localhost:8000` in your browser.

---

## Running Evaluations

**Build the gold result cache first (run once):**
```bash
# For DB26 (50 questions)
pipenv run python scripts/build_db26_gold.py

# For DB25 (100 questions)
pipenv run python scripts/build_db25_gold.py
```

This executes all gold SPARQL queries against the evaluation endpoint and caches the results to `data/db26_gold_final.json` and `data/db25_gold_final.json`. Building the cache eliminates gold-side timing variance between evaluation runs.

**Run a full evaluation:**
```bash
# DB26, 50 questions, DeepSeek
pipenv run python -m src.evaluate 50 deepseek/deepseek-v3.2 0 db26

# DB25, 100 questions, DeepSeek
pipenv run python -m src.evaluate 100 deepseek/deepseek-v3.2 0 db25

# DB26, starting from question 10, Qwen
pipenv run python -m src.evaluate 40 qwen/qwen3.5-122b-a10b 10 db26
```

Arguments: `n_questions  model_id  start_index  benchmark`

Results are saved to `data/evals/eval_{benchmark}_{timestamp}.json`.

---

## Benchmark Results

Evaluated on the Text2SPARQL 2026 DB26 benchmark (50 questions) using the dbo-only architecture.

| Model | Result-set match | Avg F1 |
|---|---|---|
| Claude Sonnet 4.6 | 24/50 (48%) | 0.55 |
| DeepSeek v3.2 | 18/50 (36%) | 0.41 |
| Qwen 3.5 122B | TBD | TBD |
| LIBER-AI-CLAUDE (baseline) | — | 0.32 |

Evaluated on Text2SPARQL 2025 DB25 benchmark (100 questions):

| Model | Result-set match | Avg F1 |
|---|---|---|
| DeepSeek v3.2 | 46/100 (46%) | 0.53 |

Note: 14 DB25 gold queries have compatibility issues with the evaluation endpoint (11 use invalid SPARQL COUNT/SUM syntax, 3 are genuine full-scan timeouts). These questions automatically score 0.

---

## Project Structure

```
agentic-kgqa/
├── src/
│   ├── agent.py              # LangGraph pipeline, all node functions, KGQAState
│   ├── sparql_client.py      # Centralised SPARQL execution
│   ├── query_executor.py     # Query execution with dbo->dbp swap
│   ├── validator.py          # Validator node with agentic probe
│   ├── ontology_lookup.py    # dbo: embedding index lookup (Nomic Embed v1.5)
│   ├── schema_introspector.py # OWL metadata enrichment (domain, range, label)
│   ├── entity_linking.py     # Redis-based entity linking
│   ├── evaluate.py           # Evaluation harness with P/R/F1 metrics
│   └── api.py                # FastAPI streaming endpoint + web UI
├── scripts/
│   ├── build_ontology_index.py  # Build dbo: Nomic embedding index
│   ├── build_db26_gold.py       # Pre-compute DB26 gold result cache
│   └── build_db25_gold.py       # Pre-compute DB25 gold result cache
├── benchmark/
│   ├── questions_db26.yml    # Text2SPARQL 2026 benchmark (50 questions)
│   └── questions_db25.yaml   # Text2SPARQL 2025 benchmark (100 questions)
├── data/                     # Index files, gold caches, eval results (gitignored)
└── docs/
```

---

## Architecture Decisions

**dbo-only ontology index:** The pipeline uses only the curated dbo: ontology for static property lookup, dropping an earlier AI-labelled dbp: index. A controlled comparison on DB26 showed the two approaches were statistically indistinguishable in performance, while the dbo-only approach is simpler and more scientifically defensible and scalable. dbp: properties still enter the pipeline via the deterministic swap and the live agentic probe.

**Agentic probe over static dbp: index:** Instead of embedding 49k dbp: properties with AI-generated labels (not reproducible at scale), the Validator queries the live endpoint to discover which dbp: properties actually have data for a given subject entity. This grounds the Query Builder's retry in real confirmed data rather than embedding similarity.

**Class-safe dbo to dbp swap:** The deterministic namespace swap only rewrites lowerCamelCase property URIs. UpperCamelCase class URIs (used in rdf:type constraints) are left untouched since no dbp: equivalent exists for classes like dbo:FictionalCharacter.

---

## GSoC Blog

Weekly coding blogs documenting progress: https://mallasiddharthreddy.github.io/blogs/gsoc-26/
