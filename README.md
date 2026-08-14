# Agentic Question Answering over DBpedia

A GSoC 2026 project that translates natural language questions into SPARQL queries against the DBpedia knowledge graph using an agentic LangGraph pipeline.

**Contributor:** Malla Siddharth Reddy  
**Organization:** DBpedia  
**Mentors:** Tommaso Soru, Ronit Banerjee, Gandharva Naveen, Abdulsobur  
**GSoC Blog (weekly blogs):** https://mallasiddharthreddy.github.io/blogs/gsoc-26/

---

## How It Works

The pipeline consists of six LangGraph nodes that run in sequence:

1. **Planner** — Analyses the question and extracts entities, concepts, answer type, aggregator, join type, whether a type filter is needed, and the number of relationship hops (`num_hops`) required to answer the question
2. **Entity Linker** — Maps entity mentions to DBpedia resource URIs using a Redis surface-form index with LLM disambiguation
3. **Ontology Explorer** — Looks up the top-15 most semantically similar dbo: properties for each concept using a Nomic Embed v1.5 cosine index, then enriches each candidate with rdfs:domain, rdfs:range, and rdfs:label from the DBpedia OWL ontology file via the Schema Introspector
4. **Query Builder** — Generates a SPARQL query using the linked entities and ontology candidates, treating `num_hops` from the Planner as a hard constraint on query structure
5. **Query Executor** — Executes the query against the DBpedia endpoint with a class-safe deterministic dbo: to dbp: namespace swap as a first-level fallback
6. **Validator** — Inspects the result and either passes, retries the Query Builder with live agentic probe context, or gives up

The Validator node runs several checks and a two-stage agentic probe when a query fails:

- **Dead URI detection** — if a subject URI returns zero properties even from an unfiltered probe, the Validator checks for DBpedia disambiguation suffixes (e.g. `_(series)`) and retries with the cleaned URI instead of giving up
- **Two-hop probe chaining** — for multi-hop questions (`num_hops >= 2`), the Validator resolves what the intermediate variable in the query actually binds to, and probes that entity too, not just the original subject. This recovers cases where the second-hop property only exists on the intermediate entity
- **Type-aware probe filtering** — when probing the original subject of a multi-hop question, only properties with an actual resource-URI value are accepted, since that value needs to be usable as the subject of the next triple. A plain string value there could never be joined against
- **Two-stage probe** — Stage 1 searches for dbp: properties matching the concept keyword on the subject entity, Stage 2 falls back to listing all dbp: properties for that entity

Probe results are injected as grounded context into the Query Builder on retry.

```
Planner -> Entity Linker -> Ontology Explorer -> Query Builder -> Query Executor -> Validator
                                                       ^                               |
                                               retry_query_builder <-- probe_context  |
                                               pass / give_up ----------------------> END
```

A browser-based UI is also available for live demonstration, showing every step of the pipeline (including hop count, agentic probe results, and validator decisions) as it runs. See "Running the Pipeline" below.

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
NEF_REDIS_HOST=your_redis_host
NEF_REDIS_PORT=6379
NEF_REDIS_PASSWORD=your_redis_password
```
LLM calls go through OpenRouter by default. To use a different OpenAI-compatible
provider (local, Bedrock, etc.), edit `_get_llm_client()` in `src/agent.py` and
`_get_judge_client()` in `src/evaluate.py`.

**3. Build the dbo ontology index:**

The DBpedia OWL ontology file is included in the repo at `resources/dbpedia-20250806.owl.rdf` — no separate download needed. Run:
```bash
pipenv run python scripts/build_ontology_index.py
```

This produces `data/nomic_embeddings_dbo.pt`, `data/nomic_uris_dbo.json`, and `data/nomic_labels_dbo.json`.

**4. Set up Redis entity linking (optional):**

The entity linker uses a Redis database pre-loaded with DBpedia surface forms. Contact the project maintainers for access, or refer to the DBpedia entity linking documentation. Redis is not strictly required: if unavailable, the pipeline falls back to constructing entity URIs directly from the name, with lower linking accuracy.

**5. DBpedia endpoint:**

All queries go to `http://research.liberai.org:7878/sparql`. To use a different endpoint, update the `ENDPOINT` constant in `src/sparql_client.py`, `src/evaluate.py`, `scripts/build_db26_gold.py`, and `scripts/build_db25_gold.py`.

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
Then open `http://localhost:8000` in your browser. Type a question and watch the pipeline's steps live — Planner analysis (including hop count and query metadata), entity linking, ontology lookup, generated SPARQL, executor fallback, validator decisions, and full agentic probe results, exactly as they print in the terminal.

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
# DB26, 50 questions, Claude
pipenv run python -m src.evaluate 50 anthropic/claude-sonnet-4.6 0 db26

# DB25, 100 questions, DeepSeek
pipenv run python -m src.evaluate 100 deepseek/deepseek-v3.2 0 db25

# DB26, starting from question 10, Qwen
pipenv run python -m src.evaluate 40 qwen/qwen3.5-122b-a10b 10 db26
```

Arguments: `n_questions  model_id  start_index  benchmark`

Each run reports average F1, precision, recall, and average/median/mode of agent steps per question (a measure of how many retries the pipeline needed). Results are saved to `data/evals/eval_{benchmark}_{timestamp}.json`.

---

## Benchmark Results

Evaluated on the Text2SPARQL 2026 DB26 benchmark (50 questions) using the dbo-only architecture with all pipeline improvements applied.

| Model | Result-set match | Avg F1 |
|---|---|---|
| Claude Sonnet 4.6 | 27/50 (54%) | 0.6119 |
| Qwen 3.5 122B | 22/50 (44%) | 0.5109 |
| DeepSeek v3.2 | 18/50 (36%) | 0.41 |
| LIBER-AI-CLAUDE (pre-GSoC baseline) | — | 0.32 |
| LIBER-AI-QWEN (pre-GSoC baseline) | — | 0.31 |

The Claude Sonnet 4.6 result of F1=0.6119 is a 90% relative improvement over the LIBER-AI-CLAUDE pre-GSoC baseline (0.32), and is competitive with 2nd place on the official Text2SPARQL 2026 leaderboard (F1=0.614).

Evaluated on Text2SPARQL 2025 DB25 benchmark (100 questions):

| Model | Result-set match | Avg F1 |
|---|---|---|
| DeepSeek v3.2 | 50/100 (50%) | 0.5538 |

Note: several DB25 gold queries have compatibility issues with the evaluation endpoint (invalid SPARQL COUNT/SUM syntax in the gold queries, and a few genuine full-scan timeouts). These questions automatically score 0, so the true pipeline performance on well-formed questions is somewhat higher than the aggregate F1 suggests.

Full raw output for all three final runs is in `eval-results/`. For a per-question
breakdown of every DB26 failure on Claude and Qwen, including a categorised analysis
of what is causing each one, see `docs/final-results-analysis.md`.

---

## Project Structure

```
agentic-kgqa/
├── src/
│   ├── agent.py              # LangGraph pipeline, all node functions, KGQAState
│   ├── sparql_client.py      # Centralised SPARQL execution
│   ├── query_executor.py     # Query execution with dbo->dbp swap
│   ├── validator.py          # Validator node: agentic probe, dead URI detection,
│   │                         #   two-hop probe chaining, type-aware filtering
│   ├── ontology_lookup.py    # dbo: embedding index lookup (Nomic Embed v1.5)
│   ├── schema_introspector.py # OWL metadata enrichment (domain, range, label)
│   ├── entity_linking.py     # Redis-based entity linking
│   ├── evaluate.py           # Evaluation harness with P/R/F1 and steps metrics
│   └── api.py                # FastAPI streaming endpoint + web UI
├── scripts/
│   ├── build_ontology_index.py  # Build dbo: Nomic embedding index
│   ├── build_db26_gold.py       # Pre-compute DB26 gold result cache
│   └── build_db25_gold.py       # Pre-compute DB25 gold result cache
├── benchmark/
│   ├── questions_db26.yml    # Text2SPARQL 2026 benchmark (50 questions)
│   └── questions_db25.yaml   # Text2SPARQL 2025 benchmark (100 questions)
├── resources/
│   └── dbpedia-20250806.owl.rdf  # DBpedia OWL ontology file (included in repo)
├── eval-results/              # Full raw output of the final confirmed evaluation runs
│   ├── claude-db26-full-final.json
│   ├── qwen-db26-full-final.json
│   └── DeepSeek-db25-full-final.json
├── data/                     # Index files, gold caches, eval results will be stored here once you run the steps above (gitignored)
└── docs/
    └── final-results-analysis.md  # Per-question failure analysis for Claude and Qwen on DB26
```

---

## Architecture Decisions

**dbo-only ontology index:** The pipeline uses only the curated dbo: ontology for static property lookup, dropping an earlier AI-labelled dbp: index. A controlled comparison on DB26 showed the two approaches were statistically indistinguishable in performance, while the dbo-only approach is simpler and more scientifically defensible and scalable. dbp: properties still enter the pipeline via the deterministic swap and the live agentic probe.

**Agentic probe over static dbp: index:** Instead of embedding 49k dbp: properties with AI-generated labels (not reproducible at scale), the Validator queries the live endpoint to discover which dbp: properties actually have data for a given subject entity. This grounds the Query Builder's retry in real confirmed data rather than embedding similarity.

**Class-safe dbo to dbp swap:** The deterministic namespace swap only rewrites lowerCamelCase property URIs. UpperCamelCase class URIs (used in rdf:type constraints) are left untouched since no dbp: equivalent exists for classes like dbo:FictionalCharacter.

**num_hops as an explicit Planner field:** Multi-hop questions were sometimes collapsed into a single triple when the Query Builder inferred hop count purely from natural language phrasing. The Planner now explicitly outputs `num_hops`, and the Query Builder treats it as a hard constraint (e.g. `num_hops=2` must produce exactly two triples connected by an intermediate variable), removing this class of structural error.

**Dead URI detection:** When the Entity Linker falls back to a heuristic URI construction (on a Redis miss) and that heuristic includes a DBpedia disambiguation suffix like `_(series)`, the constructed URI can point to a non-existent resource. The Validator detects this — when the unfiltered probe finds zero properties for a subject — strips the suffix, and retries with the cleaned canonical URI.

**Two-hop probe chaining:** For multi-hop questions, the original subject-only probe could never discover a second-hop property that only exists on the intermediate entity (e.g. a country's area, when the question starts from a university located in that country). The Validator now resolves what the intermediate variable actually binds to and probes that entity as well.

**Type-aware probe filtering:** When probing the first hop of a multi-hop question, only properties with a resource-URI value are surfaced, since that value must be usable as the subject of the second triple. A plain string value found by the probe (e.g. a literal describing a unit's parent organisation) can never be joined further, so accepting it would only produce another failed retry.

---

