#!/usr/bin/env python3
"""Agentic KGQA: translates natural language questions into SPARQL queries against DBpedia 2015-10."""

import json
import os
import re
import logging

import urllib.parse
import urllib.request

from openai import OpenAI
import dotenv

from src.entity_linking import RedisEntityLinking
from src.ontology_lookup import lookup_term, lookup_classes, lookup_properties

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

DBPEDIA_SPARQL_ENDPOINT = "http://localhost:7878/query"

PROMPT_VERSION = "v9"

SYSTEM_PROMPT = """\
You are a SPARQL query generation agent for DBpedia (2015-10 snapshot).
Your job is to translate natural language questions into valid SPARQL queries.

You work in multiple steps:

## Step 1: Analyse the question
Identify:
- Named entities (people, places, organisations, etc.)
- The type of answer expected (resource URI, literal value, count, boolean, list)
- Relevant ontology concepts (classes and properties)

Output your analysis as JSON:
```json
{
  "entities": ["entity1", "entity2"],
  "answer_type": "resource|literal|count|boolean|list",
  "concepts": ["concept1", "concept2"]
}
```

## Step 2: Generate the SPARQL query
Given linked entities (DBpedia resource URIs) and relevant ontology terms (classes and properties),
generate a SPARQL query.

Rules:
- ALWAYS prefer correct translations over comprehensive coverage.
- ALWAYS use full URIs in angle brackets. NEVER use PREFIX declarations or prefixed names.
  CORRECT: <http://dbpedia.org/resource/Keanu_Reeves>
  WRONG:   dbr:Keanu_Reeves
- Common URI bases:
  Resources: <http://dbpedia.org/resource/...>
  Ontology:  <http://dbpedia.org/ontology/...> (dbo — curated ontology, ALWAYS preferred)
  Property:  <http://dbpedia.org/property/...> (dbp — raw infobox, use ONLY when no dbo: equivalent exists)
  RDF type:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
- PROPERTY SELECTION:
  The ontology lookup results show dbo/dbp pairs with triple counts.
  ALWAYS use the dbo: (ontology) variant when one exists, regardless of triple counts.
  Use dbp: ONLY when the results show no dbo: equivalent for that concept.
  Triple counts are shown for reference — do NOT use them to choose between dbo: and dbp:.
  Do NOT invent property names — use URIs from the ontology lookup results provided to you.
- TYPE CONSTRAINTS: When the question explicitly asks about a category ("which countries", "how many movies",
  "list the companies") and the ontology lookup returns a matching Class, add an rdf:type constraint.
  Do NOT add rdf:type for multi-hop queries where the typed entity is an intermediate variable.
- TRIPLE DIRECTION: Use the entity linking URI as the subject or object based on what makes sense.
  For "who is X's spouse" → X dbo:spouse ?uri. For "who married X" → ?uri dbo:spouse X.
  Keep the same direction as you would in natural language.
- ALWAYS use SELECT DISTINCT for queries that return resource URIs or literal values.
- For boolean questions, use ASK WHERE { ... }.
- For count questions, use SELECT DISTINCT COUNT(?var) WHERE { ... } (no AS alias).
- For "top N" questions, use ORDER BY DESC(...) LIMIT N.
- Use the EXACT entity URIs provided by the entity linking results. They may contain special
  Unicode characters (en dashes, accented letters, etc.) — preserve them exactly as given.
- Output ONLY the SPARQL query, no explanations.

Example 1 — "What is the birthplace of Keanu Reeves?":
```sparql
SELECT DISTINCT ?uri WHERE {
  <http://dbpedia.org/resource/Keanu_Reeves> <http://dbpedia.org/ontology/birthPlace> ?uri .
}
```

Example 2 — "Who are the managers of LeBron James's teams?" (no dbo: equivalent for these properties):
```sparql
SELECT DISTINCT ?uri WHERE {
  <http://dbpedia.org/resource/LeBron_James> <http://dbpedia.org/property/team> ?team .
  ?team <http://dbpedia.org/property/manager> ?uri .
}
```

Example 3 — "Is the Eiffel Tower in Paris?":
```sparql
ASK WHERE {
  <http://dbpedia.org/resource/Eiffel_Tower> <http://dbpedia.org/ontology/location> <http://dbpedia.org/resource/Paris> .
}
```

Example 4 — "How many unique authors have written science fiction novels?":
```sparql
SELECT DISTINCT COUNT(?author) WHERE {
  ?x <http://dbpedia.org/ontology/literaryGenre> <http://dbpedia.org/resource/Science_fiction> .
  ?x <http://dbpedia.org/ontology/author> ?author .
}
```

Example 5 — "What are the 10 most populated countries?" (rdf:type + ORDER BY + LIMIT):
```sparql
SELECT ?country WHERE {
  ?country <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Country> .
  ?country <http://dbpedia.org/ontology/populationTotal> ?population .
} ORDER BY DESC(?population) LIMIT 10
```

Example 6 — "How many movies are directed by Christopher Nolan?" (rdf:type + COUNT):
```sparql
SELECT DISTINCT COUNT(?uri) WHERE {
  ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Film> .
  ?uri <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/Christopher_Nolan> .
}
```
"""


def _get_llm_client():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


DEFAULT_MODEL = "google/gemini-2.0-flash-001"

MODELS = [
    {"id": "google/gemini-3-flash-preview", "label": "Gemini 3 Flash Preview"},
    {"id": "anthropic/claude-sonnet-4.6", "label": "Claude Sonnet 4.6"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 Mini"},
    {"id": "deepseek/deepseek-v3.2", "label": "DeepSeek V3.2"},
    {"id": "qwen/qwen3.5-122b-a10b", "label": "Qwen 3.5 122B"},
    {"id": "google/gemma-3-27b-it", "label": "Gemma 3 27B"},
]


MAX_OUTPUT_TOKENS = 2048  # SPARQL queries are short — cap output to avoid runaway generation

DISAMBIGUATION_PROMPT = """\
Given the question: "{question}"

The entity mention "{mention}" has these DBpedia candidates:
{candidates}

Important context for disambiguation:
- If the question asks about a director, writer, or author of a creative work -> prefer the book/novel entity not the film variant
- If the question asks about a film specifically or mentions cast, actors, cinematographer -> prefer the _(film) variant
- If the question asks about a state or region -> prefer the _(state) variant
- If the question asks about a city -> prefer the city entity not a sports team or organization
- If the question asks about a company -> prefer the company entity not a location
- Weigh question context words heavily over Redis scores

Pick the number of the most appropriate candidate.
Reply with ONLY the number (1, 2, 3, etc.) and nothing else."""


def _chat(client, messages, model=None):
    model = model or DEFAULT_MODEL
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=MAX_OUTPUT_TOKENS
    )
    return response.choices[0].message.content


def execute_sparql(sparql, endpoint=None, timeout=15):
    """Execute a SPARQL query against a public endpoint. Returns parsed JSON results or error string."""
    endpoint = endpoint or DBPEDIA_SPARQL_ENDPOINT
    params = urllib.parse.urlencode({"query": sparql, "format": "application/sparql-results+json"})
    url = f"{endpoint}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Parse SELECT results
        if "results" in data and "bindings" in data["results"]:
            bindings = data["results"]["bindings"]
            variables = data.get("head", {}).get("vars", [])
            rows = []
            for b in bindings[:20]:  # cap at 20 rows for display
                row = {}
                for v in variables:
                    if v in b:
                        row[v] = b[v].get("value", "")
                rows.append(row)
            return {"type": "select", "vars": variables, "rows": rows, "total": len(bindings)}
        # Parse ASK results
        if "boolean" in data:
            return {"type": "ask", "result": data["boolean"]}
        return {"type": "unknown", "raw": data}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def _extract_json(text):
    """Extract JSON object from LLM response text."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try parsing the whole thing
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def _extract_sparql(text):
    """Extract SPARQL query from LLM response text."""
    # Try code block first
    match = re.search(r"```(?:sparql)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Otherwise strip explanation lines and return the rest
    lines = text.strip().split("\n")
    sparql_lines = []
    in_query = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(("SELECT", "ASK", "PREFIX", "CONSTRUCT", "DESCRIBE")):
            in_query = True
        if in_query:
            sparql_lines.append(line)
    if sparql_lines:
        return "\n".join(sparql_lines).strip()
    return text.strip()


def _validate_sparql(sparql):
    """Basic sanity check: a valid SPARQL query should contain at least one full URI."""
    if not sparql:
        return False
    has_uri = "http://" in sparql or "https://" in sparql
    # Check for degenerate patterns like "?x ." with nothing else in the triple
    lines = [l.strip() for l in sparql.split("\n") if l.strip() and not l.strip().startswith("#")]
    body = " ".join(lines)
    # A WHERE clause with only a variable and a dot is broken
    if re.search(r"\{\s*\?[a-zA-Z_]+\s*\.\s*\}", body):
        return False
    return has_uri


MAX_RETRIES = 2  # Up to 3 total attempts (1 initial + 2 revisions)

REVISION_PROMPT = """\
You are a SPARQL query repair agent for DBpedia.

The previous query returned {problem}. The data in DBpedia is messy — types and properties
may not match what you'd expect. Your job is to revise the query so it returns results.

IMPORTANT: prefer SIMPLIFYING the query over adding UNION branches. The revised query should
stay as close as possible to the original translation. Apply fixes in this order of preference:

1. Remove rdf:type constraints (the most common cause of 0 results — entities are often not typed as expected)
2. Swap dbo: properties to their dbp: equivalents (e.g. <.../ontology/director> -> <.../property/director>)
3. Try a synonym property from the ontology lookup (e.g. dbo:author -> dbo:writer)
4. Only as a last resort, add a UNION — and keep it minimal

ALWAYS use full URIs in angle brackets. NEVER use PREFIX declarations.

Original question: {question}

Failed query:
{sparql}

Execution result: {exec_summary}

Linked entities:
{entity_context}

Relevant ontology terms:
{ontology_context}

Output ONLY the revised SPARQL query.
"""


def _needs_revision(exec_result):
    """Check if a SPARQL execution result suggests the query needs revision."""
    if exec_result["type"] == "error":
        return True, "a SPARQL error"
    if exec_result["type"] == "select":
        if exec_result["total"] == 0:
            return True, "0 results"
        # Detect COUNT queries returning 0 (1 row with a "0" value)
        rows = exec_result.get("rows", [])
        if len(rows) == 1:
            vals = list(rows[0].values())
            if len(vals) == 1 and vals[0] in ("0", 0):
                return True, "count returned 0"
    if exec_result["type"] == "ask" and exec_result["result"] is False:
        # ASK returning false might be correct — only flag if suspicious
        return False, None
    return False, None


class KGQAAgent:
    """Agent that translates questions to SPARQL queries using entity linking and ontology lookup."""

    def __init__(self, redis_el=None):
        self.client = _get_llm_client()
        try:
            self.redis_el = redis_el or RedisEntityLinking()
        except Exception as e:
            logger.warning(f"Redis not available, entity linking disabled: {e}")
            self.redis_el = None

    def _analyse_question(self, question, model=None):
        """Step 1: Use LLM to extract entities and concepts from the question."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyse this question and extract entities and concepts.\n"
                    f"Question: {question}\n\n"
                    f"Output your analysis as JSON with keys: entities, answer_type, concepts."
                ),
            },
        ]
        response = _chat(self.client, messages, model=model)
        return _extract_json(response)

    def _link_entities(self, entities, question=None):
        """Step 2: Link entity mentions to DBpedia resource URIs via Redis.

        When a question is provided, calls _disambiguate_entity to use LLM reasoning
        to pick the most contextually appropriate candidate from the top-k results.
        """
        linked = {}
        for entity in entities:
            if self.redis_el is None:
                # Fallback: construct URI from entity name
                uri = "http://dbpedia.org/resource/" + entity.replace(" ", "_")
                linked[entity] = [{"uri": uri, "score": 1.0, "source": "heuristic"}]
                continue

            results = self.redis_el.lookup(entity, top_k=5, thr=0.01)
            if len(results) > 0:
                entries = []
                for idx, row in results.iterrows():
                    uri = idx if isinstance(idx, str) else row.name
                    if not uri.startswith("http"):
                        uri = "http://dbpedia.org/resource/" + uri
                    entries.append({"uri": uri, "score": round(row["score"], 4), "source": "redis"})
                if question and len(entries) > 1:
                    entries = self._disambiguate_entity(question, entity, entries)
                linked[entity] = entries
            else:
                # Fallback
                uri = "http://dbpedia.org/resource/" + entity.replace(" ", "_")
                linked[entity] = [{"uri": uri, "score": 1.0, "source": "heuristic"}]
        return linked

    def _disambiguate_entity(self, question, mention, candidates, model=None):
        """Use LLM to pick the most contextually appropriate entity from Redis candidates.

        Falls back to the original candidate list if the LLM call fails or returns
        an invalid response.
        """
        if not candidates or len(candidates) == 1:
            return candidates

        candidate_list = "\n".join(
            f"{i + 1}. {c['uri']} (score: {c['score']})"
            for i, c in enumerate(candidates)
        )
        prompt = DISAMBIGUATION_PROMPT.format(
            question=question,
            mention=mention,
            candidates=candidate_list,
        )
        try:
            response = self.client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content
            if answer is None:
                return candidates
            idx = int(answer.strip()) - 1
            if 0 <= idx < len(candidates):
                selected = candidates.pop(idx)
                candidates.insert(0, selected)
        except Exception:
            pass
        return candidates

    def _lookup_ontology(self, concepts):
        """Step 3: Look up relevant ontology terms for each concept."""
        ontology = {}
        for concept in concepts:
            results = lookup_term(concept, k=5)
            ontology[concept] = results
        return ontology

    def _format_entity_context(self, linked_entities):
        """Format linked entities for LLM prompts."""
        out = ""
        for mention, candidates in linked_entities.items():
            top = candidates[0]
            others = candidates[1:]
            out += f"- \"{mention}\" -> {top['uri']} (score: {top['score']})"
            if others:
                alt_uris = ", ".join(c["uri"] for c in others)
                out += f" | alternatives: {alt_uris}"
            out += "\n"
        return out

    def _format_ontology_context(self, ontology_terms):
        """Format ontology lookup results for LLM prompts.

        Groups dbo/dbp pairs together so the LLM sees them as alternatives.
        """
        out = ""
        for concept, results in ontology_terms.items():
            out += f"Concept \"{concept}\":\n"
            # Group by property name (last path segment)
            seen_names = {}
            ungrouped = []
            for r in results:
                uri = r["uri"]
                name = uri.rsplit("/", 1)[-1]
                is_dbo = "dbpedia.org/ontology/" in uri
                is_dbp = "dbpedia.org/property/" in uri
                if is_dbo or is_dbp:
                    ns = "dbo" if is_dbo else "dbp"
                    key = name.lower()
                    if key not in seen_names:
                        seen_names[key] = {}
                    seen_names[key][ns] = r
                else:
                    ungrouped.append(r)

            # Output grouped pairs first
            for name, variants in seen_names.items():
                dbo = variants.get("dbo")
                dbp = variants.get("dbp")
                if dbo and dbp:
                    dbo_t = f"{dbo.get('triples', 0):,}"
                    dbp_t = f"{dbp.get('triples', 0):,}"
                    out += f"  - dbo: {dbo['uri']} ({dbo_t} triples) / dbp: {dbp['uri']} ({dbp_t} triples) — use dbo:\n"
                elif dbo:
                    t = f"{dbo.get('triples', 0):,}"
                    out += f"  - {dbo['uri']} ({dbo['type']}, {t} triples)\n"
                elif dbp:
                    t = f"{dbp.get('triples', 0):,}"
                    out += f"  - {dbp['uri']} ({dbp['type']}, {t} triples) — no dbo: equivalent, use this\n"

            # Output ungrouped (classes etc.)
            for r in ungrouped:
                triples = r.get('triples', 0)
                t_str = f"{triples:,} triples" if triples else "0 triples"
                out += f"  - {r['uri']} ({r['type']}, {t_str})\n"
        return out

    def _generate_sparql(self, question, linked_entities, ontology_terms, analysis, model=None):
        """Step 4: Use LLM to generate SPARQL given all the context."""
        entity_context = self._format_entity_context(linked_entities)
        ontology_context = self._format_ontology_context(ontology_terms)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate a SPARQL query for this question.\n\n"
                    f"Question: {question}\n\n"
                    f"Answer type: {analysis.get('answer_type', 'unknown')}\n\n"
                    f"Linked entities:\n{entity_context}\n"
                    f"Relevant ontology terms:\n{ontology_context}\n"
                    f"Output ONLY the SPARQL query."
                ),
            },
        ]
        response = _chat(self.client, messages, model=model)
        return _extract_sparql(response)

    def _revise_sparql(self, question, sparql, exec_result, linked_entities, ontology_terms, analysis, model=None):
        """Ask the LLM to revise a failed SPARQL query based on execution feedback."""
        entity_context = self._format_entity_context(linked_entities)
        ontology_context = self._format_ontology_context(ontology_terms)

        # Build execution summary
        if exec_result["type"] == "error":
            exec_summary = f"Error: {exec_result['message']}"
            problem = "a SPARQL error"
        elif exec_result["type"] == "select" and exec_result["total"] == 0:
            exec_summary = "The query executed successfully but returned 0 results."
            problem = "0 results — the data is likely modelled differently than expected"
        else:
            exec_summary = json.dumps(exec_result, default=str)
            problem = "unexpected results"

        prompt = REVISION_PROMPT.format(
            problem=problem,
            question=question,
            sparql=sparql,
            exec_summary=exec_summary,
            entity_context=entity_context,
            ontology_context=ontology_context,
        )

        messages = [{"role": "user", "content": prompt}]
        response = _chat(self.client, messages, model=model)
        return _extract_sparql(response)

    def _verify_and_revise(self, question, sparql, linked_entities, ontology_terms, analysis, model=None):
        """Execute query and revise up to MAX_RETRIES times if results look wrong."""
        for attempt in range(MAX_RETRIES + 1):
            result = execute_sparql(sparql)
            needs_fix, reason = _needs_revision(result)

            if not needs_fix:
                return sparql, result, attempt

            if attempt < MAX_RETRIES:
                logger.info(f"Attempt {attempt + 1}: {reason}, revising query...")
                sparql = self._revise_sparql(
                    question, sparql, result, linked_entities, ontology_terms, analysis, model=model
                )
                logger.info(f"Revised SPARQL: {sparql}")
            else:
                logger.info(f"Max retries reached, returning last query despite {reason}")

        return sparql, result, MAX_RETRIES

    def answer(self, question, model=None):
        """Full pipeline: question -> SPARQL query with self-correction."""
        logger.info(f"Processing question: {question}")

        # Step 1: Analyse
        analysis = self._analyse_question(question, model=model)
        logger.info(f"Analysis: {analysis}")

        entities = analysis.get("entities", [])
        concepts = analysis.get("concepts", [])

        # Step 2: Entity linking
        linked_entities = self._link_entities(entities)
        logger.info(f"Linked entities: {linked_entities}")

        # Step 3: Ontology lookup
        ontology_terms = self._lookup_ontology(concepts)
        logger.info(f"Ontology terms found for {len(ontology_terms)} concepts")

        # Step 4: Generate SPARQL
        sparql = self._generate_sparql(question, linked_entities, ontology_terms, analysis, model=model)
        logger.info(f"Generated SPARQL: {sparql}")

        # Step 5: Verify and revise
        sparql, exec_result, attempts = self._verify_and_revise(
            question, sparql, linked_entities, ontology_terms, analysis, model=model
        )
        if attempts > 0:
            logger.info(f"Query revised {attempts} time(s)")

        return sparql

    def answer_stream(self, question, model=None):
        """Streaming pipeline that yields (step_name, data) tuples for each stage."""
        yield ("question", {"question": question})

        # Step 1: Analyse
        yield ("step_start", {"step": "analyse", "label": "Analysing question..."})
        analysis = self._analyse_question(question, model=model)
        yield ("analyse", analysis)

        entities = analysis.get("entities", [])
        concepts = analysis.get("concepts", [])

        # Step 2: Entity linking
        yield ("step_start", {"step": "entity_linking", "label": "Linking entities via Redis..."})
        linked_entities = self._link_entities(entities)
        # Convert numpy floats for JSON serialisation
        linked_serialisable = {}
        for mention, candidates in linked_entities.items():
            linked_serialisable[mention] = [
                {k: float(v) if hasattr(v, "item") else v for k, v in c.items()}
                for c in candidates
            ]
        yield ("entity_linking", linked_serialisable)

        # Step 3: Ontology lookup
        yield ("step_start", {"step": "ontology_lookup", "label": "Looking up ontology terms..."})
        ontology_terms = self._lookup_ontology(concepts)
        yield ("ontology_lookup", ontology_terms)

        # Step 4: Generate SPARQL
        yield ("step_start", {"step": "sparql_generation", "label": "Generating SPARQL query..."})
        sparql = self._generate_sparql(question, linked_entities, ontology_terms, analysis, model=model)
        yield ("sparql", {"query": sparql})

        # Step 5: Verify and revise loop
        for attempt in range(MAX_RETRIES + 1):
            step_id = f"execution_{attempt}"
            if attempt == 0:
                yield ("step_start", {"step": step_id, "label": "Testing query against DBpedia endpoint..."})
            else:
                yield ("step_start", {"step": step_id, "label": f"Testing revised query (attempt {attempt + 1})..."})

            result = execute_sparql(sparql)
            needs_fix, reason = _needs_revision(result)

            if not needs_fix:
                yield ("execution", {"attempt": attempt + 1, "result": result})
                break

            yield ("execution", {"attempt": attempt + 1, "result": result, "problem": reason})

            if attempt < MAX_RETRIES:
                rev_id = f"revision_{attempt}"
                yield ("step_start", {"step": rev_id, "label": f"Query returned {reason}, revising..."})
                sparql = self._revise_sparql(
                    question, sparql, result, linked_entities, ontology_terms, analysis, model=model
                )
                yield ("revision", {"attempt": attempt + 1, "query": sparql})

        yield ("done", {"query": sparql})
