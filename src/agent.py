#!/usr/bin/env python3
"""Agentic KGQA: translates natural language questions into SPARQL queries against DBpedia 2015-10."""

import json
import os
import re
import logging
import functools

import urllib.parse
import urllib.request
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from openai import OpenAI
import dotenv

from src.entity_linking import RedisEntityLinking
from src.ontology_lookup import lookup_term, lookup_classes, lookup_properties
from src import schema_introspector

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
  The ontology lookup results show ranked candidates with confidence scores.
  ALWAYS reason over all candidates before choosing — consider the question wording and each candidate's label.
  The top-ranked candidate (highest score) has strong semantic similarity to the concept — give it extra weight.
  Only choose a lower-ranked candidate if its label is a significantly better match for the specific wording of the question.
  Example: for concept 'author', top-1 is dbo:author (88%) and dbo:writer (81%) also appears. The question says 'written' but dbo:author is the canonical DBpedia property — prefer dbo:author unless the question specifically says 'writer'.
  Example: for concept 'country', if the question says 'where does X START' or 'source of X', prefer dbo:sourceCountry over dbo:country even if dbo:country ranks higher — the question's 'start/source' word signals a more specific predicate.
  ALWAYS use the dbo: (ontology) variant when one exists, regardless of scores.
  Use dbp: ONLY when the results show no dbo: equivalent for that concept.
  Do NOT invent property names — use URIs from the ontology lookup results provided to you.
- AGGREGATION: Use the aggregator field from the analysis:
  COUNT -> SELECT DISTINCT COUNT(?var) WHERE (no AS alias)
  SUM -> SELECT SUM(?var) WHERE
  GROUP_BY -> SELECT ?var WHERE { ... } GROUP BY ?var ORDER BY DESC(COUNT(...))
  ORDER_BY_DESC -> ORDER BY DESC(?var) LIMIT N
  ORDER_BY_ASC -> ORDER BY ASC(?var)
  NONE -> plain SELECT DISTINCT
- JOIN TYPE: Use the join_type field from the analysis:
  INTERSECTION -> shared variable pattern: <X> pred ?uri . <Y> pred ?uri (finds common values)
  UNION -> { <X> pred ?uri } UNION { <Y> pred ?uri } (finds values from either)
  SINGLE -> normal single triple pattern
- TYPE FILTER: If has_type_filter is true, add rdf:type constraint using the matching Class from ontology results.
- DOMAIN/RANGE GUIDANCE: The ontology terms include domain and range metadata.
  Use domain to validate the subject type — if domain=Film and the variable is the subject, optionally add rdf:type dbo:Film.
  Use range to understand the return type — if range=nonNegativeInteger the result is a number not a URI.
  Only add rdf:type from domain/range when the question asks for a specific category AND the domain/range matches that category.
- MULTI-ENTITY QUESTIONS: When a question asks about TWO entities using 'and' (e.g. 'Where were X and Y born?', 'What did X and Y have in common?'), use a JOIN pattern with a shared variable — NOT a UNION.
  CORRECT: <X> dbo:birthPlace ?uri . <Y> dbo:birthPlace ?uri  (finds places where BOTH were born)
  WRONG:   { <X> dbo:birthPlace ?uri } UNION { <Y> dbo:birthPlace ?uri }  (finds places where EITHER was born)
  Use UNION only when the question explicitly says 'or' or asks for results from either entity separately.
- TRIPLE DIRECTION: Use the entity linking URI as the subject or object based on what makes sense.
  For "who is X's spouse" → X dbo:spouse ?uri. For "who married X" → ?uri dbo:spouse X.
  Keep the same direction as you would in natural language.
  IMPORTANT: Use domain/range metadata to determine correct triple direction.
  If domain=Film and one entity is a Film while another is a Person, the Film must be the subject.
  If domain=Organisation and the linked entities are Products (not Organisations), then the Organisation is the UNKNOWN variable (?uri) as subject, and the Products are the objects:
    CORRECT: ?uri dbo:product <IPhone> . ?uri dbo:product <IPad>  (finding unknown Organisation)
    WRONG:   <IPhone> dbo:product ?uri  (iPhone is not an Organisation)
  Example: dbo:starring has domain=Work, range=Actor. Film is subject, Person is object:
    CORRECT: <Film> dbo:starring <Person>
    WRONG:   <Person> dbo:starring <Film>
  Rule: the linked entity goes in the position (subject or object) that matches its type against domain/range.
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
2. Try a synonym property from the ontology lookup (e.g. dbo:author -> dbo:writer)
3. Only as a last resort, add a UNION — and keep it minimal

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


_ONTOLOGY_URI_RE = re.compile(r"http://dbpedia\.org/ontology/([^\s<>\"'),.;]+)")


def _swap_dbo_to_dbp(sparql):
    """Deterministically swap dbo: ontology URIs to dbp: property URIs —
    PROPERTY positions only.

    This is the first fallback step in the executor — faster and more reliable
    than asking the LLM to perform the swap inside a revision prompt.

    DBpedia naming convention: properties are lowerCamelCase, classes are
    UpperCamelCase. There is no dbp: namespace equivalent for classes —
    dbp:FictionalCharacter, dbp:TelevisionShow, dbp:Software etc do not exist.
    A naive blanket ontology/ -> property/ replace corrupts any rdf:type
    constraint in the query (silently turning a valid class URI into a
    non-existent dbp: URI, which then always returns zero results), so this
    only swaps dbo: URIs whose local name starts with a lowercase letter
    (i.e. properties), leaving class URIs untouched.
    """
    def _maybe_swap(match):
        local_name = match.group(1)
        if local_name and local_name[0].islower():
            return f"http://dbpedia.org/property/{local_name}"
        return match.group(0)  # leave class URIs (UpperCamelCase) untouched

    return _ONTOLOGY_URI_RE.sub(_maybe_swap, sparql)


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


# ─── LangGraph State ─────────────────────────────────────────────────────────

class KGQAState(TypedDict):
    """Typed state that flows through all LangGraph nodes.

    Each node receives the full state, performs its task, and returns
    only the fields it updates. LangGraph merges updates back automatically.
    """
    # Input
    question:         str
    model:            Optional[str]

    # Planner node output
    entities:         list
    concepts:         list
    answer_type:      str
    aggregator:       str
    join_type:        str
    has_type_filter:  bool

    # Entity Linker node output
    linked_entities:  dict

    # Ontology Explorer + Schema Introspector node output
    ontology_terms:   dict

    # Query Builder node output
    sparql:           str

    # Query Executor node output
    exec_result:      dict
    attempts:         int
    swap_attempted:   bool

    # Validator node output
    is_valid:         bool
    retry_target:     str   # "query_builder" | "entity_linker" | "end"


# ─── Standalone Node Functions ────────────────────────────────────────────────
# These functions mirror the KGQAAgent class methods and are used by the
# LangGraph StateGraph. The KGQAAgent class methods remain intact for
# backward compatibility with CorporateKGQAAgent and existing callers.

def planner_node(state: KGQAState, client, model: Optional[str] = None) -> dict:
    """Planner node: analyse the question and extract entities, concepts, and
    query metadata (aggregator, join_type, has_type_filter).

    Corresponds to KGQAAgent._analyse_question().
    Returns partial state update.
    """
    question = state["question"]
    model = model or state.get("model") or DEFAULT_MODEL

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analyse this question and extract entities and concepts.\n"
                f"Question: {question}\n\n"
                f"Output your analysis as JSON with these exact keys:\n"
                f"- entities: list of named entities (people, places, organisations, works)\n"
                f"- answer_type: one of resource, literal, count, boolean, list\n"
                f"- concepts: list of relationship/property keywords\n"
                f"- aggregator: one of NONE, COUNT, SUM, GROUP_BY, ORDER_BY_DESC, ORDER_BY_ASC\n"
                f"  (NONE = plain SELECT, COUNT = how many, SUM = total of values, "
                f"GROUP_BY = list ranked by count, ORDER_BY_DESC/ASC = top-N or ranked list)\n"
                f"- join_type: one of INTERSECTION, UNION, SINGLE\n"
                f"  (INTERSECTION = question asks what two entities have IN COMMON using 'and', "
                f"UNION = question asks about either entity using 'or', "
                f"SINGLE = normal single-entity question)\n"
                f"- has_type_filter: true if the question explicitly asks for a category "
                f"like 'which movies', 'list countries', 'how many companies' etc, false otherwise\n\n"
                f"Examples:\n"
                f"Q: How many movies directed by Nolan? -> aggregator=COUNT, join_type=SINGLE, has_type_filter=true\n"
                f"Q: Where were JK Rowling and Einstein born? -> aggregator=NONE, join_type=INTERSECTION, has_type_filter=false\n"
                f"Q: Which organizations were founded in 1990? -> aggregator=NONE, join_type=SINGLE, has_type_filter=true\n"
                f"Q: List 10 countries by population -> aggregator=ORDER_BY_DESC, join_type=SINGLE, has_type_filter=true\n"
                f"Q: How many people study at California universities? -> aggregator=SUM, join_type=SINGLE, has_type_filter=true"
            ),
        },
    ]
    response = _chat(client, messages, model=model)
    analysis = _extract_json(response)

    return {
        "entities":        analysis.get("entities", []),
        "concepts":        analysis.get("concepts", []),
        "answer_type":     analysis.get("answer_type", "resource"),
        "aggregator":      analysis.get("aggregator", "NONE"),
        "join_type":       analysis.get("join_type", "SINGLE"),
        "has_type_filter": analysis.get("has_type_filter", False),
    }


def entity_linker_node(state: KGQAState, redis_el, client, model: Optional[str] = None) -> dict:
    """Entity Linker node: link entity mentions to DBpedia resource URIs via Redis.

    Corresponds to KGQAAgent._link_entities() + KGQAAgent._disambiguate_entity().
    Returns partial state update.
    """
    entities = state["entities"]
    question = state["question"]
    model = model or state.get("model") or DEFAULT_MODEL
    linked = {}

    for entity in entities:
        if redis_el is None:
            uri = "http://dbpedia.org/resource/" + entity.replace(" ", "_")
            linked[entity] = [{"uri": uri, "score": 1.0, "source": "heuristic"}]
            continue

        results = redis_el.lookup(entity, top_k=5, thr=0.01)
        if len(results) > 0:
            entries = []
            for idx, row in results.iterrows():
                uri = idx if isinstance(idx, str) else row.name
                if not uri.startswith("http"):
                    uri = "http://dbpedia.org/resource/" + uri
                entries.append({"uri": uri, "score": round(row["score"], 4), "source": "redis"})
            if question and len(entries) > 1:
                entries = _disambiguate(client, question, entity, entries, model=model)
            linked[entity] = entries
        else:
            uri = "http://dbpedia.org/resource/" + entity.replace(" ", "_")
            linked[entity] = [{"uri": uri, "score": 1.0, "source": "heuristic"}]

    return {"linked_entities": linked}


def ontology_explorer_node(state: KGQAState) -> dict:
    """Ontology Explorer + Schema Introspector node: look up relevant ontology
    terms for each concept and enrich with rdfs:domain and rdfs:range.

    Corresponds to KGQAAgent._lookup_ontology().
    Returns partial state update.
    """
    concepts = state["concepts"]
    ontology = {}
    for concept in concepts:
        results = lookup_term(concept)
        results = schema_introspector.enrich(results)
        ontology[concept] = results
    return {"ontology_terms": ontology}


def query_builder_node(state: KGQAState, client, model: Optional[str] = None) -> dict:
    """Query Builder node: generate a SPARQL query from all available context.

    Corresponds to KGQAAgent._generate_sparql().
    Returns partial state update.
    """
    question        = state["question"]
    linked_entities = state["linked_entities"]
    ontology_terms  = state["ontology_terms"]
    model = model or state.get("model") or DEFAULT_MODEL

    # Reuse formatting helpers from KGQAAgent
    entity_context   = _format_entity_context(linked_entities)
    ontology_context = _format_ontology_context(ontology_terms)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Generate a SPARQL query for this question.\n\n"
                f"Question: {question}\n\n"
                f"Answer type: {state.get('answer_type', 'unknown')}\n"
                f"Aggregator: {state.get('aggregator', 'NONE')}\n"
                f"Join type: {state.get('join_type', 'SINGLE')}\n"
                f"Has type filter: {state.get('has_type_filter', False)}\n\n"
                f"Linked entities:\n{entity_context}\n"
                f"Relevant ontology terms:\n{ontology_context}\n"
                f"Output ONLY the SPARQL query."
            ),
        },
    ]
    response = _chat(client, messages, model=model)
    sparql = _extract_sparql(response)

    return {
        "sparql":        sparql,
        "attempts":      0,
        "swap_attempted": False,
    }


def query_executor_node(state: KGQAState, client, model: Optional[str] = None) -> dict:
    """Query Executor node: execute the SPARQL query with deterministic dbo->dbp
    fallback, then LLM revision if still failing.

    Corresponds to KGQAAgent._verify_and_revise().
    Returns partial state update.
    """
    question        = state["question"]
    sparql          = state["sparql"]
    linked_entities = state["linked_entities"]
    ontology_terms  = state["ontology_terms"]
    attempts        = state.get("attempts", 0)
    swap_attempted  = state.get("swap_attempted", False)
    model = model or state.get("model") or DEFAULT_MODEL

    # Step 1: execute current sparql
    exec_result = execute_sparql(sparql)
    needs_fix, reason = _needs_revision(exec_result)

    if not needs_fix:
        return {
            "exec_result":   exec_result,
            "attempts":      attempts + 1,
            "swap_attempted": swap_attempted,
            "sparql":        sparql,
        }

    # Step 2: deterministic dbo->dbp swap (only once)
    if not swap_attempted:
        swapped = _swap_dbo_to_dbp(sparql)
        if swapped != sparql:
            logger.info("Retrying with dbo->dbp namespace swap...")
            swap_result = execute_sparql(swapped)
            swap_needs_fix, _ = _needs_revision(swap_result)
            if not swap_needs_fix:
                return {
                    "exec_result":    swap_result,
                    "sparql":         swapped,
                    "attempts":       attempts + 1,
                    "swap_attempted": True,
                }
            # Swap didn't help — continue with swapped query for LLM revision
            sparql = swapped

    # Step 3: LLM revision
    entity_context   = _format_entity_context(linked_entities)
    ontology_context = _format_ontology_context(ontology_terms)

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
    response = _chat(client, messages, model=model)
    revised_sparql = _extract_sparql(response)
    revised_result = execute_sparql(revised_sparql)

    return {
        "sparql":         revised_sparql,
        "exec_result":    revised_result,
        "attempts":       attempts + 1,
        "swap_attempted": True,
    }


# ─── Formatting helpers (module-level for use by both node functions and KGQAAgent) ───

def _format_entity_context(linked_entities: dict) -> str:
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


def _format_ontology_context(ontology_terms: dict) -> str:
    """Format ontology lookup results for LLM prompts.

    Separates Properties (use as predicates) from Classes (use for rdf:type only).
    Shows rdfs:domain and rdfs:range for each property candidate.
    """
    out = ""
    for concept, results in ontology_terms.items():
        out += f"Concept \"{concept}\":\n"
        properties = [r for r in results if not r["uri"].split("/")[-1][0].isupper()]
        classes    = [r for r in results if r["uri"].split("/")[-1][0].isupper()]

        if properties:
            out += "  Properties (use as predicates):\n"
            for r in properties:
                uri    = r["uri"]
                conf   = r.get("confidence_pct", round(r.get("score", 0) * 100, 1))
                domain = r.get("domain") or "unknown"
                range_ = r.get("range") or "unknown"
                ns     = "dbo:" if "ontology" in uri else "dbp:"
                out += f"    - {uri} (domain: {domain}, range: {range_}, score: {conf}%) — use {ns}\n"

        if classes:
            out += "  Classes (use only for rdf:type constraints, NOT as predicates):\n"
            for r in classes:
                uri  = r["uri"]
                conf = r.get("confidence_pct", round(r.get("score", 0) * 100, 1))
                out += f"    - {uri} (score: {conf}%)\n"

    return out


def _disambiguate(client, question: str, mention: str, candidates: list, model: Optional[str] = None) -> list:
    """Use LLM to pick the most contextually appropriate entity from Redis candidates.

    Extracted from KGQAAgent._disambiguate_entity() for use in entity_linker_node.
    Falls back to original list on failure.
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
        response = client.chat.completions.create(
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


# ─── KGQAAgent ────────────────────────────────────────────────────────────────

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
                    f"Output your analysis as JSON with these exact keys:\n"
                    f"- entities: list of named entities (people, places, organisations, works)\n"
                    f"- answer_type: one of resource, literal, count, boolean, list\n"
                    f"- concepts: list of relationship/property keywords\n"
                    f"- aggregator: one of NONE, COUNT, SUM, GROUP_BY, ORDER_BY_DESC, ORDER_BY_ASC\n"
                    f"  (NONE = plain SELECT, COUNT = how many, SUM = total of values, GROUP_BY = list ranked by count, ORDER_BY_DESC/ASC = top-N or ranked list)\n"
                    f"- join_type: one of INTERSECTION, UNION, SINGLE\n"
                    f"  (INTERSECTION = question asks what two entities have IN COMMON using 'and', UNION = question asks about either entity using 'or', SINGLE = normal single-entity question)\n"
                    f"- has_type_filter: true if the question explicitly asks for a category like 'which movies', 'list countries', 'how many companies' etc, false otherwise\n\n"
                    f"Examples:\n"
                    f"Q: How many movies directed by Nolan? -> aggregator=COUNT, join_type=SINGLE, has_type_filter=true\n"
                    f"Q: Where were JK Rowling and Einstein born? -> aggregator=NONE, join_type=INTERSECTION, has_type_filter=false\n"
                    f"Q: Which organizations were founded in 1990? -> aggregator=NONE, join_type=SINGLE, has_type_filter=true\n"
                    f"Q: List 10 countries by population -> aggregator=ORDER_BY_DESC, join_type=SINGLE, has_type_filter=true\n"
                    f"Q: How many people study at California universities? -> aggregator=SUM, join_type=SINGLE, has_type_filter=true"
                ),
            },
        ]
        response = _chat(self.client, messages, model=model)
        return _extract_json(response)

    def _link_entities(self, entities, question=None, model=None):
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
                    entries = self._disambiguate_entity(question, entity, entries, model=model)
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
        """Step 3: Look up relevant ontology terms for each concept.

        Each concept is looked up in the Nomic index, then enriched with
        rdfs:domain and rdfs:range from the OWL ontology via Schema Introspector.
        """
        ontology = {}
        for concept in concepts:
            results = lookup_term(concept)
            results = schema_introspector.enrich(results)
            ontology[concept] = results
        return ontology

    def _format_entity_context(self, linked_entities):
        """Format linked entities for LLM prompts."""
        return _format_entity_context(linked_entities)

    def _format_ontology_context(self, ontology_terms):
        """Format ontology lookup results for LLM prompts.

        Separates Properties (use as predicates) from Classes (use for rdf:type only).
        Shows rdfs:domain and rdfs:range for each property candidate from Schema Introspector.
        """
        return _format_ontology_context(ontology_terms)

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
                    f"Answer type: {analysis.get('answer_type', 'unknown')}\n"
                    f"Aggregator: {analysis.get('aggregator', 'NONE')}\n"
                    f"Join type: {analysis.get('join_type', 'SINGLE')}\n"
                    f"Has type filter: {analysis.get('has_type_filter', False)}\n\n"
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
        """Execute query with deterministic dbo->dbp fallback then LLM revision.

        Steps:
          1. Execute original dbo: query.
          2. If 0 results: swap dbo->dbp deterministically and retry.
          3. If still 0 results: LLM revision up to MAX_RETRIES times.
        """
        # Step 1: initial execution
        result = execute_sparql(sparql)
        needs_fix, reason = _needs_revision(result)

        if not needs_fix:
            return sparql, result, 1

        # Step 2: deterministic dbo->dbp swap
        swapped = _swap_dbo_to_dbp(sparql)
        if swapped != sparql:
            logger.info("Retrying with dbo->dbp namespace swap...")
            result = execute_sparql(swapped)
            needs_fix, reason = _needs_revision(result)
            if not needs_fix:
                return swapped, result, 2
            sparql = swapped  # continue LLM revision from swapped query

        # Step 3: LLM-based revision
        for attempt in range(MAX_RETRIES):
            logger.info(f"LLM revision attempt {attempt + 1}: {reason}...")
            sparql = self._revise_sparql(
                question, sparql, result, linked_entities, ontology_terms, analysis, model=model
            )
            logger.info(f"Revised SPARQL: {sparql}")
            result = execute_sparql(sparql)
            needs_fix, reason = _needs_revision(result)
            if not needs_fix:
                return sparql, result, attempt + 3

        logger.info(f"Max retries reached, returning last query despite {reason}")
        return sparql, result, MAX_RETRIES + 2

    def _answer_sequential(self, question, model=None):
        """Sequential pipeline fallback used by CorporateKGQAAgent subclass.

        CorporateKGQAAgent overrides _link_entities, _lookup_ontology,
        _generate_sparql, _revise_sparql and _verify_and_revise. These overrides
        are bypassed when using build_graph() since the graph calls standalone
        node functions directly. This method preserves the old sequential flow
        so subclass overrides continue to work correctly.
        """
        logger.info(f"Processing question (sequential): {question}")

        analysis = self._analyse_question(question, model=model)
        logger.info(f"Analysis: {analysis}")

        entities = analysis.get("entities", [])
        concepts = analysis.get("concepts", [])

        linked_entities = self._link_entities(entities, question=question, model=model)
        logger.info(f"Linked entities: {linked_entities}")

        ontology_terms = self._lookup_ontology(concepts)
        logger.info(f"Ontology terms found for {len(ontology_terms)} concepts")

        sparql = self._generate_sparql(question, linked_entities, ontology_terms, analysis, model=model)
        logger.info(f"Generated SPARQL: {sparql}")

        sparql, exec_result, attempts = self._verify_and_revise(
            question, sparql, linked_entities, ontology_terms, analysis, model=model
        )
        if attempts > 0:
            logger.info(f"Query revised {attempts} time(s)")

        return sparql

    def answer(self, question, model=None):
        """Full pipeline: question -> SPARQL query.

        For KGQAAgent instances uses the compiled LangGraph graph.
        For subclasses (e.g. CorporateKGQAAgent) falls back to the sequential
        pipeline so their method overrides are respected.
        """
        if type(self) is KGQAAgent:
            logger.info(f"Processing question (graph): {question}")
            graph = build_graph(redis_el=self.redis_el, model=model)
            result = graph.invoke({"question": question, "model": model})
            return result["sparql"]
        return self._answer_sequential(question, model=model)

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
        linked_entities = self._link_entities(entities, question=question, model=model)
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


# ─── LangGraph ────────────────────────────────────────────────────────────────

def build_graph(redis_el=None, model: Optional[str] = None):
    """Build and compile the LangGraph StateGraph for the KGQA pipeline.

    Wires all node functions into a linear StateGraph:
      START -> planner -> entity_linker -> ontology_explorer
             -> query_builder -> query_executor -> END

    Node functions that require runtime dependencies (LLM client, Redis)
    are wrapped with functools.partial to bind those dependencies at
    graph construction time.

    Args:
        redis_el: RedisEntityLinking instance (or None for heuristic fallback)
        model:    OpenRouter model ID to use (or None for DEFAULT_MODEL)

    Returns:
        A compiled LangGraph graph ready for graph.invoke({"question": ...})
    """
    client = _get_llm_client()

    # Bind runtime dependencies to node functions via partial
    bound_planner = functools.partial(planner_node, client=client, model=model)
    bound_entity_linker = functools.partial(
        entity_linker_node, redis_el=redis_el, client=client, model=model
    )
    bound_query_builder = functools.partial(query_builder_node, client=client, model=model)
    bound_query_executor = functools.partial(query_executor_node, client=client, model=model)

    # Build the graph
    graph = StateGraph(KGQAState)

    # Add nodes
    graph.add_node("planner",          bound_planner)
    graph.add_node("entity_linker",    bound_entity_linker)
    graph.add_node("ontology_explorer", ontology_explorer_node)
    graph.add_node("query_builder",    bound_query_builder)
    graph.add_node("query_executor",   bound_query_executor)

    # Add linear edges
    graph.set_entry_point("planner")
    graph.add_edge("planner",           "entity_linker")
    graph.add_edge("entity_linker",     "ontology_explorer")
    graph.add_edge("ontology_explorer", "query_builder")
    graph.add_edge("query_builder",     "query_executor")
    graph.add_edge("query_executor",    END)

    return graph.compile()
