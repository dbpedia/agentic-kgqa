#!/usr/bin/env python3
"""Agentic KGQA for the corporate (CK25) dataset.

Uses a pv:-namespace system prompt and embedding-based instance lookup
(no Redis dependency). Subclasses KGQAAgent to reuse the verify-and-revise loop.
"""

import os
import logging

from src.agent import KGQAAgent, execute_sparql
from src.corporate_lookup import lookup_vocab_term, lookup_instance

dotenv_loaded = False
try:
    import dotenv
    dotenv.load_dotenv(override=True)
    dotenv_loaded = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

CORPORATE_SPARQL_ENDPOINT = os.getenv(
    "CORPORATE_SPARQL_ENDPOINT", "http://localhost:7879/query"
)

CORPORATE_SYSTEM_PROMPT = """\
You are a SPARQL query generation agent for a corporate product knowledge graph.
Your job is to translate natural language questions into valid SPARQL queries.

## Vocabulary

All schema terms use the `pv:` namespace: http://ld.company.org/prod-vocab/
All instance URIs use: http://ld.company.org/prod-instances/
Countries may be referenced via DBpedia: http://dbpedia.org/resource/

## Classes
- pv:Product, pv:Hardware (subclass of Product), pv:Service (subclass of Product)
- pv:ProductCategory
- pv:Employee, pv:Manager (subclass of Employee)
- pv:Department
- pv:Supplier
- pv:BillOfMaterial, pv:BomPart
- pv:Price

## Key Properties
- pv:name, rdfs:label — human-readable name/label of an instance
- pv:hasManager, pv:hasDirectReport — employee hierarchy
- pv:memberOf — employee belongs to a department
- pv:responsibleFor — department is responsible for a product
- pv:hasCategory — product belongs to a category
- pv:areaOfExpertise — employee's product category expertise
- pv:hasSupplier — product has a supplier
- pv:hasProductManager — product has a product manager (employee)
- pv:compatibleProduct — product is compatible with another product
- pv:eligibleFor — service is eligible for a product
- pv:hasBomPart — BOM includes a BOM part; pv:hasPart — BOM part includes a product
- pv:price → pv:amount (decimal) + pv:currency (string)
- pv:addressCountry, pv:addressLocality, pv:country (→ DBpedia Country URI)
- pv:reliabilityIndex, pv:weight_g, pv:height_mm, pv:width_mm, pv:depth_mm

## Rules
- ALWAYS use full URIs in angle brackets. NEVER use PREFIX declarations or prefixed names.
  CORRECT: <http://ld.company.org/prod-vocab/Employee>
  WRONG:   pv:Employee
- ALWAYS use SELECT DISTINCT for queries returning resource URIs or text values.
- For count questions: SELECT DISTINCT COUNT(?var) WHERE { ... } (no AS alias).
- For boolean questions: ASK WHERE { ... }
- For "top N" or ordered questions: ORDER BY ... LIMIT N
- rdf:type URI: <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
- rdfs:label URI: <http://www.w3.org/2000/01/rdf-schema#label>
- Use the EXACT instance URIs from entity linking results — they may contain URL-encoded
  characters (e.g., %40 for @). Preserve them exactly as given.
- Do NOT invent property or class URIs — use only what the ontology lookup returns.
- Output ONLY the SPARQL query, no explanations.

## Examples

Example 1 — "What is the email of Elena Herzog?":
```sparql
SELECT DISTINCT ?email WHERE {
  <http://ld.company.org/prod-instances/empl-Elena.Herzog%40company.org> <http://ld.company.org/prod-vocab/email> ?email .
}
```

Example 2 — "How many hardware products are in the catalog?":
```sparql
SELECT DISTINCT COUNT(?hw) WHERE {
  ?hw <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ld.company.org/prod-vocab/Hardware> .
}
```

Example 3 — "Which employees are members of the Engineering department?":
```sparql
SELECT DISTINCT ?name WHERE {
  ?emp <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ld.company.org/prod-vocab/Employee> .
  ?emp <http://ld.company.org/prod-vocab/memberOf> <http://ld.company.org/prod-instances/dept-73191> .
  ?emp <http://www.w3.org/2000/01/rdf-schema#label> ?name .
}
```

Example 4 — "List all suppliers from Germany":
```sparql
SELECT DISTINCT ?name WHERE {
  ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ld.company.org/prod-vocab/Supplier> .
  ?s <http://ld.company.org/prod-vocab/addressCountry> "Germany" .
  ?s <http://www.w3.org/2000/01/rdf-schema#label> ?name .
}
```

Example 5 — "Who manages Arendt Beitel?":
```sparql
SELECT DISTINCT ?name WHERE {
  <http://ld.company.org/prod-instances/empl-Arendt.Beitel%40company.org> <http://ld.company.org/prod-vocab/hasManager> ?mgr .
  ?mgr <http://www.w3.org/2000/01/rdf-schema#label> ?name .
}
```
"""


class CorporateKGQAAgent(KGQAAgent):
    """KGQA agent for the corporate dataset.

    Overrides entity linking (uses embedding-based instance lookup)
    and ontology lookup (uses corporate vocab vectors). Shares the
    verify-and-revise loop from KGQAAgent.
    """

    def __init__(self):
        # Skip Redis init from parent
        from src.agent import _get_llm_client
        self.client = _get_llm_client()
        self.redis_el = None  # not used

    # Override: use embedding-based instance lookup
    def _link_entities(self, entities):
        linked = {}
        for entity in entities:
            candidates = lookup_instance(entity, k=3)
            if candidates:
                linked[entity] = candidates
            else:
                linked[entity] = [{"uri": entity, "score": 0.0, "source": "unknown"}]
        return linked

    # Override: use corporate vocab lookup
    def _lookup_ontology(self, concepts):
        ontology = {}
        for concept in concepts:
            ontology[concept] = lookup_vocab_term(concept, k=5)
        return ontology

    # Override: use corporate system prompt
    def _analyse_question(self, question, model=None):
        from src.agent import _chat, _extract_json
        messages = [
            {"role": "system", "content": CORPORATE_SYSTEM_PROMPT},
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

    # Override: use corporate system prompt
    def _generate_sparql(self, question, linked_entities, ontology_terms, analysis, model=None):
        from src.agent import _chat, _extract_sparql
        entity_context = self._format_entity_context(linked_entities)
        ontology_context = self._format_ontology_context_corporate(ontology_terms)

        messages = [
            {"role": "system", "content": CORPORATE_SYSTEM_PROMPT},
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

    # Override: corporate endpoint + corporate system prompt for revision
    def _revise_sparql(self, question, sparql, exec_result, linked_entities, ontology_terms, analysis, model=None):
        import json
        from src.agent import _chat, _extract_sparql
        entity_context = self._format_entity_context(linked_entities)
        ontology_context = self._format_ontology_context_corporate(ontology_terms)

        if exec_result["type"] == "error":
            exec_summary = f"Error: {exec_result['message']}"
            problem = "a SPARQL error"
        elif exec_result["type"] == "select" and exec_result["total"] == 0:
            exec_summary = "The query executed successfully but returned 0 results."
            problem = "0 results"
        else:
            exec_summary = json.dumps(exec_result, default=str)
            problem = "unexpected results"

        prompt = (
            f"The following SPARQL query for the corporate dataset returned {problem}.\n\n"
            f"Question: {question}\n\n"
            f"Failed query:\n{sparql}\n\n"
            f"Execution result: {exec_summary}\n\n"
            f"Linked entities:\n{entity_context}\n"
            f"Relevant ontology terms:\n{ontology_context}\n\n"
            f"Rules:\n"
            f"- Use full URIs only, no PREFIX declarations\n"
            f"- Try removing rdf:type constraints first\n"
            f"- Try alternative properties from the ontology lookup\n"
            f"- Output ONLY the revised SPARQL query."
        )
        messages = [{"role": "user", "content": prompt}]
        response = _chat(self.client, messages, model=model)
        return _extract_sparql(response)

    # Override: use corporate SPARQL endpoint
    def _verify_and_revise(self, question, sparql, linked_entities, ontology_terms, analysis, model=None):
        from src.agent import _needs_revision, MAX_RETRIES
        for attempt in range(MAX_RETRIES + 1):
            result = execute_sparql(sparql, endpoint=CORPORATE_SPARQL_ENDPOINT)
            needs_fix, reason = _needs_revision(result)

            if not needs_fix:
                return sparql, result, attempt

            if attempt < MAX_RETRIES:
                sparql = self._revise_sparql(
                    question, sparql, result, linked_entities, ontology_terms, analysis, model=model
                )
            else:
                logger.info(f"Max retries reached, returning last query despite {reason}")

        return sparql, result, MAX_RETRIES

    def _format_ontology_context_corporate(self, ontology_terms):
        """Format corporate vocab lookup results for LLM prompts."""
        out = ""
        for concept, results in ontology_terms.items():
            out += f'Concept "{concept}":\n'
            for r in results:
                out += f"  - {r['uri']} ({r['type']}, score: {r['score']})\n"
        return out

    def answer_stream(self, question, model=None):
        """Streaming pipeline that yields (step_name, data) tuples."""
        from src.agent import _needs_revision, MAX_RETRIES
        yield ("question", {"question": question})

        yield ("step_start", {"step": "analyse", "label": "Analysing question..."})
        analysis = self._analyse_question(question, model=model)
        yield ("analyse", analysis)

        entities = analysis.get("entities", [])
        concepts = analysis.get("concepts", [])

        yield ("step_start", {"step": "entity_linking", "label": "Looking up instances..."})
        linked_entities = self._link_entities(entities)
        linked_serialisable = {}
        for mention, candidates in linked_entities.items():
            linked_serialisable[mention] = [
                {k: float(v) if hasattr(v, "item") else v for k, v in c.items()}
                for c in candidates
            ]
        yield ("entity_linking", linked_serialisable)

        yield ("step_start", {"step": "ontology_lookup", "label": "Looking up vocab terms..."})
        ontology_terms = self._lookup_ontology(concepts)
        yield ("ontology_lookup", ontology_terms)

        yield ("step_start", {"step": "sparql_generation", "label": "Generating SPARQL query..."})
        sparql = self._generate_sparql(question, linked_entities, ontology_terms, analysis, model=model)
        yield ("sparql", {"query": sparql})

        for attempt in range(MAX_RETRIES + 1):
            step_id = f"execution_{attempt}"
            label = "Testing query against corporate endpoint..."
            if attempt > 0:
                label = f"Testing revised query (attempt {attempt + 1})..."
            yield ("step_start", {"step": step_id, "label": label})

            result = execute_sparql(sparql, endpoint=CORPORATE_SPARQL_ENDPOINT)
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
