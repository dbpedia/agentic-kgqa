#!/usr/bin/env python3
"""Validator node: inspects execution result and decides what to do next.

Architecture:
  - Query Executor runs SPARQL + class-safe dbo->dbp swap
  - If result is still bad, Validator takes over
  - Validator probes live DBpedia (dbp: namespace only) to find actual
    properties for the subject entities in the failing query
  - Passes probe results back to Query Builder for a smarter retry
  - MAX_RETRIES = 2 (Validator retries, not executor retries)
  - After MAX_RETRIES, gives up and returns best effort

Probe strategy (two stages per subject x concept pair):
  Stage 1 -- keyword-filtered probe: search dbp: properties whose name
             contains a concept keyword. Fast, precise.
  Stage 2 -- if Stage 1 finds nothing for that subject, fall back to an
             UNFILTERED probe: list ALL dbp: properties for that subject
             (capped at UNFILTERED_PROBE_LIMIT) and let the LLM reason over
             the full list.

  Results are grouped by SUBJECT FIRST, then concept, so the LLM can
  clearly see which property belongs to which entity -- critical for
  multi-entity (INTERSECTION/UNION) queries.

Actions:
  "pass"                -- result is good
  "retry_query_builder" -- probe found data, retry QB with enriched context
  "give_up"             -- max retries reached or probe found nothing
"""

import re

from src.sparql_client import execute, needs_fix

MAX_VALIDATOR_RETRIES  = 2
UNFILTERED_PROBE_LIMIT = 30

# Feature flag: two-hop probe chaining. When True, for num_hops>=2 questions
# the Validator also resolves the intermediate entity (by executing just the
# first-hop triple) and probes THAT entity too, not just the original subject.
# Set to False to instantly revert to subject-only probing if this regresses
# results -- no other code needs to change.
ENABLE_TWO_HOP_PROBE_CHAINING = True


# ─── Agentic Probe ────────────────────────────────────────────────────────────

def _probe_keyword(subject: str, concept: str) -> list:
    """Stage 1: keyword-filtered dbp: probe for a single subject x concept pair."""
    keywords = [w.lower() for w in re.split(r"\s+", concept) if len(w) > 2]
    if not keywords:
        return []

    filter_parts = " || ".join(
        f'CONTAINS(LCASE(STR(?p)), "{kw}")' for kw in keywords
    )
    query = f"""
SELECT DISTINCT ?p ?o WHERE {{
  <{subject}> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
  FILTER({filter_parts})
}}
LIMIT 10
"""
    return _extract_props(execute(query, timeout=10))


def _probe_unfiltered(subject: str) -> list:
    """Stage 2 fallback: list ALL dbp: properties for a subject."""
    query = f"""
SELECT DISTINCT ?p ?o WHERE {{
  <{subject}> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
}}
LIMIT {UNFILTERED_PROBE_LIMIT}
"""
    return _extract_props(execute(query, timeout=10))


def _extract_props(result: dict) -> list:
    """Extract deduplicated (uri, value) pairs from a SPARQL SELECT result."""
    props = []
    seen  = set()
    if result["type"] == "select":
        for row in result.get("rows", []):
            p    = row.get("p", "")
            o    = row.get("o", "")
            name = p.split("/")[-1]
            if p and name not in seen:
                seen.add(name)
                props.append((p, o))
    return props


def _probe_dbpedia(subjects: list, concepts: list) -> dict:
    """Run the two-stage probe for every subject x concept pair.

    Returns: {subject: {concept: [(property_uri, sample_value), ...]}}
    """
    found = {}
    for subject in subjects:
        subject_found = {}
        for concept in concepts:
            print(f"  [VALIDATOR:PROBE] subject={subject.split('/')[-1]} concept='{concept}'")
            props = _probe_keyword(subject, concept)
            if props:
                for p, o in props:
                    print(f"  [VALIDATOR:PROBE]   found {p.split('/')[-1]} = {str(o)[:40]}")
            else:
                print(f"  [VALIDATOR:PROBE]   keyword probe found nothing, trying unfiltered fallback...")
                props = _probe_unfiltered(subject)
                if props:
                    print(f"  [VALIDATOR:PROBE]   unfiltered probe found {len(props)} properties")
                else:
                    print(f"  [VALIDATOR:PROBE]   unfiltered probe also found nothing")
            if props:
                subject_found[concept] = props
        if subject_found:
            found[subject] = subject_found
    return found


def _format_probe_context(probe_results: dict) -> str:
    """Format probe results for the Query Builder retry prompt.

    Grouped by subject first so the LLM can see which property belongs to
    which entity.
    """
    if not probe_results:
        return ""
    out = "\nACTUAL DBpedia properties found via live probe (use these instead of guessing):\n"
    for subject, by_concept in probe_results.items():
        subject_name = subject.split("/")[-1].replace("_", " ")
        out += f'  Subject <{subject}> ("{subject_name}"):\n'
        seen = set()
        for concept, props in by_concept.items():
            out += f'    Concept "{concept}":\n'
            for p, o in props:
                name = p.split("/")[-1]
                if name not in seen:
                    seen.add(name)
                    out += f"      - {p}  (example value: {str(o)[:60]})\n"
    return out


def _extract_subjects(sparql: str) -> list:
    """Extract DBpedia resource URIs from a SPARQL query."""
    uris   = re.findall(r"<(http://dbpedia\.org/resource/[^>]+)>", sparql)
    seen   = set()
    result = []
    for u in uris:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# ─── Two-hop probe chaining ────────────────────────────────────────────────────

_FIRST_HOP_TRIPLE_RE = re.compile(
    r"<(http://dbpedia\.org/resource/[^>]+)>\s+"   # subject: literal entity URI
    r"<(http://dbpedia\.org/[^>]+)>\s+"             # predicate URI
    r"\?(\w+)\s*\."                                  # object: a variable
)


def _find_intermediate_hops(sparql: str) -> list:
    """Find first-hop triples whose object variable is later used as the
    SUBJECT of another triple -- i.e. the variable is an intermediate hop,
    not the final answer variable.

    Returns [(subject_uri, predicate_uri, variable_name), ...]
    """
    first_hop_triples = _FIRST_HOP_TRIPLE_RE.findall(sparql)
    if not first_hop_triples:
        return []

    subject_var_re = re.compile(r"\?(\w+)\s+<http://dbpedia\.org/")
    vars_used_as_subject = set(subject_var_re.findall(sparql))

    return [
        (subj, pred, var)
        for subj, pred, var in first_hop_triples
        if var in vars_used_as_subject
    ]


def _resolve_intermediate_entities(sparql: str) -> list:
    """For a two-hop (or more) query, resolve what the intermediate variable
    actually binds to by executing just the first-hop triple standalone.

    This lets the Validator probe the REAL intermediate entity (e.g. the
    country Oxford is in) instead of only ever probing the original subject
    (Oxford itself) -- the original subject-only probe can never discover a
    second-hop property that only exists on the intermediate entity.

    Returns a deduplicated list of resolved intermediate resource URIs.
    """
    hops = _find_intermediate_hops(sparql)
    resolved, seen = [], set()

    for subj, pred, var in hops:
        probe_query = f"""
SELECT DISTINCT ?{var} WHERE {{
  <{subj}> <{pred}> ?{var} .
}}
LIMIT 5
"""
        result = execute(probe_query, timeout=10)
        if result["type"] == "select":
            for row in result.get("rows", []):
                val = row.get(var, "")
                if val.startswith("http://dbpedia.org/resource/") and val not in seen:
                    seen.add(val)
                    resolved.append(val)

    return resolved


def _strip_disambiguation_suffix(uri: str):
    """Strip disambiguation suffix from a DBpedia resource URI.

    If the URI contains '_(' (e.g. Oddworld_(series), Back_to_Black_(album)),
    this is a DBpedia disambiguation pattern. The bare title without the suffix
    is often the canonical resource that actually holds the data.

    Returns the cleaned URI string if a suffix was found, None otherwise.
    """
    if "/resource/" not in uri:
        return None
    prefix, resource = uri.split("/resource/", 1)
    idx = resource.find("_(")
    if idx == -1:
        return None
    cleaned = resource[:idx]
    return f"{prefix}/resource/{cleaned}"


# ─── Main validate function ────────────────────────────────────────────────────

def validate(state: dict) -> dict:
    """Inspect the execution result and decide the next action."""
    sparql             = state.get("sparql_final", "")
    result             = state.get("exec_result", {"type": "error", "message": "no result"})
    validator_attempts = state.get("validator_attempts", 0)
    concepts           = state.get("concepts", [])

    fix, reason = needs_fix(result)

    # Rule 1: result is good
    if not fix:
        print(f"[VALIDATOR] Result is good ({_summarise(result)}). Action: PASS")
        return {**state,
                "validator_action":   "pass",
                "validator_reason":   f"good result: {_summarise(result)}",
                "probe_results":      state.get("probe_results", {}),
                "validator_attempts": validator_attempts}

    # Rule 2: max retries reached
    if validator_attempts >= MAX_VALIDATOR_RETRIES:
        print(f"[VALIDATOR] Max retries ({MAX_VALIDATOR_RETRIES}) reached. Action: GIVE_UP")
        return {**state,
                "validator_action":   "give_up",
                "validator_reason":   f"max retries reached after {validator_attempts} attempts",
                "probe_results":      state.get("probe_results", {}),
                "validator_attempts": validator_attempts}

    # Rule 3: endpoint down
    if result.get("type") == "error":
        msg = result.get("message", "").lower()
        if any(k in msg for k in ["timeout", "connection", "refused", "urlopen"]):
            print(f"[VALIDATOR] Endpoint unavailable. Action: GIVE_UP")
            return {**state,
                    "validator_action":   "give_up",
                    "validator_reason":   f"endpoint error: {result.get('message','')[:80]}",
                    "probe_results":      state.get("probe_results", {}),
                    "validator_attempts": validator_attempts}

    # Rule 4: run agentic probe and retry Query Builder
    print(f"[VALIDATOR] Result needs fix ({reason}). Running agentic probe...")
    subjects = _extract_subjects(sparql)

    # Two-hop probe chaining: for multi-hop questions, also resolve and probe
    # the intermediate entity, not just the original subject. This directly
    # addresses cases where the second-hop predicate only exists on the
    # intermediate entity (e.g. Oxford -> country -> ?country -> areaTotal),
    # which subject-only probing could never discover.
    if ENABLE_TWO_HOP_PROBE_CHAINING and state.get("num_hops", 1) >= 2:
        intermediate_uris = _resolve_intermediate_entities(sparql)
        for uri in intermediate_uris:
            if uri not in subjects:
                subjects.append(uri)
                print(f"  [VALIDATOR] Resolved intermediate entity for two-hop probe: {uri.split('/')[-1]}")

    if not subjects:
        print(f"[VALIDATOR] No subjects found in SPARQL. Action: GIVE_UP")
        return {**state,
                "validator_action":   "give_up",
                "validator_reason":   "no DBpedia resource URIs found in SPARQL to probe",
                "probe_results":      state.get("probe_results", {}),
                "validator_attempts": validator_attempts + 1}

    probe_results = _probe_dbpedia(subjects, concepts)

    if not probe_results:
        # Rule 4.5: dead URI detection
        # If ALL subjects returned zero properties from the unfiltered probe,
        # the URI itself is likely wrong (dead resource). Check for DBpedia
        # disambiguation suffixes like _(series), _(film), _(band) etc.
        # If found, strip the suffix and retry with the cleaned URI.
        cleaned_entities = {}
        for subject in subjects:
            cleaned = _strip_disambiguation_suffix(subject)
            if cleaned:
                print(f"[VALIDATOR] Dead URI detected: {subject.split('/')[-1]}")
                print(f"[VALIDATOR] Stripped suffix -> {cleaned.split('/')[-1]}")
                linked = state.get("linked_entities", {})
                for mention, candidates in linked.items():
                    if candidates and candidates[0]["uri"] == subject:
                        cleaned_entities[mention] = [
                            {"uri": cleaned, "score": 1.0, "source": "validator_cleaned"}
                        ] + candidates[1:]

        if cleaned_entities:
            updated_linked = {**state.get("linked_entities", {}), **cleaned_entities}
            print(f"[VALIDATOR] Retrying with cleaned entity URIs. Action: RETRY_QUERY_BUILDER")
            return {**state,
                    "linked_entities":   updated_linked,
                    "probe_context":      "",
                    "probe_results":      {},
                    "validator_action":   "retry_query_builder",
                    "validator_reason":   "dead URI detected and stripped disambiguation suffix",
                    "validator_attempts": validator_attempts + 1}

        print(f"[VALIDATOR] Probe found nothing. Action: GIVE_UP")
        return {**state,
                "validator_action":   "give_up",
                "validator_reason":   "agentic probe found no matching dbp: properties",
                "probe_results":      {},
                "validator_attempts": validator_attempts + 1}

    probe_context = _format_probe_context(probe_results)
    n_subjects = len(probe_results)
    print(f"[VALIDATOR] Probe found properties for {n_subjects} subject(s). Action: RETRY_QUERY_BUILDER")

    return {**state,
            "validator_action":   "retry_query_builder",
            "validator_reason":   "probe found dbp: properties; retrying Query Builder with grounded context",
            "probe_results":      probe_results,
            "probe_context":      probe_context,
            "validator_attempts": validator_attempts + 1}


def _summarise(result: dict) -> str:
    if result["type"] == "ask":
        return f"ASK={result['result']}"
    if result["type"] == "select":
        return f"SELECT {result['total']} rows"
    if result["type"] == "error":
        return f"ERROR: {result['message'][:60]}"
    return result["type"]
