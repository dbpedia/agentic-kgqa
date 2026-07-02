#!/usr/bin/env python3
"""Query Executor node: runs SPARQL with one deterministic fallback.

Responsibilities:
  1. Execute the generated SPARQL against the DBpedia endpoint
  2. If that fails: class-safe dbo: -> dbp: namespace swap and retry ONCE

That is ALL this node does. Complex recovery (agentic probing, routing)
is handled by the Validator node which runs after this one.
"""

import re

from src.sparql_client import execute, needs_fix

_ONTOLOGY_URI_RE = re.compile(r"http://dbpedia\.org/ontology/([^\s<>\"'),.;]+)")


def _swap_dbo_to_dbp(sparql: str) -> str:
    """Deterministically swap dbo: ontology URIs to dbp: property URIs --
    PROPERTY positions only.

    DBpedia naming convention: properties are lowerCamelCase, classes are
    UpperCamelCase. There is no dbp: namespace equivalent for classes --
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


def _summarise(result: dict) -> str:
    if result["type"] == "ask":
        return f"ASK={result['result']}"
    if result["type"] == "select":
        return f"SELECT {result['total']} rows"
    if result["type"] == "error":
        return f"ERROR: {result['message'][:60]}"
    return result["type"]


def run(sparql: str) -> dict:
    """Execute SPARQL with one deterministic dbo->dbp fallback.

    Returns:
        {
            "sparql":    final SPARQL string that was executed,
            "result":    result dict from sparql_client,
            "attempts":  number of execution attempts (1 or 2),
            "fallback":  None | "dbo_to_dbp",
        }
    """
    print(f"\n[EXECUTOR] Executing initial query...")
    result      = execute(sparql)
    fix, reason = needs_fix(result)

    if not fix:
        print(f"[EXECUTOR] Result: {_summarise(result)} -- PASS")
        return {"sparql": sparql, "result": result, "attempts": 1, "fallback": None}

    print(f"[EXECUTOR] Failed ({reason}). Trying dbo->dbp swap...")
    swapped = _swap_dbo_to_dbp(sparql)

    if swapped == sparql:
        print(f"[EXECUTOR] No dbo: URIs found to swap. Returning original failure.")
        return {"sparql": sparql, "result": result, "attempts": 1, "fallback": None}

    swap_result = execute(swapped)
    fix2, reason2 = needs_fix(swap_result)

    if not fix2:
        print(f"[EXECUTOR] dbo->dbp swap succeeded: {_summarise(swap_result)}")
        return {"sparql": swapped, "result": swap_result, "attempts": 2, "fallback": "dbo_to_dbp"}

    print(f"[EXECUTOR] dbo->dbp swap also failed ({reason2}). Passing to Validator.")
    return {"sparql": swapped, "result": swap_result, "attempts": 2, "fallback": "dbo_to_dbp"}
