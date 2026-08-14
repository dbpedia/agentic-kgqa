#!/usr/bin/env python3
"""Centralised SPARQL client.

ALL pipeline queries go to the ENDPOINT defined below. This module is
imported by query_executor.py and validator.py so they all share a single
execution path and result format.
"""

import json
import urllib.parse
import urllib.request

# All pipeline SPARQL queries go through this fixed endpoint.
ENDPOINT = "http://research.liberai.org:7878/sparql"
TIMEOUT  = 40


def execute(sparql: str, timeout: int = TIMEOUT) -> dict:
    """Execute a SPARQL query.

    Returns a result dict:
      {"type": "select", "vars": [...], "rows": [...], "total": N}
      {"type": "ask",    "result": True/False}
      {"type": "error",  "message": "..."}
    """
    params = urllib.parse.urlencode({
        "query":  sparql,
        "format": "application/sparql-results+json",
    })
    url = f"{ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={
        "Accept":     "application/sparql-results+json",
        "User-Agent": "DBpediaKGQA-GSoC2026/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if "boolean" in data:
            return {"type": "ask", "result": data["boolean"]}

        if "results" in data and "bindings" in data["results"]:
            bindings  = data["results"]["bindings"]
            variables = data.get("head", {}).get("vars", [])
            rows = []
            for b in bindings[:200]:
                row = {v: b[v].get("value", "") for v in variables if v in b}
                rows.append(row)
            return {
                "type":  "select",
                "vars":  variables,
                "rows":  rows,
                "total": len(bindings),
            }

        return {"type": "unknown", "raw": str(data)[:200]}

    except urllib.error.HTTPError as e:
        return {"type": "error", "message": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"type": "error", "message": str(e)[:200]}


def needs_fix(result: dict) -> tuple:
    """Return (True, reason) if the result needs a fallback attempt."""
    if result["type"] == "error":
        return True, f"SPARQL error: {result.get('message', '')[:80]}"
    if result["type"] == "select":
        if result["total"] == 0:
            return True, "SELECT returned 0 results"
        rows = result.get("rows", [])
        if len(rows) == 1:
            vals = list(rows[0].values())
            if len(vals) == 1 and vals[0] in ("0", 0):
                return True, "COUNT returned 0"
    return False, ""
