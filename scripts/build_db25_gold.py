#!/usr/bin/env python3
"""Pre-compute and cache DB25 gold query results against the evaluation endpoint.

Executes all 100 gold SPARQL queries ONCE against the DBpedia evaluation
endpoint and saves the results to disk. This removes gold-side execution as a
source of run-to-run variance: every future evaluate.py run reads cached gold
results instead of re-executing 100 gold queries fresh each time, so any
timeout/flakiness on the gold side can no longer differ between comparison
runs -- only the generated query behaviour varies between runs.

Usage:
    pipenv run python scripts/build_db25_gold.py

Output: data/db25_gold_final.json
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

_REPO_ROOT      = Path(__file__).parent.parent
QUESTIONS_FILE  = _REPO_ROOT / "benchmark" / "questions_db25.yaml"
OUTPUT_FILE     = _REPO_ROOT / "data" / "db25_gold_final.json"
ENDPOINT        = "http://research.liberai.org:7878/sparql"
TIMEOUT         = 40    # seconds
SLEEP_BETWEEN   = 0.5   # light pacing between requests
MAX_ROWS        = 200   # row cap, matches sparql_client.py


def _execute(sparql: str, timeout: int = TIMEOUT) -> dict:
    """Execute a SPARQL query against the evaluation endpoint."""
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
            rows = [
                {v: b[v].get("value", "") for v in variables if v in b}
                for b in bindings[:MAX_ROWS]
            ]
            return {"type": "select", "vars": variables, "rows": rows, "total": len(bindings)}

        return {"type": "unknown", "raw": str(data)[:200]}

    except urllib.error.HTTPError as e:
        return {"type": "error", "message": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"type": "error", "message": str(e)[:200]}


def main():
    print("=" * 70)
    print("Building DB25 gold cache")
    print(f"Endpoint: {ENDPOINT}  (timeout={TIMEOUT}s)")
    print("=" * 70)

    with open(QUESTIONS_FILE) as f:
        data = yaml.safe_load(f)

    questions = data["questions"]
    total = len(questions)
    print(f"\nLoaded {total} questions from {QUESTIONS_FILE.name}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results  = []
    errors   = 0
    timeouts = []

    for q in questions:
        qid      = q["id"]
        question = q["question"]["en"]
        sparql   = q["query"]["sparql"].strip()

        print(f"\n[{qid:3d}/{total}] {question[:60]}")

        result = _execute(sparql)

        if result["type"] == "error":
            print(f"  ERROR: {result['message']}")
            errors += 1
            if "timed out" in result["message"].lower() or "timeout" in result["message"].lower():
                timeouts.append(qid)
        elif result["type"] == "ask":
            print(f"  ASK -> {result['result']}")
        elif result["type"] == "select":
            print(f"  SELECT -> {result['total']} rows")
            if result["rows"]:
                print(f"  Sample: {list(result['rows'][0].values())[:3]}")
        else:
            print(f"  UNKNOWN result type")

        results.append({
            "id":          qid,
            "question":    question,
            "gold_sparql": sparql,
            "gold_result": result,
        })

        time.sleep(SLEEP_BETWEEN)

    output = {
        "source":    f"DBpedia evaluation endpoint ({ENDPOINT}), timeout={TIMEOUT}s",
        "timeout":   TIMEOUT,
        "questions": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Done. {total} questions processed, {errors} errors.")
    if timeouts:
        print(f"Timed-out question IDs: {timeouts}")
        print(f"(Genuinely expensive full-scan queries -- not fixable from pipeline side.)")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
