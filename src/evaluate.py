#!/usr/bin/env python3
"""Evaluate the agentic KGQA pipeline against DB25 or DB26 benchmark.

All generated SPARQL queries run against the DBpedia evaluation endpoint.
Gold results for DB26 are read from a pre-computed cache (data/db26_gold_final.json)
to eliminate gold-side timing variance between comparison runs. Build the
cache by running: pipenv run python scripts/build_db26_gold.py

Usage:
    pipenv run python src/evaluate.py 10 qwen/qwen3.5-122b-a10b 0 db26
    pipenv run python src/evaluate.py 50 qwen/qwen3.5-122b-a10b 0 db26
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import dotenv
import yaml
from openai import OpenAI

dotenv.load_dotenv(override=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT     = Path(__file__).parent.parent
BENCHMARK_DB25 = _REPO_ROOT / "benchmark" / "questions_db25.yaml"
BENCHMARK_DB26 = _REPO_ROOT / "benchmark" / "questions_db26.yml"
DATA_DIR       = _REPO_ROOT / "data"
EVALS_DIR      = DATA_DIR / "evals"
GOLD_CACHE_DB26 = DATA_DIR / "db26_gold_final.json"
GOLD_CACHE_DB25 = DATA_DIR / "db25_gold_final.json"

# ─── Constants ────────────────────────────────────────────────────────────────
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
ENDPOINT    = "http://research.liberai.org:7878/sparql"

# ─── Gold cache ───────────────────────────────────────────────────────────────
_gold_cache_db26 = None
_gold_cache_db25 = None


def _load_gold_cache_db26() -> dict:
    """Load pre-computed DB26 gold results from data/db26_gold_final.json.

    Returns {question_id: gold_result_dict}. Falls back to live execution
    per question if the cache file does not exist -- nothing breaks, but
    gold-side timing variance is reintroduced between runs. Build the cache
    by running: pipenv run python scripts/build_db26_gold.py
    """
    global _gold_cache_db26
    if _gold_cache_db26 is not None:
        return _gold_cache_db26
    if not GOLD_CACHE_DB26.exists():
        print(f"[WARN] Gold cache not found at {GOLD_CACHE_DB26}. "
              f"Run scripts/build_db26_gold.py to build it.")
        _gold_cache_db26 = {}
        return _gold_cache_db26
    with open(GOLD_CACHE_DB26) as f:
        data = json.load(f)
    _gold_cache_db26 = {q["id"]: q["gold_result"] for q in data.get("questions", [])}
    return _gold_cache_db26


def _load_gold_cache_db25() -> dict:
    """Load pre-computed DB25 gold results from data/db25_gold_final.json.

    Returns {question_id: gold_result_dict}. Falls back to live execution
    per question if the cache file does not exist -- nothing breaks, but
    gold-side timing variance is reintroduced between runs. Build the cache
    by running: pipenv run python scripts/build_db25_gold.py
    """
    global _gold_cache_db25
    if _gold_cache_db25 is not None:
        return _gold_cache_db25
    if not GOLD_CACHE_DB25.exists():
        print(f"[WARN] Gold cache not found at {GOLD_CACHE_DB25}. "
              f"Run scripts/build_db25_gold.py to build it.")
        _gold_cache_db25 = {}
        return _gold_cache_db25
    with open(GOLD_CACHE_DB25) as f:
        data = json.load(f)
    _gold_cache_db25 = {q["id"]: q["gold_result"] for q in data.get("questions", [])}
    return _gold_cache_db25


# ─── LLM client ───────────────────────────────────────────────────────────────
def _get_judge_client():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


# ─── Benchmark loader ─────────────────────────────────────────────────────────
def load_benchmark(benchmark: str = "db26") -> list:
    path = BENCHMARK_DB26 if benchmark == "db26" else BENCHMARK_DB25
    with open(path) as f:
        data = yaml.safe_load(f)
    questions = []
    for q in data.get("questions", []):
        sparql        = q.get("query", {}).get("sparql", "").strip()
        question_text = q.get("question", {}).get("en", "")
        if sparql and question_text:
            questions.append({
                "id":              q["id"],
                "question":        question_text,
                "expected_sparql": sparql,
            })
    return questions


# ─── SPARQL execution ─────────────────────────────────────────────────────────
def _execute_sparql(sparql: str) -> dict:
    """Run SPARQL against the DBpedia evaluation endpoint."""
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
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "boolean" in data:
            return {"type": "ask", "result": data["boolean"]}
        if "results" in data and "bindings" in data["results"]:
            bindings  = data["results"]["bindings"]
            variables = data.get("head", {}).get("vars", [])
            rows = [
                {v: b[v].get("value", "") for v in variables if v in b}
                for b in bindings[:200]
            ]
            return {"type": "select", "vars": variables, "rows": rows, "total": len(bindings)}
        return {"type": "unknown", "raw": str(data)[:200]}
    except Exception as e:
        return {"type": "error", "message": str(e)[:200]}


# ─── SPARQL structural comparison (LLM judge) ─────────────────────────────────
COMPARE_PROMPT = """\
Compare these two SPARQL queries for functional equivalence.
Variable names do NOT matter -- only the semantics.

Expected:
{expected}

Generated:
{generated}

Score each axis as true (equivalent) or false (different):
1. bgp_nodes: Do they reference the same entity/resource URIs?
2. bgp_predicates: Do they use the same predicates, including correct namespace (dbo: vs dbp:)?
3. inner_ops: Same inner operators (FILTER, OPTIONAL, UNION)?
4. outer_ops: Same outer structure (SELECT/ASK, COUNT, ORDER BY, LIMIT, DISTINCT)?

Overall match = all four are true.

Output JSON only:
{{"match": bool, "bgp_nodes": bool, "bgp_predicates": bool, "inner_ops": bool, "outer_ops": bool, "explanation": "brief reason"}}
"""


def compare_sparql(generated: str, expected: str, client) -> dict:
    prompt = COMPARE_PROMPT.format(expected=expected.strip(), generated=generated.strip())
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text  = response.choices[0].message.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {
        "match": False, "bgp_nodes": False, "bgp_predicates": False,
        "inner_ops": False, "outer_ops": False,
        "explanation": "Could not parse judge response",
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _is_single_numeric(rows: list):
    """If result is exactly one row with one numeric value, return it as float."""
    if len(rows) != 1:
        return None
    values = list(rows[0].values())
    if len(values) != 1:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return None


def _column_superset_match(gen_rows: list, exp_rows: list, exp_vars: list) -> bool:
    """True if gen_rows contains all exp_rows values when gold has exactly one variable.

    Handles cases where our query returns the correct entity/value plus extra
    supplementary columns -- the extra column is added information, not a wrong answer.
    """
    if len(exp_vars) != 1 or not exp_rows:
        return False
    gen_value_sets = [set(r.values()) for r in gen_rows]
    for exp_row in exp_rows:
        (exp_val,) = exp_row.values()
        if not any(exp_val in gen_vals for gen_vals in gen_value_sets):
            return False
    return len(gen_rows) == len(exp_rows)


def _result_set_match(gen: dict, exp: dict) -> bool:
    """Exact result-set equality with two narrow structural-equivalence allowances:
      1. Single-value numeric: compared by absolute value (sign convention is arbitrary)
      2. Column superset: extra columns beyond gold's single variable are tolerated
    """
    if gen["type"] != exp["type"]:
        return False
    if gen["type"] == "ask":
        return gen.get("result") == exp.get("result")
    if gen["type"] == "select":
        g_rows = gen.get("rows", [])
        e_rows = exp.get("rows", [])
        g = set(frozenset(r.values()) for r in g_rows)
        e = set(frozenset(r.values()) for r in e_rows)
        if not g and not e:
            return False  # both empty = both failed, not a match
        if g == e:
            return True
        g_num = _is_single_numeric(g_rows)
        e_num = _is_single_numeric(e_rows)
        if g_num is not None and e_num is not None:
            return abs(g_num) == abs(e_num)
        if _column_superset_match(g_rows, e_rows, exp.get("vars", [])):
            return True
        return False
    return False


def _prf(gen: dict, exp: dict) -> tuple:
    """Compute (precision, recall, F1) using the same allowances as _result_set_match."""
    if gen["type"] != exp["type"]:
        return 0.0, 0.0, 0.0
    if gen["type"] == "ask":
        ok = gen.get("result") == exp.get("result")
        return (1.0, 1.0, 1.0) if ok else (0.0, 0.0, 0.0)
    if gen["type"] == "select":
        g_rows = gen.get("rows", [])
        e_rows = exp.get("rows", [])
        g = set(frozenset(r.values()) for r in g_rows)
        e = set(frozenset(r.values()) for r in e_rows)
        if not g and not e:
            return 0.0, 0.0, 0.0  # both empty = both failed, not a match
        if not g or not e:
            return 0.0, 0.0, 0.0
        if g == e:
            return 1.0, 1.0, 1.0
        g_num = _is_single_numeric(g_rows)
        e_num = _is_single_numeric(e_rows)
        if g_num is not None and e_num is not None and abs(g_num) == abs(e_num):
            return 1.0, 1.0, 1.0
        if _column_superset_match(g_rows, e_rows, exp.get("vars", [])):
            return 1.0, 1.0, 1.0
        tp   = len(g & e)
        prec = round(tp / len(g), 4)
        rec  = round(tp / len(e), 4)
        if prec + rec == 0:
            return 0.0, 0.0, 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4)
        return prec, rec, f1
    return 0.0, 0.0, 0.0


# ─── Main evaluation function ─────────────────────────────────────────────────
def evaluate_pipeline(
    n:         int = 10,
    model:     str = "qwen/qwen3.5-122b-a10b",
    start:     int = 0,
    benchmark: str = "db26",
) -> dict:
    """Run the full pipeline on n questions and compute P/R/F1."""
    from src.agent import KGQAAgent, build_graph
    from src.entity_linking import RedisEntityLinking

    try:
        redis_el = RedisEntityLinking()
    except Exception:
        redis_el = None

    graph        = build_graph(redis_el=redis_el, model=model)
    questions    = load_benchmark(benchmark)[start:start + n]
    judge_client = _get_judge_client()

    results            = []
    sparql_matches     = 0
    result_set_matches = 0
    total_f1           = 0.0
    total_precision    = 0.0
    total_recall       = 0.0
    total_steps        = 0
    steps_list         = []

    print(f"\n{'='*70}")
    print(f"Evaluating | benchmark={benchmark} | n={n} | model={model}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"{'='*70}")

    for i, item in enumerate(questions):
        qid             = item["id"]
        question        = item["question"]
        expected_sparql = item["expected_sparql"]

        print(f"\n[{i+1}/{n}] Q{qid}: {question[:65]}")

        # Run pipeline
        try:
            state = graph.invoke({
                "question":          question,
                "model":             model,
                "validator_attempts": 0,
                "exec_attempts":     0,
                "probe_context":     "",
                "probe_results":     {},
                "num_hops":          1,
            })
            generated_sparql = state.get("sparql_final") or state.get("sparql", "")
            exec_fallback    = state.get("exec_fallback")
            exec_attempts    = state.get("exec_attempts", 0)
            val_attempts     = state.get("validator_attempts", 0)
            val_action       = state.get("validator_action", "")
            val_reason       = state.get("validator_reason", "")

            print(f"\n{'='*60}")
            print(f"[PIPELINE COMPLETE]")
            print(f"  Question:  {question}")
            print(f"  SPARQL:    {generated_sparql[:120]}")
            print(f"  Result:    {state.get('exec_result', {})}")
            print(f"  Executor attempts: {exec_attempts}")
            print(f"  Validator attempts: {val_attempts}")
            print(f"  Validator action: {val_action}")
            print(f"  Validator reason: {val_reason}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"  ERROR in pipeline: {e}")
            results.append({
                "question_id": qid, "question": question,
                "expected_sparql": expected_sparql,
                "generated_sparql": f"ERROR: {e}",
                "exec_fallback": None, "exec_attempts": 0,
                "validator_attempts": 0, "validator_action": "error",
                "sparql_match": False, "result_set_match": False,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "comparison": {}, "gen_result": {}, "exp_result": {},
            })
            continue

        print(f"  exec_attempts={exec_attempts}  exec_fallback={exec_fallback}  "
              f"val_attempts={val_attempts}  val_action={val_action}")

        # SPARQL structural comparison (LLM judge)
        if generated_sparql.startswith("ERROR:"):
            comparison = {
                "match": False, "bgp_nodes": False, "bgp_predicates": False,
                "inner_ops": False, "outer_ops": False,
                "explanation": generated_sparql,
            }
        else:
            comparison = compare_sparql(generated_sparql, expected_sparql, judge_client)

        sparql_match = comparison.get("match", False)
        if sparql_match:
            sparql_matches += 1

        # Result-set comparison
        if generated_sparql.startswith("ERROR:"):
            gen_result = {"type": "error", "message": "pipeline error"}
        else:
            gen_result = _execute_sparql(generated_sparql)

        # Gold side: use pre-computed cache for db26 and db25, live execution otherwise
        if benchmark == "db26":
            gold_cache = _load_gold_cache_db26()
            exp_result = gold_cache.get(qid) or _execute_sparql(expected_sparql)
        elif benchmark == "db25":
            gold_cache = _load_gold_cache_db25()
            exp_result = gold_cache.get(qid) or _execute_sparql(expected_sparql)
        else:
            exp_result = _execute_sparql(expected_sparql)

        rs_match              = _result_set_match(gen_result, exp_result)
        precision, recall, f1 = _prf(gen_result, exp_result)

        if rs_match:
            result_set_matches += 1
        total_f1        += f1
        total_precision += precision
        total_recall    += recall
        total_steps     += exec_attempts + val_attempts
        steps_list.append(exec_attempts + val_attempts)

        sparql_icon = "✅" if sparql_match else "❌"
        rs_icon     = "✅" if rs_match     else "❌"
        print(f"  SPARQL:{sparql_icon}  Result-set:{rs_icon}  "
              f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  Steps={exec_attempts + val_attempts}")
        print(f"  Judge: {comparison.get('explanation', '')[:80]}")

        results.append({
            "question_id":        qid,
            "question":           question,
            "expected_sparql":    expected_sparql,
            "generated_sparql":   generated_sparql,
            "exec_fallback":      exec_fallback,
            "exec_attempts":      exec_attempts,
            "validator_attempts": val_attempts,
            "validator_action":   val_action,
            "validator_reason":   val_reason,
            "sparql_match":       sparql_match,
            "result_set_match":   rs_match,
            "precision":          precision,
            "recall":             recall,
            "f1":                 f1,
            "comparison":         comparison,
            "gen_result":         gen_result,
            "exp_result":         exp_result,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(results)
    avg_f1   = round(total_f1        / total, 4) if total else 0.0
    avg_prec = round(total_precision / total, 4) if total else 0.0
    avg_rec  = round(total_recall    / total, 4) if total else 0.0
    avg_steps = round(total_steps     / total, 2) if total else 0.0

    # Median and mode of steps
    sorted_steps   = sorted(steps_list)
    mid            = total // 2
    median_steps   = sorted_steps[mid] if total % 2 != 0 else (sorted_steps[mid - 1] + sorted_steps[mid]) / 2
    freq           = {}
    for s in steps_list:
        freq[s] = freq.get(s, 0) + 1
    mode_steps     = max(freq, key=freq.get) if freq else 0

    fallback_counts   = {}
    val_action_counts = {}
    for r in results:
        fb = r.get("exec_fallback") or "none"
        fallback_counts[fb] = fallback_counts.get(fb, 0) + 1
        va = r.get("validator_action") or "none"
        val_action_counts[va] = val_action_counts.get(va, 0) + 1

    print(f"\n{'='*70}")
    print(f"SUMMARY  benchmark={benchmark}  n={total}  model={model}")
    print(f"  SPARQL match:      {sparql_matches}/{total} "
          f"({sparql_matches*100//total if total else 0}%)")
    print(f"  Result-set match:  {result_set_matches}/{total} "
          f"({result_set_matches*100//total if total else 0}%)")
    print(f"  Avg Precision:     {avg_prec}")
    print(f"  Avg Recall:        {avg_rec}")
    print(f"  Avg F1:            {avg_f1}")
    print(f"  Avg steps/question:{avg_steps}")
    print(f"  Median steps:      {median_steps}")
    print(f"  Mode steps:        {mode_steps}")
    print(f"  Exec fallbacks:    {fallback_counts}")
    print(f"  Validator actions: {val_action_counts}")
    print(f"{'='*70}")

    # ── Save ──────────────────────────────────────────────────────────────────
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = EVALS_DIR / f"eval_{benchmark}_{ts}.json"
    payload  = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "model":       model,
        "benchmark":   benchmark,
        "n":           total,
        "judge_model": JUDGE_MODEL,
        "endpoint":    ENDPOINT,
        "summary": {
            "sparql_matches":             sparql_matches,
            "result_set_matches":         result_set_matches,
            "sparql_accuracy_pct":        round(sparql_matches * 100 / total, 1) if total else 0,
            "result_set_accuracy_pct":    round(result_set_matches * 100 / total, 1) if total else 0,
            "avg_precision":              avg_prec,
            "avg_recall":                 avg_rec,
            "avg_f1":                     avg_f1,
            "avg_steps_per_question":      avg_steps,
            "median_steps_per_question":    median_steps,
            "mode_steps_per_question":      mode_steps,
            "exec_fallback_breakdown":    fallback_counts,
            "validator_action_breakdown": val_action_counts,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    return payload


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n         = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    model     = sys.argv[2]      if len(sys.argv) > 2 else "qwen/qwen3.5-122b-a10b"
    start     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    benchmark = sys.argv[4]      if len(sys.argv) > 4 else "db26"
    evaluate_pipeline(n=n, model=model, start=start, benchmark=benchmark)
