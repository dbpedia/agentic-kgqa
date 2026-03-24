"""Evaluation module: benchmark the KGQA agent against expected SPARQL queries using LLM-based comparison."""

import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).parent.parent / "benchmark" / "questions_db25.yaml"
EVALS_DIR = Path(__file__).parent.parent / "data" / "evals"

# Always use Claude Sonnet 4.6 for evaluation regardless of the generation model
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"

COMPARE_PROMPT = """\
Compare these two SPARQL queries for functional equivalence.
Variable names do NOT matter — only the semantics.

Expected:
{expected}

Generated:
{generated}

Score each axis as true (equivalent) or false (different):
1. bgp: Do they have the same basic graph patterns (same triple patterns, order doesn't matter)?
2. inner_ops: Same inner-query operators (FILTER, VALUES, OPTIONAL, UNION)?
3. outer_ops: Same outer-query structure (SELECT/ASK, COUNT, ORDER BY, LIMIT, DISTINCT)?

Overall match = all three are true.

Output JSON only: {{"match": bool, "bgp": bool, "inner_ops": bool, "outer_ops": bool, "explanation": "brief reason"}}
"""


def _get_judge_client():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def load_benchmark(path=None):
    """Load benchmark questions from YAML. Returns list of {id, question, expected_sparql}."""
    path = path or BENCHMARK_PATH
    with open(path) as f:
        data = yaml.safe_load(f)

    questions = []
    for q in data.get("questions", []):
        sparql = q.get("query", {}).get("sparql", "").strip()
        question_text = q.get("question", {}).get("en", "")
        if sparql and question_text:
            questions.append({
                "id": q["id"],
                "question": question_text,
                "expected_sparql": sparql,
            })
    return questions


def save_result(result_data):
    """Save evaluation result to data/evals/. Returns the result id."""
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    model_slug = result_data.get("model", "unknown").split("/")[-1]
    result_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{model_slug}"
    result_data["id"] = result_id
    result_data["timestamp"] = ts.isoformat()

    path = EVALS_DIR / f"{result_id}.json"
    with open(path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    logger.info(f"Saved eval result to {path}")
    return result_id


def list_results():
    """List saved evaluation results (metadata only), newest first."""
    if not EVALS_DIR.exists():
        return []
    results = []
    for p in sorted(EVALS_DIR.glob("*.json"), reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            results.append({
                "id": data.get("id", p.stem),
                "timestamp": data.get("timestamp"),
                "mode": data.get("mode", "single"),
                "model": data.get("model"),
                "n": data.get("n"),
                "accuracy": data.get("summary", {}).get("accuracy") if data.get("mode") != "all" else None,
                "models_count": len(data.get("models", [])) if data.get("mode") == "all" else None,
            })
        except Exception as e:
            logger.warning(f"Could not read eval result {p}: {e}")
    return results


def load_result(result_id):
    """Load a saved evaluation result by id."""
    # Sanitise to prevent path traversal
    safe_id = Path(result_id).name
    path = EVALS_DIR / f"{safe_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_sparql(generated, expected, client=None):
    """Use Claude Sonnet 4.6 to compare two SPARQL queries for functional equivalence.

    Returns dict with keys: match, bgp, inner_ops, outer_ops, explanation.
    """
    client = client or _get_judge_client()
    prompt = COMPARE_PROMPT.format(expected=expected.strip(), generated=generated.strip())

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content

    # Extract JSON from response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "match": False,
        "bgp": False,
        "inner_ops": False,
        "outer_ops": False,
        "explanation": f"Could not parse judge response: {text[:200]}",
    }


def evaluate_stream(agent, n=10, model=None, shuffle=True):
    """Generator yielding (event_type, data) tuples for SSE evaluation.

    Randomly samples n questions from the benchmark, runs the agent, and compares results.
    If shuffle=False, takes the first n questions in benchmark order.
    """
    questions = load_benchmark()
    if shuffle:
        sample = random.sample(questions, min(n, len(questions)))
    else:
        sample = questions[:n]
    judge_client = _get_judge_client()

    yield ("eval_start", {"total": len(sample), "model": model or "default", "judge": JUDGE_MODEL})

    results = []
    for i, item in enumerate(sample):
        qid = item["id"]
        question = item["question"]
        expected = item["expected_sparql"]

        yield ("eval_progress", {"index": i + 1, "total": len(sample), "question_id": qid, "question": question})

        # Run agent
        try:
            generated = agent.answer(question, model=model)
        except Exception as e:
            logger.error(f"Agent error on question {qid}: {e}")
            generated = f"ERROR: {e}"

        # Compare
        if generated.startswith("ERROR:"):
            comparison = {
                "match": False, "bgp": False, "inner_ops": False, "outer_ops": False,
                "explanation": generated,
            }
        else:
            comparison = compare_sparql(generated, expected, client=judge_client)

        result = {
            "question_id": qid,
            "question": question,
            "expected": expected,
            "generated": generated,
            "comparison": comparison,
        }
        results.append(result)

        yield ("eval_question", result)

    # Summary
    total = len(results)
    matches = sum(1 for r in results if r["comparison"].get("match"))
    bgp_matches = sum(1 for r in results if r["comparison"].get("bgp"))
    inner_matches = sum(1 for r in results if r["comparison"].get("inner_ops"))
    outer_matches = sum(1 for r in results if r["comparison"].get("outer_ops"))

    summary = {
        "total": total,
        "matches": matches,
        "accuracy": round(matches / total * 100, 1) if total else 0,
        "bgp_accuracy": round(bgp_matches / total * 100, 1) if total else 0,
        "inner_ops_accuracy": round(inner_matches / total * 100, 1) if total else 0,
        "outer_ops_accuracy": round(outer_matches / total * 100, 1) if total else 0,
    }
    yield ("eval_done", summary)

    # Auto-save
    result_id = save_result({
        "mode": "single",
        "model": model or "default",
        "n": len(sample),
        "judge": JUDGE_MODEL,
        "questions": results,
        "summary": summary,
    })
    yield ("eval_saved", {"id": result_id})


def _summarise(results):
    """Compute accuracy stats from a list of per-question results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "matches": 0, "accuracy": 0, "bgp_accuracy": 0, "inner_ops_accuracy": 0, "outer_ops_accuracy": 0}
    matches = sum(1 for r in results if r["comparison"].get("match"))
    bgp = sum(1 for r in results if r["comparison"].get("bgp"))
    inner = sum(1 for r in results if r["comparison"].get("inner_ops"))
    outer = sum(1 for r in results if r["comparison"].get("outer_ops"))
    return {
        "total": total,
        "matches": matches,
        "accuracy": round(matches / total * 100, 1),
        "bgp_accuracy": round(bgp / total * 100, 1),
        "inner_ops_accuracy": round(inner / total * 100, 1),
        "outer_ops_accuracy": round(outer / total * 100, 1),
    }


def evaluate_all_stream(agent, models, n=10, shuffle=True):
    """Run evaluation across ALL models on the same sample, yielding per-model results and a comparison summary.

    Events:
      - eval_all_start: {total_questions, models, judge}
      - eval_model_start: {model_id, model_label, model_index, total_models}
      - eval_progress: {model_id, index, total, question_id, question}
      - eval_question: {model_id, question_id, question, expected, generated, comparison}
      - eval_model_done: {model_id, model_label, summary}
      - eval_all_done: {models: [{model_id, model_label, summary}], questions: [question_ids]}
    """
    questions = load_benchmark()
    if shuffle:
        sample = random.sample(questions, min(n, len(questions)))
    else:
        sample = questions[:n]
    judge_client = _get_judge_client()

    model_ids = [m["id"] for m in models]
    model_labels = {m["id"]: m["label"] for m in models}

    yield ("eval_all_start", {
        "total_questions": len(sample),
        "models": [{"id": m["id"], "label": m["label"]} for m in models],
        "judge": JUDGE_MODEL,
    })

    all_model_summaries = []
    all_questions = []  # collect all per-question results for saving

    for mi, model_id in enumerate(model_ids):
        label = model_labels[model_id]
        yield ("eval_model_start", {
            "model_id": model_id, "model_label": label,
            "model_index": mi + 1, "total_models": len(model_ids),
        })

        model_results = []
        for qi, item in enumerate(sample):
            qid = item["id"]
            question = item["question"]
            expected = item["expected_sparql"]

            yield ("eval_progress", {
                "model_id": model_id, "index": qi + 1, "total": len(sample),
                "question_id": qid, "question": question,
            })

            try:
                generated = agent.answer(question, model=model_id)
            except Exception as e:
                logger.error(f"Agent error on question {qid} with {model_id}: {e}")
                generated = f"ERROR: {e}"

            if generated.startswith("ERROR:"):
                comparison = {
                    "match": False, "bgp": False, "inner_ops": False, "outer_ops": False,
                    "explanation": generated,
                }
            else:
                comparison = compare_sparql(generated, expected, client=judge_client)

            result = {
                "model_id": model_id,
                "question_id": qid,
                "question": question,
                "expected": expected,
                "generated": generated,
                "comparison": comparison,
            }
            model_results.append(result)
            all_questions.append(result)
            yield ("eval_question", result)

        summary = _summarise(model_results)
        all_model_summaries.append({
            "model_id": model_id,
            "model_label": label,
            "summary": summary,
        })
        yield ("eval_model_done", {
            "model_id": model_id, "model_label": label, "summary": summary,
        })

    all_done_data = {
        "models": all_model_summaries,
        "question_ids": [q["id"] for q in sample],
    }
    yield ("eval_all_done", all_done_data)

    # Auto-save
    result_id = save_result({
        "mode": "all",
        "model": "all",
        "n": len(sample),
        "judge": JUDGE_MODEL,
        "questions": all_questions,
        "models": all_model_summaries,
        "summary": all_done_data,
    })
    yield ("eval_saved", {"id": result_id})
