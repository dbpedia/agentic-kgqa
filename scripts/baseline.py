#!/usr/bin/env python3
"""
BASELINE: Replicates Tommaso's exact entity linking approach from agentic-kgqa.
Uses Redis surface forms + heuristic fallback. No LLM reasoning.
Evaluates against DB25 benchmark using gold surface forms.

Run: python baseline.py
"""

import yaml
import re
import os
import sys
import pandas as pd
import dotenv

dotenv.load_dotenv('/Users/siddharth/Desktop/DBpedia/agentic-kgqa/.env', override=True)

sys.path.insert(0, '/Users/siddharth/Desktop/DBpedia/agentic-kgqa')
from src.entity_linking import RedisEntityLinking

BENCHMARK_PATH = '/Users/siddharth/Desktop/DBpedia/agentic-kgqa/benchmark/questions_db25.yaml'
TOP_K = 3

def load_benchmark(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def extract_gold_entities(sparql):
    return re.findall(r'<http://dbpedia\.org/resource/([^>]+)>', sparql)

def uri_to_surface(uri):
    return uri.replace('_', ' ').replace('(', '').replace(')', '')

def check_correct(uri, gold_entities):
    if not uri:
        return False
    return any(
        g.lower().replace('_', ' ') in uri.lower()
        or uri.endswith(g)
        or g in uri
        for g in gold_entities
    )

def get_redis_candidates(el, mention):
    results = el.lookup(mention, top_k=TOP_K, thr=0.01)
    if len(results) == 0:
        return []
    candidates = []
    for idx, row in results.iterrows():
        uri = idx if isinstance(idx, str) else row.name
        if not uri.startswith("http"):
            uri = "http://dbpedia.org/resource/" + uri
        candidates.append({
            "uri": uri,
            "score": round(row["score"], 4)
        })
    return candidates

def tommaso_link(el, entity):
    """Replicates Tommaso's _link_entities logic exactly.
    Returns top-1 Redis result or heuristic fallback."""
    candidates = get_redis_candidates(el, entity)
    if candidates:
        return candidates[0]["uri"]
    else:
        # Tommaso's heuristic fallback when Redis finds nothing
        return "http://dbpedia.org/resource/" + entity.replace(" ", "_")

def main():
    print("Loading benchmark...")
    data = load_benchmark(BENCHMARK_PATH)

    print("Connecting to Redis...")
    el = RedisEntityLinking()

    results = []
    print(f"\nRunning BASELINE evaluation (Tommaso's approach)...\n")
    print("=" * 60)

    for q in data['questions']:
        qid = q['id']
        question = q['question']['en']
        sparql = q['query']['sparql']
        gold_entities = extract_gold_entities(sparql)

        if not gold_entities:
            print(f"Q{qid} | {question[:70]}")
            print(f"  No gold entities in SPARQL, skipping\n")
            continue

        print(f"Q{qid} | {question[:70]}")
        print(f"  Gold: {gold_entities}")

        for gold_uri in gold_entities:
            surface = uri_to_surface(gold_uri)
            predicted = tommaso_link(el, surface)
            correct = check_correct(predicted, gold_entities)

            print(f"  '{surface}' -> {predicted} | Correct: {correct}")

            results.append({
                "id": qid,
                "question": question[:70],
                "gold_uri": gold_uri,
                "surface": surface,
                "predicted": predicted,
                "correct": correct
            })

        print()

    df = pd.DataFrame(results)
    df.to_csv('/Users/siddharth/Desktop/entity-linker-dev/baseline_results.csv', index=False)

    total = len(df)
    correct = df["correct"].sum()

    print("=" * 60)
    print(f"\nBASELINE RESULTS (Tommaso's approach — Redis top-1 + heuristic fallback)")
    print(f"Total entity mentions: {total}")
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"\nFailed cases:")
    for _, row in df[df["correct"] == False].iterrows():
        print(f"  Q{row['id']} | '{row['surface']}' -> {row['predicted']}")
    print(f"\nResults saved to baseline_results.csv")

if __name__ == "__main__":
    main()
