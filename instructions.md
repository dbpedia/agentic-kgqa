# Agentic KGQA: Instructions for Claude Code

## Goal

Create a Question Answering agent that answers questions against the DBpedia knowledge graph, 2015-10 snapshot.

## Tools

- The DBpedia ontology used by the benchmark. You may adopt the same approach as in `agentic-kgqa.ipynb` (`lookup_term` function).
- The Redis entity linking service. You may adopt the same approach as in `src/entity_linking.py`.

## Benchmark

- The benchmark is available at `benchmark/`. The file contains the dataset from last year's text2sparql challenge.
- The evaluation will be carried out by the challenge organisers. We need to open an API endpoint. See `src/api_example.py` for an example.

## Important points

- DBpedia data is messy. Do not attempt to double-check whether a result is correct. Just focus on the translation into SPARQL.
- For now, focus on the English questions.
