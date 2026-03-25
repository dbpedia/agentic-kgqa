# Evaluation Analysis: Prompt Evolution v1 → v2 → v3

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v1→v3 |
|--------|:---:|:---:|:---:|:---:|
| **Overall match** | 20.0% | 6.0% | **18.0%** | −2pp |
| BGP Nodes | 25.0%* | 57.0% | **61.0%** | +36pp |
| BGP Predicates | 25.0%* | 26.0% | **26.0%** | +1pp |
| Inner operators | 84.0% | 81.0% | **79.0%** | −5pp |
| Outer operators | 75.0% | 52.0% | **76.0%** | +1pp |

*v1 had a single combined BGP metric (25.0%), shown in both columns for comparison.

## Version history

### v1 — Baseline
- "Prefer ontology properties over dbp: when both are available"
- Single example using `dbo:birthPlace`
- No explicit DISTINCT rule

### v2 — Namespace over-correction (regression: 20% → 6%)
**What changed**: removed dbo preference, told LLM to pick the higher-scoring ontology lookup result.

**What went wrong**:
1. **Namespace flip** (41 failures): `dbp:` embeddings score systematically higher after reindex → agent over-used `dbp:` (63 queries) vs benchmark expectation (38 queries using dbp)
2. **Missing DISTINCT** (35 failures): new example and rules dropped DISTINCT emphasis → DeepSeek stopped adding it
3. **Outer ops collapse** (75% → 52%): DISTINCT + structural changes combined

**What went right**: BGP Nodes jumped from 25% to 57% — entity linking is solid.

### v3 — Corrective balance (recovery: 6% → 18%)
**What changed**:
1. Restored dbo preference with conditional override: "prefer dbo: unless ONLY a dbp: match exists"
2. Restored explicit DISTINCT rule: "ALWAYS use SELECT DISTINCT for resource/literal queries"
3. Three diverse examples: dbo (birthPlace), dbp (team/manager), ASK (location)

**Results**:
- 16 questions improved vs v2, 4 regressed → net +12
- Outer ops fully recovered (52% → 76%), matching v1
- BGP Nodes continued improving (57% → 61%)
- BGP Predicates unchanged at 26% — the hard problem

## Current failure analysis (v3, 82 failures)

| Category | Count | % of failures |
|---|:---:|:---:|
| Nodes OK, Predicates FAIL | 37 | 45% |
| Both nodes and predicates FAIL | 37 | 45% |
| BGP OK, operators wrong | 6 | 7% |
| Nodes FAIL, Predicates OK | 2 | 2% |

### The predicate problem (74 failures, 90% of all failures)

Predicates remain the dominant bottleneck. The 37 "nodes OK, predicates FAIL" cases are the clearest targets — the agent finds the right entities but picks the wrong property. The 37 "both FAIL" cases are more complex (wrong entity + wrong property, or structural mismatch).

### The namespace split

| | v1 | v2 | v3 | Benchmark |
|---|:---:|:---:|:---:|:---:|
| Queries using dbo: | 83 | 49 | ~70 | 76 |
| Queries using dbp: | 25 | 63 | ~40 | 38 |

v3 is closer to the benchmark distribution than either v1 or v2, but still not precise enough at the per-question level.

## Key insight: the translation trade-off

There is a fundamental tension between two goals:

1. **Schema fidelity** — translate the question assuming the data conforms to the ontology. Use `dbo:` properties, add `rdf:type` constraints. Produces cleaner, more "correct" queries but may miss data that's only in `dbp:`.

2. **Data fidelity** — translate the question to discover as much data as possible. Use `dbp:` properties (broader coverage), drop type constraints. Produces messier queries but finds more results.

The benchmark itself is inconsistent — some questions expect schema-faithful translations (using `dbo:`) while others expect data-faithful ones (using `dbp:`). **There's no single prompt that can guess which approach each question needs** without external knowledge of the actual data distribution.

## Potential next steps (priority order)

### High impact: property-level intelligence

1. **Frequency-weighted ontology lookup**: weight results by triple count in the actual dataset. If `dbo:director` has 50k triples and `dbp:director` has 500, the score should reflect that. This gives the LLM a data-informed signal rather than just embedding similarity.

2. **Property mapping table**: precompute a `dbo:X ↔ dbp:X` equivalence map from the dataset. Annotate each ontology result with "also available as dbo/dbp with N triples". This makes the namespace choice explicit and data-driven.

3. **Two-pass namespace fallback**: use the self-correction loop to try `dbo:` first, then swap to `dbp:` if 0 results. This is already partially implemented but could be made more systematic.

### Medium impact: structural improvements

4. **Few-shot examples from the benchmark**: include 5 diverse examples (currently 3) covering more patterns: FILTER, ORDER BY + LIMIT, COUNT, multi-hop, etc. Each example should show the "why" of the namespace choice.

5. **Structural rules enforcement**: add explicit rules for ORDER BY, LIMIT, GROUP BY patterns. The 6 "BGP OK but ops wrong" failures suggest the model sometimes gets the triples right but structures the query wrong.

### Lower impact: evaluation infrastructure

6. **Per-question regression tracking**: automatically diff v(N) vs v(N-1) and flag regressions in the History tab.

7. **Namespace accuracy metric**: track "% of queries using the correct namespace" as a dedicated axis, separate from predicate correctness.

8. **Cross-version A/B**: run two prompt versions on the same questions in a single eval for controlled comparison.
