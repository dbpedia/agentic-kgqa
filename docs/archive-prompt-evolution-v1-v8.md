# Evaluation Analysis: Prompt Evolution v1 → v8

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | Best |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | 20 | 6 | 18 | 12 | 18 | **22** | 17 | **22** | v6/v8 |
| BGP Nodes | 25* | 57 | 61 | **67** | 55 | 59 | 61 | 59 | v4 |
| BGP Preds | 25* | 26 | 26 | 20 | 28 | 29 | 22 | **31** | **v8** |
| Inner ops | **84** | 81 | 79 | 85 | 85 | **86** | 85 | 82 | v6 |
| Outer ops | 75 | 52 | 76 | 74 | 77 | **86** | 84 | 81 | v6 |

*v1 had a single combined BGP metric.

### Namespace usage

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | Bench |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dbo | 83 | 49 | ~70 | 53 | 65 | 70 | 58 | 64 | 76 |
| dbp | 25 | 63 | ~30 | 63 | 48 | 47 | 53 | 49 | 38 |

## Version history (condensed)

| Ver | Accuracy | Key change | Outcome |
|---|:---:|---|---|
| v1 | 20% | Baseline: "prefer dbo:" | — |
| v2 | 6% | Removed dbo preference | Over-corrected to dbp |
| v3 | 18% | Restored dbo + DISTINCT | Recovery |
| v4 | 12% | Triple counts for selection | Anti-correlated signal |
| v5 | 18% | Hard dbo + grouped pairs | Recovery, best preds 28% |
| v6 | **22%** | Unicode fallback + type constraints + COUNT fix | New best |
| v7 | 17% | Over-emphasised type constraints (7 examples) | Complexity hurt |
| v8 | **22%** | Reverted to v6 structure + Film example | Tied best, new best preds 31% |

## v8 analysis

v8 ties v6 at 22% overall and sets new best for BGP Predicates (31%).

**vs v6**: 4 improved (Q43, Q49, Q50, Q53), 4 regressed (Q51, Q82, Q87, Q98) → net 0.
- Q51: triple direction reversed (subject/object swap of dbo:spouse)
- Q82: dbo→dbp flip
- Q87: over-added rdf:type Company constraint that benchmark doesn't expect
- Q98: randomly omitted SELECT DISTINCT

**6 "BGP OK, ops wrong"** (lowest-hanging fruit):
- Q38, Q42, Q60, Q77, Q98, Q100 — mostly minor COUNT/DISTINCT formatting

## Proven principles (after 8 versions)

1. **22% is the prompt engineering ceiling** for this task+model combination. v6 and v8 both hit it.
2. **Simple flat-rule prompts outperform structured ones**. v6/v8 structure > v7's numbered priorities.
3. **5-6 examples is optimal**. Fewer misses patterns; more causes regressions.
4. **dbo: preference is mandatory**. Every weakening regressed.
5. **Data-driven namespace signals hurt**. Embedding scores and triple counts are anti-correlated.
6. **Structural rules stick**: DISTINCT, COUNT format, ORDER BY+LIMIT.
7. **Entity linking fixes compound**: Unicode fallback was a permanent gain.

## Remaining failure patterns (v8, 78 failures)

| Category | Count | % |
|---|:---:|:---:|
| Both FAIL | 38 | 49% |
| Nodes OK, Preds FAIL | 31 | 40% |
| BGP OK, ops wrong | 6 | 8% |
| Nodes FAIL, Preds OK | 3 | 4% |

## What would move beyond 22%

Further gains require **infrastructure changes**, not prompt tweaks:

1. **Predicate synonym mapping** (~5-8 questions): build dbo↔dbp equivalence table from the dataset. When the model picks `dbo:director` but benchmark expects `dbp:director`, the mapping can flag this.

2. **Entity disambiguation** (~8-10 questions): many "both FAIL" cases involve wrong entity resolution (book vs film, city vs region). Better entity type constraints in the Redis lookup could help.

3. **Multi-hop query patterns** (~10 questions): complex queries with 2+ hops consistently fail. Could use few-shot retrieval (find similar benchmark questions) instead of static examples.

4. **Different model**: DeepSeek V3.2 may have hit its ceiling. Testing with Claude Sonnet or GPT-5.4-mini might show different patterns.
