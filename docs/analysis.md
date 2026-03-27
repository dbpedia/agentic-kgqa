# Evaluation Analysis: Prompt Evolution v1 → v7

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v4 | v5 | v6 | v7 | Best |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall match** | 20.0% | 6.0% | 18.0% | 12.0% | 18.0% | **22.0%** | 17.0% | **v6** |
| BGP Nodes | 25.0%* | 57.0% | 61.0% | 67.0% | 55.0% | 59.0% | **61.0%** | v4 (67%) |
| BGP Predicates | 25.0%* | 26.0% | 26.0% | 20.0% | 28.0% | **29.0%** | 22.0% | **v6** |
| Inner operators | 84.0% | 81.0% | 79.0% | 85.0% | 85.0% | **86.0%** | 85.0% | **v6** |
| Outer operators | 75.0% | 52.0% | 76.0% | 74.0% | 77.0% | **86.0%** | 84.0% | **v6** |

*v1 had a single combined BGP metric.

### Key insight: v6 is the optimal prompt

v6 holds the best scores on 4 of 5 metrics. Every attempt to improve beyond v6 (v7) caused regressions. The prompt engineering ceiling appears to be around 22% for this task.

### Namespace usage

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | Benchmark |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dbo: | 83 | 49 | ~70 | 53 | 65 | 70 | 58 | 76 |
| dbp: | 25 | 63 | ~30 | 63 | 48 | 47 | 53 | 38 |

v6 had the closest namespace distribution (70/47). v7 drifted back toward dbp (58/53).

## Version history

### v1 — Baseline (20%). Simple "prefer dbo:" rule.
### v2 — Over-correction (6%). Removed dbo preference → over-used dbp.
### v3 — Recovery (18%). Restored dbo preference + DISTINCT.
### v4 — Regression (12%). Triple counts anti-correlated with benchmark.
### v5 — Recovery (18%). Hard dbo preference + grouped pairs.
### v6 — New best (22%). Unicode fallback + type constraints + COUNT fix.
### v7 — Regression (17%). Over-emphasised type constraints.

**v7 post-mortem**: restructuring rules as "CRITICAL RULES" with type constraints at #1 priority caused the model to over-focus on type patterns while losing accuracy on property selection. 7 regressions included namespace flips (Q55, Q82) and wrong property names (Q63, Q87). Only 2 improvements (Q43, Q53). The extra examples (7 total) may have overwhelmed the context.

## Current failure breakdown (v6, 78 failures — the best version)

| Category | Count | % |
|---|:---:|:---:|
| Both FAIL | 37 | 47% |
| Nodes OK, Preds FAIL | 34 | 44% |
| BGP OK, ops wrong | 3 | 4% |
| Nodes FAIL, Preds OK | 4 | 5% |

## Proven principles

1. **Simple prompts win**. v6's flat rule list outperformed v7's numbered priorities.
2. **Fewer examples > more examples**. 5-6 examples work; 7 caused regressions.
3. **dbo: preference is essential**. Every version that weakened it (v2, v4) regressed.
4. **Data signals hurt namespace selection**. Embedding scores (v2) and triple counts (v4) are anti-correlated with benchmark expectations.
5. **Structural rules stick**. DISTINCT, COUNT format, ORDER BY + LIMIT — once added, they're reliably applied.
6. **Entity linking improvements compound**. Unicode fallback (v6) was a permanent gain.

## v8 strategy

Revert to v6's exact prompt structure (flat rules, same order) but keep the v7 Film+COUNT example (Example 6) since it targets Q25. Drop Example 7 (Company) to stay at 6 examples. This is a conservative "v6 + one surgical addition" approach.
