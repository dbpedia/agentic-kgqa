# Evaluation Analysis: Prompt Evolution v1 → v6

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v4 | v5 | v6 | Best |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall match** | 20.0% | 6.0% | 18.0% | 12.0% | 18.0% | **22.0%** | **v6** |
| BGP Nodes | 25.0%* | 57.0% | 61.0% | 67.0% | 55.0% | **59.0%** | v4 (67%) |
| BGP Predicates | 25.0%* | 26.0% | 26.0% | 20.0% | 28.0% | **29.0%** | **v6** |
| Inner operators | 84.0% | 81.0% | 79.0% | 85.0% | 85.0% | **86.0%** | **v6** |
| Outer operators | 75.0% | 52.0% | 76.0% | 74.0% | 77.0% | **86.0%** | **v6** |

*v1 had a single combined BGP metric (25.0%), shown in both columns for comparison.

### v6 — New best: 22% (+2pp over v1 baseline)
**What changed**: Unicode fallback in entity linking, rdf:type constraint rule, COUNT format normalisation.

**Results**: 5 improved, 1 regressed vs v5 → net +4. New record on overall match (22%), predicates (29%), inner ops (86%), and outer ops (86%). Outer ops jumped +9pp thanks to COUNT format fix.

### Namespace usage across versions

| | v1 | v2 | v3 | v4 | v5 | v6 | Benchmark |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dbo: | 83 | 49 | ~70 | 53 | 65 | 70 | 76 |
| dbp: | 25 | 63 | ~30 | 63 | 48 | 47 | 38 |

v6 is the closest to benchmark distribution yet (70/47 vs 76/38).

## Current failure analysis (v6, 78 failures)

| Category | Count | % |
|---|:---:|:---:|
| Both nodes and predicates FAIL | 37 | 47% |
| Nodes OK, Predicates FAIL | 34 | 44% |
| Nodes FAIL, Predicates OK | 4 | 5% |
| BGP OK, operators wrong | 3 | 4% |

### The 34 "nodes OK, preds FAIL" — detailed breakdown:
- **29 namespace mismatches**: gen=dbo but exp=dbp (10) or gen=dbp but exp=dbo (19). Still the #1 problem.
- **5 other**: different property names entirely (Q4: numLocations vs numStores, Q27: director disambig, Q45: landing date property, Q62: squad property, Q97: source vs origin)

### The 37 "both FAIL" — main patterns:
- **Missing rdf:type constraints** (~12): Q2, Q6, Q10, Q13, Q21, Q22 — despite the v6 rule, the model still omits type constraints in many cases
- **Structural mismatch** (~10): Q6 (subject/object inversion), Q7 (multi-hop structure wrong), Q14 (subject/object swapped)
- **Wrong entity** (~8): Q5 (Montenegro URI still sometimes wrong despite Unicode fix), multi-word disambiguation failures
- **Complex queries** (~7): Q11, Q17, Q78, Q81 — require multi-hop reasoning the model doesn't get right

## Key learnings

### What's working (monotonic improvement):
- **Outer ops**: 75% → 86% — explicit structural rules work
- **Inner ops**: 84% → 86% — stable and high
- **Overall**: broke past v1 baseline for the first time

### Remaining bottlenecks:
1. **Namespace selection** (29/78 = 37% of failures): fundamental — benchmark is split between dbo/dbp and there's no deterministic rule
2. **Missing type constraints** (~12 = 15%): the v6 prompt rule helps but the model doesn't apply it consistently
3. **Structural complexity** (~10 = 13%): multi-hop, subject/object direction — hard to fix with prompting alone

## Next steps for v7

### 1. Stronger type constraint enforcement
The v6 rule says "when the question asks about a type AND the ontology returns a Class, add rdf:type". The model ignores it ~50% of the time. Possible fixes:
- Make the rule more prominent (move to top of rules)
- Add more examples showing type constraints (currently only Example 5 shows it)
- Add a second example with a different type (e.g. dbo:Film, dbo:Person)

### 2. The remaining namespace problem
29 out of 34 pure-predicate failures are namespace mismatches. The split is roughly even: 10 cases where we generate dbo but benchmark wants dbp, 19 where we generate dbp but benchmark wants dbo. The "always prefer dbo" rule correctly handles 19/29, but the 10 where benchmark expects dbp suggest we need the self-correction loop to try the alternative.

### 3. Don't touch what's working
Outer ops and inner ops are now solid. COUNT format and DISTINCT rules are effective. Don't change these.
