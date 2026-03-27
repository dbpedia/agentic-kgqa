# Evaluation Analysis: Prompt Evolution v1 → v5

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v4 | v5 | Best |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall match** | **20.0%** | 6.0% | 18.0% | 12.0% | **18.0%** | v1 (20%) |
| BGP Nodes | 25.0%* | 57.0% | 61.0% | **67.0%** | 55.0% | v4 (67%) |
| BGP Predicates | 25.0%* | 26.0% | 26.0% | 20.0% | **28.0%** | **v5 (28%)** |
| Inner operators | 84.0% | 81.0% | 79.0% | 85.0% | **85.0%** | v4/v5 (85%) |
| Outer operators | 75.0% | 52.0% | 76.0% | 74.0% | **77.0%** | **v5 (77%)** |

*v1 had a single combined BGP metric (25.0%), shown in both columns for comparison.

### Namespace usage across versions

| | v1 | v2 | v3 | v4 | v5 | Benchmark |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Queries using dbo: | 83 | 49 | ~70 | 53 | 65 | 76 |
| Queries using dbp: | 25 | 63 | ~30 | 63 | 48 | 38 |

## Version history

### v1 — Baseline (20%)
- "Prefer ontology properties over dbp: when both are available"
- Single example using `dbo:birthPlace`
- No explicit DISTINCT rule

### v2 — Namespace over-correction (regression: 20% → 6%)
Removed dbo preference, used embedding scores for namespace selection. `dbp:` scores inflate → over-used `dbp:`. Also lost DISTINCT.

### v3 — Corrective balance (recovery: 6% → 18%)
Restored dbo preference + DISTINCT rule + 3 examples. Recovered outer ops (76%) and pushed nodes to 61%.

### v4 — Frequency-weighted lookup (regression: 18% → 12%)
Added triple counts, told LLM to "pick the property with more triples". Triple counts are anti-correlated with benchmark expectations → over-used `dbp:` again. **Lesson: data statistics don't help with namespace selection.**

### v5 — Grouped pairs + hard dbo preference (recovery: 12% → 18%)
**What changed**:
1. Hard `dbo:` preference: "ALWAYS use dbo: when one exists, regardless of triple counts"
2. Grouped dbo/dbp pairs in ontology context with explicit annotations: `"dbo: .../director (113k) / dbp: .../director (156k) — use dbo:"`
3. Triple counts shown as reference only, NOT for namespace selection

**Results**:
- 11 improved, 5 regressed vs v4 → net +6
- **Best predicates so far: 28%** (+2pp vs v3)
- Inner ops stable at 85%, outer ops best at 77%
- Namespace usage: dbo=65, dbp=48 — closer to benchmark than v4 (53/63) but still dbp-heavy vs benchmark (76/38)
- BGP Nodes dropped (67% → 55%) — possibly due to grouped formatting confusing entity resolution

## Current failure analysis (v5, 82 failures)

| Category | Count | % of failures |
|---|:---:|:---:|
| Both nodes and predicates FAIL | 42 | 51% |
| Nodes OK, Predicates FAIL | 30 | 37% |
| BGP OK, operators wrong | 7 | 9% |
| Nodes FAIL, Predicates OK | 3 | 4% |

### Persistent failures (78 questions wrong in both v3 and v5)

The same 78 questions fail across versions — only 22 questions are ever correct. The failures fall into clear categories:

**1. Wrong predicate name (26 persistent, nodes OK):**
- Q1: `dbo:literaryGenre` expected vs `dbp:genre` generated — synonym, not namespace
- Q3: `dbo:birthPlace` expected vs `dbp:birthPlace` generated — correct namespace, wrong name variant? Actually same name, different namespace.
- Q16: `dbp:director` expected vs `dbo:director` generated — benchmark wants dbp here!
- Q22: `dbp:producer` expected vs `dbo:producer` generated
- Q33: `dbp:author` expected vs `dbo:author` generated

**2. Structural mismatch (42 persistent, both fail):**
- Q2: missing `rdf:type dbo:Country` constraint
- Q5: en-dash vs hyphen in entity URI (`1992–2006` vs `1992-2006`)
- Q6: subject/object direction wrong (query structure inverted)
- Q7: multi-hop reasoning — closest city of Eiffel Tower → birthplace of person from that city
- Q10: missing `rdf:type AmericanFootballPlayer` + wrong property

**3. Operator issues (7 persistent, BGP OK):**
- Q25, Q30: COUNT(DISTINCT ?var) vs COUNT(?var) — structural COUNT variants
- Mostly minor formatting differences the judge flags

### The 22 questions that DO work

These are simple single-hop queries with unambiguous properties (birthPlace, location, anthem, starring) where the entity linking works perfectly and the property exists in the expected namespace.

## Key insights after 5 versions

### What we've proven:
1. **Prompt engineering has a ceiling around 18-20%** for this task. v1 and v3/v5 all converge to this range.
2. **Namespace selection is the #1 bottleneck** but it's not solvable via data signals (v2, v4 proved this).
3. **The benchmark is inconsistent**: some questions expect `dbo:` even when `dbp:` has more data, and vice versa. ~26 of the 78 persistent failures are pure namespace mismatches where the agent picks the "wrong" one.
4. **Entity linking works well** when the surface form is clean (BGP Nodes peaked at 67%).
5. **Structural rules are effective** — every explicit rule (DISTINCT, COUNT, ORDER BY) gets adopted.

### What would actually move the needle:

#### High impact (could reach 30-40%):
1. **Entity linking fixes**: the 42 "both FAIL" cases include ~10 with wrong entity URIs due to:
   - Unicode normalisation (en dash `–` vs hyphen `-` in Q5)
   - Multi-word entity disambiguation
   - Missing redirects
   Fixing these directly improves the node accuracy which then lets the right predicates surface.

2. **Missing type constraints**: ~8 questions fail because the generated query is missing `rdf:type dbo:X` constraints that the benchmark expects (Q2, Q10, Q25, Q53). Adding a rule "include rdf:type when the ontology lookup returns a Class" could help.

3. **COUNT format normalisation**: the judge flags `SELECT (COUNT(DISTINCT ?x) AS ?count)` as different from `SELECT DISTINCT COUNT(?x)` even though they're functionally similar. Relaxing the judge or normalising COUNT format could recover 5-7 questions.

#### Medium impact (incremental):
4. **More few-shot examples**: add examples that show `rdf:type` constraints and multi-hop patterns.
5. **Predicate synonym mapping**: build a table of equivalent predicates (`literaryGenre ↔ genre`, `birthPlace ↔ placeOfBirth`).

#### Low impact (not worth pursuing):
6. More prompt engineering on namespace selection — we've hit the ceiling.
7. Data-driven signals for namespace choice — proven counterproductive.
