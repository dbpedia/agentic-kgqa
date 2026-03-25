# Evaluation Analysis: Prompt Evolution v1 → v2 → v3 → v4

## Performance summary (DeepSeek V3.2, full benchmark N=100)

| Metric | v1 | v2 | v3 | v4 | Best |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Overall match** | 20.0% | 6.0% | **18.0%** | 12.0% | v1 (20%) |
| BGP Nodes | 25.0%* | 57.0% | 61.0% | **67.0%** | v4 (67%) |
| BGP Predicates | 25.0%* | 26.0% | 26.0% | **20.0%** | v3 (26%) |
| Inner operators | **84.0%** | 81.0% | 79.0% | **85.0%** | v4 (85%) |
| Outer operators | 75.0% | 52.0% | **76.0%** | 74.0% | v3 (76%) |

*v1 had a single combined BGP metric (25.0%), shown in both columns for comparison.

## Version history

### v1 — Baseline (20%)
- "Prefer ontology properties over dbp: when both are available"
- Single example using `dbo:birthPlace`
- No explicit DISTINCT rule
- Namespace usage: dbo=83, dbp=25 (benchmark: dbo=76, dbp=38)

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
- Namespace usage: dbo≈70, dbp≈30 (closer to benchmark)

### v4 — Frequency-weighted lookup (regression: 18% → 12%)
**What changed**:
1. Added triple counts from actual dataset to ontology lookup results (e.g. "dbo:director, 113,636 triples")
2. Prompt rule: "pick the property with MORE triples; never use 0-triple properties"
3. 5 few-shot examples (added COUNT and ORDER BY + LIMIT patterns)

**What went wrong**:
1. **Namespace flip again** (same v2 pattern): triple counts steered the model toward `dbp:` because many `dbp:` properties have MORE triples than their `dbo:` equivalents (e.g. `dbp:director` has 156k triples vs `dbo:director`'s 113k). The prompt said "pick the one with more triples" → model picked `dbp:` in 63 queries (v3 had 30).
2. **11 regressions**: Q8 (Eiffel Tower location), Q12 (Boeing manufacturer), Q15 (Louvre director), Q53 (NYC HQ), Q55 (Inception director), Q68 (Eiffel Tower), Q72 (Bohemian Rhapsody), Q73 (California capital), Q85 (Tom Hanks wife) — all flipped from correct `dbo:` to incorrect `dbp:`
3. **BGP Predicates dropped** from 26% to 20% — frequency data made the problem worse, not better

**What went right**:
- BGP Nodes improved again (61% → 67%) — the frequency data helps with entity-related properties
- Inner ops improved (79% → 85%) — the new COUNT and ORDER BY examples helped
- 5 questions improved: Q16, Q62, Q94, Q95, Q98

## The namespace paradox

| | v1 | v2 | v3 | v4 | Benchmark |
|---|:---:|:---:|:---:|:---:|:---:|
| Queries using dbo: | 83 | 49 | ~70 | 53 | 76 |
| Queries using dbp: | 25 | 63 | ~30 | 63 | 38 |

**The core paradox**: the benchmark expects `dbo:` in 76% of questions, but `dbp:` properties often have MORE triples in the actual dataset. Any signal based on data frequency (triple counts, embedding scores) pushes the model toward `dbp:`, away from what the benchmark expects.

This confirms the translation trade-off identified in v3 analysis:
- The benchmark expects **schema-faithful** translations (using curated `dbo:` properties)
- The data favours **data-faithful** translations (using raw `dbp:` properties with more coverage)
- **Triple counts are anti-correlated with benchmark expectations for namespace choice**

## Current failure analysis (v4, 88 failures)

| Category | Count | % of failures |
|---|:---:|:---:|
| Nodes OK, Predicates FAIL | 51 | 58% |
| Both nodes and predicates FAIL | 29 | 33% |
| BGP OK, operators wrong | 4 | 5% |
| Nodes FAIL, Predicates OK | 4 | 5% |

The predicate problem has gotten worse (51 pure predicate failures, up from 37 in v3). Frequency data actively misguides namespace selection.

## Key learning: what signals work and what don't

### Signals that HELP:
- **Entity linking** (Redis surface form lookup) — BGP Nodes improved steadily: 25% → 57% → 61% → 67%
- **Explicit structural rules** (DISTINCT, COUNT patterns) — Inner/outer ops are now solid when enforced
- **Few-shot examples** — each new example pattern gets adopted well

### Signals that HURT:
- **Embedding similarity scores** — `dbp:` scores inflate after reindex (v2 lesson)
- **Triple counts** — anti-correlated with benchmark's namespace preference (v4 lesson)
- **"Pick the higher/more" heuristics** — any ranking signal pushes toward `dbp:`

### The fundamental issue:
The benchmark's namespace preference is **not derivable from data statistics**. It appears to reflect the benchmark authors' modelling choice: prefer the curated ontology unless the data simply doesn't exist there. This is a **schema-level** decision, not a **data-level** one.

## Potential next steps (revised after v4)

### High impact: fixing the namespace problem

1. **Revert to hard dbo: preference** (like v1/v3) but with smarter fallback. The data shows that simple "prefer dbo:" outperforms any data-driven signal. Show triple counts in the prompt as context but DON'T tell the model to use them for namespace selection.

2. **Two-pass namespace fallback**: generate with `dbo:` first. If the self-correction loop gets 0 results, the revision prompt already knows to try `dbp:`. This leverages the existing infrastructure without changing the default preference.

3. **Explicit dbo/dbp mapping in ontology results**: instead of showing both as separate results, group them: "dbo:director (113k triples) / dbp:director (156k triples) — use dbo: by default". This frames the choice clearly.

### Medium impact: improving the remaining 33% "both FAIL"

4. **Entity linking improvements**: the 29 "both FAIL" cases often involve wrong entity URIs (multi-word entities, disambiguation). Fuzzy matching or Unicode normalisation (en dash vs hyphen) in the Redis lookup could help.

5. **Structural pattern library**: the 4 "BGP OK, ops wrong" cases are low-hanging fruit. Add explicit rules for remaining patterns (GROUP BY, HAVING, subqueries).

### What NOT to do

6. **Don't add more data-driven signals to namespace selection** — v2 and v4 both proved this makes things worse. The benchmark's namespace preference is a schema convention, not a data pattern.

7. **Don't over-engineer the prompt** — diminishing returns. v1 (20%) and v3 (18%) had the simplest prompts. Complexity in v2 (6%) and v4 (12%) caused regressions. Keep it simple.
