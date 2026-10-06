# Final Results Analysis: DB26 (Claude and Qwen)

Final confirmed evaluation results on DB26, the primary benchmark, with a question by
question breakdown of every failure for Claude Sonnet 4.6 and Qwen 3.5 122B. Numbers are
read directly from `eval-results/claude-db26-full-final.json` and
`eval-results/qwen-db26-full-final.json`, the full raw evaluation output committed to
this repo.

---

## 1. Summary Results

| Model | Result-set match | Avg F1 | Avg steps | Median steps | Mode steps |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 27/50 (54%) | 0.6119 | 2.94 | 1.0 | 1 |
| Qwen 3.5 122B | 22/50 (44%) | 0.5109 | 3.82 | 2.0 | 1 |
| LIBER-AI-CLAUDE (pre-GSoC baseline) | — | 0.32 | — | — | — |
| LIBER-AI-QWEN (pre-GSoC baseline) | — | 0.31 | — | — | — |

Claude's F1 of 0.6119 is a 91% relative improvement over the pre-GSoC baseline of 0.32,
and is within a fraction of a point of 2nd place on the official Text2SPARQL 2026
leaderboard, which sits at F1=0.614.

## 2. What "27/50" and "22/50" Actually Mean

Result-set match is a strict exact match. Not every other question is a total miss.

| Model | Exact match | Partial credit | Completely wrong |
|---|---|---|---|
| Claude Sonnet 4.6 | 27 | 7 | 16 |
| Qwen 3.5 122B | 22 | 6 | 22 |

Claude's 23 non-exact questions split into 7 with meaningful partial F1 and 16 flat
zeros. Qwen's 28 non-exact questions split into 6 partial and 22 flat zeros. The partial
questions are close: several find the right entity through a shorter path than gold
expects, or return almost the full correct set and miss a row or two.

---

## 3. Failure Overlap Between Models

Claude has 23 non-exact questions, Qwen has 28. Of these, **22 questions fail on both
models**. Most of what is left unsolved is a structural limit of the pipeline, not a
weakness specific to either model.

| | Question IDs |
|---|---|
| Fails on both (22) | 1, 2, 4, 5, 11, 12, 14, 18, 19, 20, 22, 25, 26, 27, 28, 33, 34, 39, 40, 44, 48, 50 |
| Claude only (1) | 32 |
| Qwen only (6) | 17, 29, 31, 37, 41, 43 |

The single-model failures mostly come down to ordinary LLM non-determinism, not a real
gap. Q37, Stewart Bovell's motto, is a good example: the type-aware probe filtering fix
built in Week 11 was verified to recover this exact question on both models in isolated
testing. On this full run Qwen's first attempt took a slightly different path than the
one tested, and the retry budget ran out before it self-corrected. This kind of run to
run variation is documented across the Week 9 to 11 blog posts.

---

## 4. Failure Categories

Every failing question was checked against its validator action, validator reason, and
generated SPARQL to find the real cause rather than guessing. Seven categories emerged.

### A: No entity anchor, full-scan superlative questions

No starting entity URI at all. "Which X has the most Y" needs a full scan with nothing
for the agentic probe to attach to when the query fails.

| Question | Both models fail |
|---|---|
| Q2, actor who appeared in the most films in a single decade | Yes |
| Q5, top 3 taxonomic kingdoms by endangered species | Yes |

### B: Known fixes that did not trigger reliably on this run

The fix already exists in the code, either the triple direction rule added in Week 9
or the type-aware probe filtering added in Week 11, but the model or the probe does
not land on the fixed path on every attempt. Verified to work in isolated testing, so
these are non-deterministic misses, not a missing capability.

| Question | Claude | Qwen |
|---|---|---|
| Q1, presidents of France | Fail, partial 0.42 | Fail |
| Q11, prime ministers of the United Kingdom | Fail | Fail |
| Q19, footballers coached by Pep Guardiola | Fail | Fail |
| Q37, motto of the military unit Stewart Bovell served in | — | Fail, Qwen only |
| Q43, birthplace of the athlete trained by Ring | — | Fail, Qwen only |

### C: Probe directionality limitation

The property the query needs lives on an entity the probe cannot reach given the
direction the query is structured in. A known, documented boundary described in the
README under two-hop probe chaining.

| Question | Both models fail |
|---|---|
| Q18, countries where Denali's children peaks are located | Yes |
| Q33, origin of the musical styles associated with Back to Black | Yes |

### D: Entity linking or disambiguation gaps

The entity exists in DBpedia but the surface form does not resolve cleanly:
apostrophes, ambiguous short titles, or naming variants like Mac_OS versus MacOS.

| Question | Both models fail |
|---|---|
| Q14, birthplace of the badminton silver medallist | Yes |
| Q29, where was 'Imagine' recorded | Qwen only |
| Q39, software on both Linux and MacOS | Yes |

### E: Complex multi-condition or multi-hop joins

Questions needing more than the current two-hop chaining handles, either three or more
conditions joined together or a hop pattern that does not match the intermediate
variable shape the Validator looks for.

| Question | Claude | Qwen |
|---|---|---|
| Q17, TV series characters where Stephen Merchant is executive producer | — | Fail, Qwen only |
| Q20, cities the rivers from Yellowstone flow through | Fail, wrong rows | Fail |
| Q25, TV shows produced through a specific collaboration | Fail | Fail |
| Q28, TV shows related by shared opening theme composer | Fail | Fail |
| Q31, Obama's birthplace that is also a university location | — | Fail, Qwen only |
| Q40, academic affiliations at Metro Manila universities | Fail | Fail |

### F: Complex aggregation or counting edge cases

Entity linking and predicate choice are correct, but the aggregation logic, a COUNT, a
date ORDER BY, a GROUP BY ranking, produces a wrong number or order.

| Question | Claude | Qwen |
|---|---|---|
| Q4, top 5 bands or musicians by studio album count | Fail, partial 0.60 | Fail, wrong rows |
| Q12, 10 oldest works by Mark Twain | Fail | Fail |
| Q32, rivers crossed by structures designed by Gustave Eiffel | Fail, wrong count | — |
| Q48, other teams HC Slovan Bratislava members played for | Fail | Fail |
| Q50, count of movies in the French language | Fail, wrong count | Fail, wrong count |

### G: Partial credit, near misses

The pipeline found the right entity and mostly the right structure, and the result
overlaps with gold without being an exact match.

| Question | Claude F1 | Qwen F1 | Note |
|---|---|---|---|
| Q22, companies that built a club's former stadium | 0.80 | 0.80 | Same partial pattern on both |
| Q26, publishers of the Oddworld series | 0.40 | 0.40 | Dead URI fix recovers the entity, returns 3 direct values instead of the full 7-value two-hop set |
| Q27, films scored by Ennio Morricone | 0.99 | 0.99 | One row short of gold on both |
| Q34, countries that speak the same language as Brazil | 0.14 | 0.14 | Wrong or incomplete predicate on both |
| Q41, other firearms designed by the AK-47's creator | — | 0.96 | Qwen only |
| Q44, other honours held by Presidential Medal of Freedom recipients | 0.25 | 0.25 | 127 to 128 correct rows, wrong composition |

---

## 5. Category Summary

| Category | Description | Question count, union of both models |
|---|---|---|
| A | No entity anchor, full-scan superlative | 2 |
| B | Known fixes that did not trigger reliably | 5 |
| C | Probe directionality limitation | 2 |
| D | Entity linking or disambiguation gap | 3 |
| E | Complex multi-condition or multi-hop join | 6 |
| F | Complex aggregation or counting edge case | 5 |
| G | Partial credit, near miss | 6 |

A and C are hardest to close without a larger architectural change. B should improve
with less non-determinism rather than a pipeline fix, since the rule is already in the
prompt. G is closest to fully solved, several are within a row or two of exact.

---

## 6. What Would Move F1 Beyond the Current Best

22 of the hardest questions are shared between both models, a good sign that the
pipeline's limits are consistent and explainable rather than model-specific quirks.

1. **Reverse-direction probing.** The probe only ever checks the subject side of a
   relationship. Extending it to also check the object side when the subject side is
   empty would directly address Q18 and Q33, and likely a few similar ones.

2. **A wider hop budget for three-hop questions.** Two-hop probe chaining resolves one
   intermediate variable. Several Category E questions need a second level of chaining
   to reach the actual answer.

3. **Result-set aware column matching.** Several Category G questions are a legitimate
   subset or superset of gold. A scoring rule that matches on semantic content rather
   than exact row equality would likely lift some of these to full credit without any
   change to the pipeline itself.

4. **Full-scan query planning for superlative questions.** "Which X has the most Y"
   needs a different query strategy since there is no entity to link or probe. The
   hardest category to solve within the current architecture.

5. **Reducing non-determinism.** Category B and most of the single-model failures are
   not capability gaps, the fixes already exist and are verified to work. The next
   lever is running a question multiple times and taking a majority result, or lowering
   temperature further if the provider allows it, since the right answer is already
   reachable, just not reliably reached on every attempt.
