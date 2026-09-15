# Experiment log

Every experiment in the order it was run, with the data it used, the result
and the decision it led to. Failed and superseded experiments are kept: they
explain why the pipeline looks the way it does.

Conventions used throughout:

- **R@k**: share of queries with at least one correct pandas name in the top k results.
- **Honest subset**: queries whose answer's name does not appear in the question title.
  Queries like "How to use dropna" are easy for keyword search and hide real differences.
- **Gap and 95% CI**: difference in R@5 between two methods on the same queries, with a
  paired bootstrap interval (10,000 resamples). R@5 was fixed as the comparison metric
  before any comparison was run. A gap counts as real only if its interval excludes zero.
- **Silver** labels are automatic; **gold** labels are hand-made (see `annotations/README.md`).

---

## 1. How much documentation is in the pandas source?

Parsed `core`, `io`, `plotting`, `errors` and `api` with `ast` and kept docstrings of at least 20 words.

| Files parsed | Docstrings kept | Words | Median words |
|---|---|---|---|
| 253 | 1,678 | 345,188 | 140 |

**Decision:** API docstrings are long enough to be a usable corpus. The user guide (reStructuredText) was left for later.

## 2. First benchmark: BM25 vs embeddings

Silver queries from the 500 most-voted pandas questions: 359 labelled queries.
Embeddings: `all-MiniLM-L6-v2`. Both methods saw identical document text.

| Subset | n | BM25 R@5 | Embeddings R@5 |
|---|---|---|---|
| All | 359 | 0.39 | 0.45 |
| Honest | 257 | 0.28 | 0.36 |

Honest-subset gap +0.09, 95% CI [+0.03, +0.14].

**Decision:** before trusting any number, check whether the automatic labels are right.

## 3. Manual review of 50 silver labels

A random sample of 50 labels (seed 42) was judged against each accepted answer.

| Correct | Loose (right name plus extras) | Wrong |
|---|---|---|
| 17 (34%) | 25 (50%) | 8 (16%) |

Causes of the 8 wrong labels: 4 name collisions with NumPy or Python (`astype`, `format`,
`lower`, `dtype`), 2 correct answers missing from the corpus (`set_option`, the
DataFrame constructor), 2 answers that are not a function at all (`&` operators, a
conceptual explanation).

**Decision:** fix the labeller and the corpus, and build a hand-labelled gold test set.

## 4. Labeller and corpus fixes

- Labeller: walk each method call back to its root, dropping calls rooted in NumPy,
  other modules or string literals, and requiring `.str` for string-only methods.
- Corpus: keep only names reachable from `import pandas`, add `_config` (`set_option`),
  attributes documented through `doc=` (`DataFrame.columns`, `DataFrame.index`) and
  method aliases (`agg = aggregate`). Result: 1,223 documents, 316,983 words.

| Subset | n | BM25 R@5 | Embeddings R@5 |
|---|---|---|---|
| Honest | 257 | 0.29 | 0.38 |

**Decision:** scores moved little but now measure the public API; proceed to gold labels.

## 5. Gold set, round 1 (100 questions)

85 answerable questions, 62 in the honest subset. Embeddings minus BM25 at R@5 on the
honest subset: +0.08, 95% CI [-0.03, +0.19].

**Decision:** 62 questions cannot establish the gap. More gold labels are needed.

## 6. More silver data

Fetched 25 pages (2,500 questions, the anonymous API limit). 1,804 silver queries.

| Subset | n | BM25 R@5 | Embeddings R@5 | Gap (95% CI) |
|---|---|---|---|---|
| Honest | 1,241 | 0.29 | 0.32 | +0.03 [+0.00, +0.06] |

**Finding:** the base embedding model's advantage over BM25 shrank on lower-voted questions.

## 7. Leakage-safe split and first fine-tuning

Gold questions were removed from silver by id and by normalised title, then an 80/20
train/validation split was drawn (seed 13). The split fails with an error if any gold
question reaches training data.

Fine-tuning `all-MiniLM-L6-v2` with MultipleNegativesRankingLoss (3 epochs, batch 32,
one preferred document per name, questions with more than 3 labels skipped):

| Validation R@5 before | after |
|---|---|
| 0.436 | 0.564 |

A scoring mistake was caught here: the first report scored the fine-tuned model on all
silver queries, including those it was trained on. Silver scoring was restricted to the
validation split.

On the 62 honest gold questions, fine-tuned minus base embeddings at R@5:
+0.03, 95% CI [-0.06, +0.13] (any correct name).

**Decision:** gold is now the bottleneck; grow it.

## 8. Gold set, round 2 (300 questions)

200 more questions from all 2,500 (seed 8), appended without changing round 1.
256 answerable, 189 in the honest subset; 5 unanswerable rows are corpus gaps.
The split was re-run so no new gold question stayed in training data, and the model was retrained.

| Honest gold (n=189), R@5 | Gap vs base embeddings (95% CI) |
|---|---|
| Any correct name | +0.10 [+0.04, +0.15] |
| Primary name only | +0.04 [-0.02, +0.10] |

**Decision:** fine-tuning helps find a correct answer, but not yet the single best one.
Next: harder training examples.

## 9. Hard negatives

For each training pair, one negative was mined with the base model: the highest-ranked
document below rank 3 whose name is not a labelled answer. Ranks 1-3 are skipped because
loose labels miss correct answers, and those sit near the top.

Validation R@5 across three seeds:

| Seed | Without hard negatives | With hard negatives |
|---|---|---|
| 1 | 0.552 | 0.675 |
| 2 | 0.552 | 0.675 |
| 3 | 0.562 | 0.685 |

Seed 3 was chosen for both configurations on validation, before gold was scored.

Gold, honest subset (n=189), R@5:

| Method | Any correct name | Primary name only |
|---|---|---|
| BM25 | 0.38 | 0.28 |
| Base embeddings | 0.42 | 0.32 |
| Fine-tuned | 0.50 | 0.35 |
| Fine-tuned + hard negatives | 0.63 | 0.47 |

| Comparison (R@5, 95% CI) | Any correct name | Primary name only |
|---|---|---|
| Hard negatives vs fine-tuned | +0.13 [+0.07, +0.19] | +0.11 [+0.06, +0.16] |
| Hard negatives vs BM25 | +0.25 [+0.17, +0.33] | +0.19 [+0.11, +0.26] |

A sample of mined negatives showed some that are arguably correct (for example
`Series.dtype` for a dtype question), which is the label-noise risk the rank skip reduces
but does not remove.

**Decision:** hard negatives become part of the pipeline.

## 10. Cross-encoder reranking (not adopted)

A cross-encoder re-scored the hard-negative retriever's top results. All comparisons are
on validation (n=308), reranking the top 10, with 256-token inputs.

| Reranker | R@1 | R@5 | R@5 gap vs retriever (95% CI) |
|---|---|---|---|
| None (retriever only) | 0.43 | 0.69 | |
| `ms-marco-MiniLM-L6-v2`, off the shelf | 0.31 | 0.64 | -0.05 [-0.08, -0.01] |
| Fine-tuned, negatives from retriever top 20 | 0.30 | 0.66 | -0.03 [-0.07, +0.01] |
| Fine-tuned, same but skipping the top 3 | 0.35 | 0.68 | -0.01 [-0.05, +0.03] |

Reranking deeper (top 20 or 50) was worse in every configuration. The off-the-shelf
reranker, at 512 tokens, turned 65 correct first results into wrong ones and fixed 33,
typically preferring a near-duplicate function (`merge_ordered` over `merge`,
`itertuples` over `iterrows`).

**Decision:** no reranker beat the retriever on validation, so none was scored on gold
and reranking is not part of the pipeline. Likely causes, not tested separately: loose
training labels, 1,636 positive pairs, and a retriever already trained on near-misses.

## 11. Answer generation: choosing a prompt

Generator: `Qwen/Qwen2.5-1.5B-Instruct`, run locally in bfloat16 with greedy decoding. It
fits the 4 GB GPU with a peak of about 3.1 GiB. Each answer is written from the hard-negative
retriever's top 3 documents, each cut to 200 words, and must cite them as `[1]`-`[3]` or reply
with an exact not-found sentence.

A first check on 3 validation questions found no citations, invented example output,
and an uncited answer from the model's own knowledge when the retrieved documents did not
contain the answer.

Three prompts were compared on 40 validation questions, all seeing the same retrieved
documents. Scoring is automatic. In 26 questions a retrieved document carried a silver label
("answer in context"); in 14 none did.

| Prompt | Cites an excerpt | Cites a missing excerpt | Cites a labelled document | Abstains, answer in context | Abstains, answer not in context | s/answer |
|---|---|---|---|---|---|---|
| A: rules in the system message | 2/40 | 2/40 | 2/40 | 0/26 | 1/14 | 10.1 |
| B: stricter rules after the question | 2/40 | 1/40 | 1/40 | 2/26 | 2/14 | 5.5 |
| C: B plus two worked examples | 12/40 | 2/40 | 7/40 | 4/26 | 6/14 | 5.5 |

B shortened answers but did not change citation behaviour; the worked examples did. B
changed several things at once (rule placement and wording), so which part shortened the
answers is not known.

**Decision:** C becomes the generation prompt. Citation remains the weak point: 28 of 40
answers cite nothing.

## 12. Answer generation on the gold set

Prompt C, `Qwen2.5-1.5B-Instruct` and the hard-negative retriever's top 3 documents, run once
on all 300 gold questions (about 18 minutes including model loading). Scored with the
automatic checks fixed before the run.

| Answerable questions (n=256) | |
|---|---|
| Retriever put a correct document in the 3 excerpts | 155 (61%) |
| Answer cites an excerpt | 63 (25%) |
| Answer cites a number with no excerpt | 0 (0%) |
| Answer cites a correct document | 30 (12%) |
| Answer wrongly abstains | 82 (32%) |

| Subset | Measure | Result |
|---|---|---|
| Correct document retrieved (n=155) | Answer cites a correct document | 30 (19%) |
| | Answer wrongly abstains | 41 (26%) |
| No correct document retrieved (n=101) | Answer abstains | 41 (41%) |
| Unanswerable (n=44) | Answer abstains | 20 (45%) |
| | Answer cites an excerpt anyway | 7 (16%) |

Reading a sample of answers showed that citation grading is both too strict and too
lenient. Too strict: an answer recommending the right function `fillna` cited the `replace`
excerpt, and an answer naming the right `reset_index` cited nothing. Too lenient: an answer
correctly citing `to_json` suggested `orient='records'` without the needed `lines=True`, and
another invented a `level='all'` argument. Some answers refused although the first excerpt
held the answer.

A second measure was added after that reading, not fixed in advance: whether the answer's
text names a correct function as a whole word, regardless of citation.

| Measure (added after reading answers) | Result |
|---|---|
| Answerable: names a correct function | 123/256 (48%) |
| ... of those, also cites a correct document | 26/123 (21%) |
| Correct document retrieved: names a correct function | 98/155 (63%) |
| No correct document retrieved: names a correct function anyway | 25/101 (25%) |

The last row is knowledge from the model itself rather than from the excerpts, which is
the ungrounded behaviour citations are meant to expose.

**Finding:** retrieval supplies a correct document for 61% of answerable questions, but the
1.5B generator uses and cites it correctly for 19% of those. The generator, not retrieval,
is now the weakest stage. Naming a correct function is far more common (63%) than citing it,
so answers are more useful than the strict citation score suggests, but they are not
reliably grounded.
