# Experiment log

Every experiment in the order it was run, with the data it used, the result
and the decision it led to. Failed and superseded experiments are kept: they
explain why the pipeline looks the way it does.

Conventions used throughout:

- **R@k**: share of queries with at least one correct pandas name in the top k results.
- **Honest subset**: queries whose answer's name does not appear in the question title. For
  gold this checks the primary name; for silver, any label. Queries like "How to use dropna"
  are easy for keyword search and hide real differences.
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

## 13. Closing corpus gaps

While labelling gold, 5 questions were marked unanswerable because their answer exists in
pandas but not in the corpus. Each had a different cause:

| Question needs | Why it was missing | Fix |
|---|---|---|
| `show_versions` | lives in `pandas/util`, which was not parsed | parse `util`; the public-name filter still drops its internal helpers |
| `dt.year` and other date fields | docstring passed as an argument: `year = _field_accessor("year", "Y", """...""")` | read the last argument of `_field_accessor`, including names bound to `textwrap.dedent("""...""")` |
| `dt.floor` | defined on `TimelikeOps`, a base class the filter dropped | add `DatelikeOps` and `TimelikeOps` to the user-reachable classes |
| `GroupBy.ngroups` (2 questions) | has no docstring in the pandas source | none; writing documentation ourselves would add text pandas does not have |

The corpus grew from 1,223 to 1,265 documents (42 added, none removed). As the labelling
rules planned, only the rows with a `corpus gap:` note changed: 3 became answerable
(`show_versions`, `floor`, `year`) and one gained `year month` as other correct names.
Gold now has 259 answerable questions, 191 in the honest subset.

Every later step was rerun with unchanged settings: silver labels (1,806), split
(train 1,234, validation 308), both retrievers with seed 3, and evaluation.

Gold, honest subset (n=191), R@5:

| Method | Any correct name | Primary name only |
|---|---|---|
| BM25 | 0.38 | 0.28 |
| Base embeddings | 0.42 | 0.32 |
| Fine-tuned | 0.49 | 0.34 |
| Fine-tuned + hard negatives | 0.61 | 0.43 |

| Comparison (R@5, 95% CI) | Any correct name | Primary name only |
|---|---|---|
| Hard negatives vs fine-tuned | +0.12 [+0.06, +0.17] | +0.09 [+0.04, +0.14] |
| Hard negatives vs BM25 | +0.23 [+0.16, +0.30] | +0.15 [+0.08, +0.23] |

The final model scored lower than in experiment 9 (0.63 and 0.47). To find out why, the
previous model (kept locally) and the retrained one were scored on both corpora, as a
diagnostic after the decision to keep the retrained model had been made:

| Gold labels | Corpus | Previous model | Retrained model |
|---|---|---|---|
| before this fix (n=189) | before | 0.630 / 0.466 | 0.608 / 0.429 |
| before this fix (n=189) | after | 0.630 / 0.466 | 0.608 / 0.429 |
| after this fix (n=191) | after | 0.634 / 0.471 | 0.607 / 0.429 |

(any correct name / primary name only)

The new documents did not change the previous model's score, so the drop comes from
retraining, not from the corpus. Retrained minus previous, paired bootstrap: -0.03
[-0.06, +0.01] for any correct name and -0.04 [-0.07, -0.01] for the primary name.

**Finding:** the same recipe and seed, on silver data that changed by two questions and a
reshuffled split, moves gold R@5 by 3-4 points. Differences of that size between single
training runs are not evidence that one setting is better; the comparisons above are
larger than that.

**Decision:** keep the retrained model, since it is what the documented commands produce.
Choosing the previous model because it scored higher on gold would select a model on the
test set.

## 14. Raising the citation rate (not adopted)

Experiment 12 found that answers often name the right function without citing its excerpt.
Two prompts aimed at that were compared with prompt C on validation questions, using the
retriever from experiment 13. A decision rule was fixed before the first run: adopt the
prompt with the most answers citing a labelled document, provided that (1) answers naming a
labelled function fall by at most 5 against C, and (2) abstentions when no labelled document
was retrieved do not fall.

- **D**: write the excerpt number right after every function named, `DataFrame.dropna [1]`.
- **E**: end with a fixed last line, `Sources: [1]`.

Round 1, validation questions 1-100 (labelled document retrieved for 57, not for 43):

| Prompt | Cites an excerpt | Cites a labelled document | Names a labelled function | Abstains, labelled doc retrieved | Abstains, not retrieved |
|---|---|---|---|---|---|
| C | 27 | 12 | 47 | 15/57 | 18/43 |
| D | 9 | 3 | 35 | 25/57 | 23/43 |
| E | 51 | 27 | 35 | 23/57 | 24/43 |

E more than doubled correct citations but failed condition 1. Reading the 12 questions
where C named a labelled function and E did not: E abstained in 9 of them, several with the
answer in an excerpt (for example refusing a `reset_index` question with `reset_index` as
excerpt 2). Some of C's named functions in those answers came with code the excerpts did
not contain.

- **F**: E, with the refusal rule narrowed: answer with an excerpt's function if it solves
  the question even when the excerpt does not show the exact case, and refuse only when no
  excerpt is about the question.

Round 2, questions 101-200, not read before (labelled document retrieved for 60, not for
40), so F was not tuned on the questions it was judged on:

| Prompt | Cites an excerpt | Cites a labelled document | Names a labelled function | Abstains, labelled doc retrieved | Abstains, not retrieved |
|---|---|---|---|---|---|
| C | 27 | 15 | 45 | 16/60 | 17/40 |
| E | 50 | 22 | 34 | 25/60 | 24/40 |
| F | 59 | 24 | 38 | 20/60 | 21/40 |

**Decision:** C stays. F raised correct citations from 15 to 24 but named a labelled
function 7 fewer times, beyond the limit fixed in advance. The prompts trade citations for
refusals: the stricter format makes the 1.5B model cite more often and also refuse more
often, including when the answer was retrieved. Seconds per answer are not compared: the
first round shared the machine with other work.

## 15. Answer generation on gold after the corpus fix

Prompt C was run once more on all 300 gold questions, because the retriever and corpus
changed in experiment 13. The "names a correct function" measure is now computed by
`scripts/evaluate_generation.py`; on the saved answers of experiment 12 it reproduces the
hand-computed 123/256, 98/155 and 25/101.

| Measure | Experiment 12 | Now |
|---|---|---|
| Answerable questions | 256 | 259 |
| Correct document in the 3 excerpts | 155 (61%) | 156 (60%) |
| Cites a correct document | 30 (12%) | 35 (14%) |
| ... when a correct document was retrieved | 30/155 (19%) | 35/156 (22%) |
| Names a correct function | 123 (48%) | 131 (51%) |
| ... when a correct document was retrieved | 98/155 (63%) | 101/156 (65%) |
| Cites a number with no excerpt | 0 | 1 |
| Wrongly abstains | 82 (32%) | 80 (31%) |
| Unanswerable: abstains | 20/44 (45%) | 19/41 (46%) |

The prompt did not change; the differences are within what the retriever change in
experiment 13 can explain and are not evidence of an improvement.

## 16. Release audit: removing wrongly attributed documents

A code review before release found two corpus problems:

- `core/interchange/dataframe_protocol.py` defines an abstract interchange-protocol class that
  is also named `DataFrame`. The public-name filter kept it, so `DataFrame.metadata` and
  `DataFrame.get_chunks` were indexed as if they were pandas DataFrame methods. Before the fix
  they appeared in the top 5 for 1 gold and 2 validation questions; none is a label.
- Source files were read in file-system order, which is not guaranteed to be the same on every
  machine. With two documents named `DataFrame`, that order decided which one training used.

The file is now excluded by name and files are read in sorted order. The corpus lost exactly
those 3 documents (1,262 remain, all others unchanged). Silver labels, train and validation
splits were identical before and after. Both retrievers were retrained with seed 3 and
everything was re-evaluated; the retrained model is used, as in experiment 13.

Validation R@5 (any correct name): fine-tuned 0.536, hard negatives 0.649 (0.633 before).

Gold, honest subset (n=191), R@5:

| Method | Any correct name | Primary name only |
|---|---|---|
| BM25 | 0.38 | 0.28 |
| Base embeddings | 0.42 | 0.32 |
| Fine-tuned | 0.49 | 0.34 |
| Fine-tuned + hard negatives | 0.62 | 0.43 |

| Comparison (R@5, 95% CI) | Any correct name | Primary name only |
|---|---|---|
| Hard negatives vs fine-tuned | +0.13 [+0.08, +0.19] | +0.09 [+0.05, +0.15] |
| Hard negatives vs BM25 | +0.25 [+0.17, +0.32] | +0.16 [+0.08, +0.23] |

Answer generation with prompt C on all 300 gold questions:

| Measure | Experiment 15 | Now |
|---|---|---|
| Correct document in the 3 excerpts | 156/259 (60%) | 157/259 (61%) |
| Cites a correct document | 35 (14%) | 33 (13%) |
| ... when a correct document was retrieved | 35/156 (22%) | 33/157 (21%) |
| Names a correct function | 131 (51%) | 132 (51%) |
| ... when a correct document was retrieved | 101/156 (65%) | 101/157 (64%) |
| Cites a number with no excerpt | 1 | 0 |
| Wrongly abstains | 80 (31%) | 79 (31%) |
| Unanswerable: abstains | 19/41 (46%) | 19/41 (46%) |

**Finding:** all changes are within run-to-run variation (experiment 13). The fix is about
correctness of what the search shows, not about scores.

## 17. Linking named functions to their sources (adopted)

Experiment 14 found that prompts raise citations only by raising refusals. This experiment
leaves the prompt and every answer unchanged and makes the link in code instead
(`named_indices` in `rag_for_pandas/generation.py`): an excerpt is **named** when the answer
writes that excerpt's function, and an answer **points to** an excerpt when it cites or
names it.

Fixed before any answer was read:

- design the rule on validation questions 1-100, where answers may be read and the rule changed;
- judge it once on validation questions 201-308, which no earlier experiment used;
- then score gold once, from the answers saved in experiment 16;
- adopt if, on the judging questions, (1) more answers point to a labelled document than cite
  one, with a paired bootstrap interval that excludes zero, and (2) reading every name link
  finds at most 5% wrong.

**Design (questions 1-100).** Matching a function's name as any whole word gave 102 links, and
reading them showed why that is too loose: many pandas names are ordinary words. "Cumulative
sum" linked `Expanding.sum`, "find the count" linked `DataFrame.count`, and "merge" used as a
verb linked `merge`. The rule therefore counts a name only when it is written as code: after
a dot (`DataFrame.dropna`, `df.dropna`), as a call (`concat(`) or inside backticks. Three
narrower guards followed:

| Guard | Reason |
|---|---|
| Skip names after `np.` and the other non-pandas roots the silver labeller already skips | `np.mean` linked `DataFrame.mean` (seen) |
| Skip a bare call when Python has a built-in of that name, such as `sum(x)` | not seen as a wrong link; such calls are common in answer code |
| Skip "the `mode` parameter" and "parameter `mode`" | not seen as a wrong link; 10 of the 100 answers name a parameter this way |

Names are compared by their last part, as in the rest of the project, so `df.dropna()` links
a `Series.dropna` excerpt as well. When the answer writes one excerpt's name in full, only
that excerpt is linked. After reading the design questions and before scoring the judging
ones, a link to a same-named method of another class was defined as a separate category
rather than as wrong; a wrong link is one where the matched text is not that pandas name at
all (an ordinary word, a parameter, another library's function).

With the final rule the design questions gave 68 links, none wrong and 15 to a same-named
method of another class.

**Judging (questions 201-308, labelled document retrieved for 57, not for 51).** Reproduced
by `python scripts/compare_prompts.py --prompts C --start 200 --questions 108`:

| Measure | Design, 1-100 | Judging, 201-308 |
|---|---|---|
| Cites a labelled document | 12/100 | 12/108 |
| Cites or names a labelled document | 33/100 | 32/108 |
| Gain (95% CI) | | +0.185 [+0.111, +0.259] |
| Name links read | 68 | 61 |
| ... wrong | 0 | 1 (1.6%) |
| ... same-named method of another class | 15 | 11 (18%) |

The wrong link: an answer used matplotlib's `plt.scatter`, which marked the pandas
`PlotAccessor.scatter` excerpt. Both conditions are met. The rule was not changed after the
judging questions were scored, so this case is still linked.

**Gold, scored once** from the saved answers, so the answers are the ones scored in
experiment 16:

| Measure | Cites | Cites or names |
|---|---|---|
| Answerable (n=259): a correct document | 33 (13%) | 92 (36%) |
| ... when a correct document was retrieved (n=157) | 33 (21%) | 92 (59%) |
| Answerable: any excerpt | 67 (26%) | 140 (54%) |
| Unanswerable (n=41): any excerpt | 8 (20%) | 13 (32%) |

Names a correct function (132, and 101 of the 157) and refusals are unchanged, because no
answer changed. Of the 101 answers that name a correct function with a correct document
retrieved, 87 now point to a correct document; the other 5 of the 92 cite one without
naming it.

**Decision:** adopted. `/answer` returns `named` beside `cited` for each source, and the page
shows both.

**Limits.**

- A link says which function the answer names. It does not check that the answer's statements
  agree with the excerpt: experiment 12 found answers that cite the right document and still
  invent an argument.
- About one link in five is to a same-named method of another class, such as
  `MultiIndex.duplicated` for an answer about `DataFrame.duplicated`.
- On unanswerable questions more excerpts are now marked (13 against 8). The model named
  those functions before; the link makes it visible.
- Links were classified by one unblinded reading: the reader knew the rule.

## 18. Warning for functions from outside the sources (adopted)

Experiment 12 found answers that name a correct function although no correct document was
retrieved: the model answering from its own knowledge. Experiment 17 made the grounded case
visible; this one flags the ungrounded case. An answer is **warned** when its prose writes a
documented pandas name as code, by the same test as `named_indices`, and none of its three
excerpts carries that name (`names_outside_sources` in `rag_for_pandas/generation.py`).

**Design.** Shaped on validation questions 1-100 and 201-308, whose answers had already been
read for experiment 17. Two choices came from that data:

- Only prose is read; fenced code blocks are removed first. Reading example code too raised
  the flagged names on those questions from 63 to 89, the additions being mostly helper calls
  such as `apply`, `columns` and `index` that the answer is not recommending.
- `DataFrame` and `Series` are skipped, as the silver labeller already does.

All 63 names flagged on the design questions were pandas names the answer recommended.

Fixed before the judging answers were read: judge once on validation questions 101-200,
generated with the current retriever and not read before, and adopt if reading every flagged
name finds at most 5% wrong (the matched text is not that pandas name: an ordinary word, a
parameter, a variable, another library's function).

**Judging (questions 101-200, labelled document retrieved for 62, not for 38).** The rule
flagged 33 names in 24 answers. None was clearly wrong. One is unclear: "the `map()`
function" could be Python's or pandas'. Two were the class of one of the excerpts (`Styler`
for an answer built on `Styler.background_gradient`, and `DatetimeIndex`): the pandas name,
but a warning nobody needs. The condition is met.

One change was made after judging and is therefore not covered by that reading: the classes
the excerpts belong to now count as in the sources. It can only remove flags (33 to 31 names
on the judging questions, 63 to 62 on the design questions). With the final rule,
reproduced by `python scripts/compare_prompts.py --prompts C --start N --questions M`:

| | 1-100 | 201-308 | 101-200 (judging) |
|---|---|---|---|
| Answers warned | 26/100 | 20/108 | 22/100 |
| ... labelled document retrieved | 10/57 | 6/57 | 12/62 |
| ... not retrieved | 16/43 | 14/51 | 10/38 |
| Names flagged | 33 | 29 | 31 |

**Gold, scored once** from the saved answers of experiment 16:

| Group | Answers warned |
|---|---|
| Answerable (n=259) | 69 (27%) |
| ... correct document retrieved (n=157) | 32 (20%) |
| ... no correct document retrieved (n=102) | 37 (36%) |
| ... of those, the answers naming a correct function anyway (n=31) | 26 (84%) |
| Unanswerable (n=41) | 10 (24%) |

No answer that abstains is warned. Of the 180 answerable questions whose answer does not
abstain, 172 now either point to a source or carry the warning; 8 have neither and get the
general "check it" hint.

**Findings.**

- The warning means "not from the sources", not "wrong". In 33 of the 69 warned answers to
  answerable questions, a flagged name is a correct label: the model was right from memory.
- The warning surfaced three validation answers (of 308) that answered the worked example's
  question instead of the real one: "To read an Excel file, use `read_excel`".

**Decision:** adopted. `/answer` returns `names_outside_sources`, and the page names those
functions and asks the reader to check them.

**Limits.**

- Only the 689 names in the corpus are known, so an invented function such as
  `.remove_time()` is not flagged.
- A function used only inside example code is not flagged.
- Flags were classified by one unblinded reading, as in experiment 17.
