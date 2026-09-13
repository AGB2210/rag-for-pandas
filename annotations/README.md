# Annotations

Hand-made labels. Unlike everything under `data/`, these cannot be regenerated
by a script, so they are tracked in version control.

## gold_queries.csv

300 Stack Overflow questions tagged `pandas`, drawn at random by
`scripts/make_gold_sheet.py` from questions with an accepted answer, in two
rounds that only ever append:

- rows 1-100: from the 500 most-voted questions (seed 7)
- rows 101-300: from the 2,500 most-voted questions, excluding rows 1-100 (seed 8)

Each row was labelled by reading the question and its accepted answer. Gold
questions are removed from training and validation data by
`scripts/split_eval_data.py`, so the model is never trained on them.

| Column | Meaning |
|---|---|
| `question_id`, `query`, `url` | The Stack Overflow question; `query` is its title |
| `primary` | The pandas name whose documentation best answers the question |
| `also_correct` | Other names that genuinely answer it, space separated |
| `answerable` | `no` when no single documented pandas name answers it: plain indexing syntax, another library, or a conceptual question |
| `notes` | Why a label was chosen, especially when it differs from the accepted answer |

Names are the final segment of a documented name (`DataFrame.dropna` is
`dropna`) and all exist in the corpus built by `scripts/measure_docstrings.py`.

Rules applied while labelling:

- Label what the question asks, not every function the answer happens to call.
- A clearly better documented function may be the primary label even when an
  older accepted answer predates it (for example `explode`); such rows say so
  in `notes`.
- Aliases that share one docstring (`agg` and `aggregate`) are listed together.
- "Answerable" means answerable from this corpus. When the right function
  exists in pandas but not in the corpus (for example `dt.year`, whose
  documentation the extractor does not read), the row is marked `no` and its
  note starts with `corpus gap:`. Those rows list what a corpus fix should add.

## Attribution

Question titles and links come from Stack Overflow and are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each row links
to its source question.
