# Annotations

Hand-made labels. Unlike everything under `data/`, these cannot be regenerated
by a script, so they are tracked in version control.

## gold_queries.csv

100 Stack Overflow questions tagged `pandas`, drawn at random (seed 7) from the
questions with an accepted answer by `scripts/make_gold_sheet.py`. Each row was
labelled by reading the question and its accepted answer.

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

## Attribution

Question titles and links come from Stack Overflow and are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each row links
to its source question.
