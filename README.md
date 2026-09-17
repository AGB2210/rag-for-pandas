# rag-for-pandas

[![CI](https://github.com/AGB2210/rag-for-pandas/actions/workflows/ci.yml/badge.svg)](https://github.com/AGB2210/rag-for-pandas/actions/workflows/ci.yml)

Search over the pandas API documentation for the function that answers a question written
in everyday words, with short cited answers, trained and evaluated on real Stack Overflow
questions.

People rarely ask "how do I use `dropna`". They ask "how do I delete rows with empty
cells", and keyword search fails when the question and the documentation use different
words. This project measures that problem honestly and closes part of it by fine-tuning a
small embedding model on Stack Overflow question/answer pairs.

## Results

Gold test set: 300 hand-labelled Stack Overflow questions, 259 answerable from the corpus.
The table shows the 191 questions whose primary answer's name does **not** appear in the
question title, where keyword matching cannot help. R@5 is the share of questions with a correct
function in the top 5 results.

| Method | R@5, any correct function | R@5, the single best function |
|---|---|---|
| BM25 keyword search | 0.38 | 0.28 |
| `all-MiniLM-L6-v2` embeddings | 0.42 | 0.32 |
| Fine-tuned on Stack Overflow pairs | 0.49 | 0.34 |
| **Fine-tuned with mined hard negatives** | **0.62** | **0.43** |

Final model vs BM25: +0.25 (95% CI [+0.17, +0.32]) and +0.16 (95% CI [+0.08, +0.23]).
Paired bootstrap over questions; R@5 was fixed as the comparison metric in advance.
Retraining with the same settings after a small data change moved these scores by 3-4
points, so smaller differences between single training runs are not meaningful.

Numbers are produced by `scripts/evaluate_retrieval.py`. Every experiment, including those
that did not work (such as cross-encoder reranking), is recorded in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Cited answers

A local `Qwen2.5-1.5B-Instruct` writes a short answer from the top 3 documents and must cite
them as `[1]`-`[3]` or say it could not find the answer. On the 259 answerable gold questions:

| | Result |
|---|---|
| Retriever supplied a correct document in the top 3 | 61% |
| Answer cites a correct document | 13% (21% when one was retrieved) |
| Answer names a correct function, cited or not | 51% (64% when one was retrieved) |
| Answer cites an excerpt that does not exist | 0% |
| Answer says it could not find the answer, on the 41 unanswerable questions | 46% |

Answer generation is the weakest stage: the small local model often names the right function
without citing its source, and sometimes adds details that are not in the documentation.
The "names a correct function" measure was added after reading answers; see experiment 12.
Prompts that made the model cite more often also made it refuse more often, including when
the answer was retrieved, so they were not adopted; see experiment 14.
Numbers are produced by `scripts/evaluate_generation.py`.

## How it works

```mermaid
flowchart LR
    A["pandas source<br/>(commit a183ef5)"] -->|parse with ast| B["Corpus<br/>1,262 API docstrings"]
    C["Stack Overflow API<br/>2,500 questions"] --> D["Silver labels<br/>1,806 questions"]
    C --> E["Gold labels<br/>300 questions, by hand"]
    D -->|remove gold, split| F["Train 1,234<br/>Validation 308"]
    F -->|mine hard negatives| G["Fine-tuned retriever"]
    B --> H["Search"]
    G --> H
    H --> I["Evaluation<br/>R@k with confidence intervals"]
    E --> I
```

1. **Corpus.** Public docstrings of at least 20 words, extracted from pandas source with
   `ast` (pandas is never imported). Only names reachable from `import pandas` are kept.
2. **Silver labels.** The pandas functions used in each question's accepted answer. Cheap
   but loose: a manual review of 50 labels from the first version of the labeller found 34%
   correct, 50% loose, 16% wrong. The labeller was then fixed but not reviewed again. Used
   only to train and to choose settings.
3. **Gold labels.** 300 questions labelled by hand with the best function and other correct
   ones. Used only for the final score. See [annotations/README.md](annotations/README.md).
4. **Leakage protection.** Gold questions are removed from training data by id and by
   normalised title; the split raises an error if any gold question gets through.
5. **Training.** MultipleNegativesRankingLoss with one mined hard negative per pair: the
   document the base model ranks highest, below its top 3, that does not carry a labelled
   name. The top 3 are skipped because loose labels miss correct answers that rank there.
6. **Answer generation.** The top 3 documents, numbered, go to a local language model with
   rules to cite them and a fixed sentence for when they do not answer the question. The
   prompt was chosen by comparing prompts on validation questions.

## Setup

Developed on Windows 11 with Python 3.12.10 and an NVIDIA RTX 3050 (4 GB); the results
above come from that machine. Installation from scratch on other systems has not been
tested. Training also runs on CPU, more slowly.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest -q
```

On Linux or macOS use `.venv/bin/python`. `requirements.txt` installs CUDA 12.8 builds of
PyTorch; removing its `--extra-index-url` line installs the CPU build. It pins the packages
the code imports directly; the packages those depend on are not pinned.

Continuous integration (`.github/workflows/ci.yml`) runs on every push and pull request: the
Python tests with a CPU build of PyTorch, the web page's component tests and build, and a build
of the Docker image. None of these need trained models, data or a GPU; the API tests use fake
models. The browser tests do need the trained retriever, so they are run locally (see
[Web page](#web-page)).

## Reproducing the pipeline

Get the exact pandas source the corpus was built from:

```bash
git init data/raw/pandas
git -C data/raw/pandas fetch --depth 1 https://github.com/pandas-dev/pandas.git a183ef5779ecce1a5f3d6b766e130cf860afdfaf
git -C data/raw/pandas checkout FETCH_HEAD
```

Then run the steps in order:

```bash
.venv/Scripts/python scripts/build_corpus.py
.venv/Scripts/python scripts/fetch_stackoverflow.py
.venv/Scripts/python scripts/build_eval_set.py
.venv/Scripts/python scripts/split_eval_data.py
.venv/Scripts/python scripts/train_retriever.py --seed 3
.venv/Scripts/python scripts/train_retriever.py --hard-negatives --seed 3 --output-dir models/retriever-minilm-ft-hn
.venv/Scripts/python scripts/evaluate_retrieval.py
.venv/Scripts/python scripts/evaluate_generation.py
```

`evaluate_generation.py` downloads `Qwen/Qwen2.5-1.5B-Instruct` (2.9 GB) on first use.
Add `--from-saved` to print the summary again from the saved answers without regenerating.

`fetch_stackoverflow.py` uses the anonymous Stack Exchange API (about 50 requests, within
the 300-per-day limit) and caches every response. The results above use questions fetched
on 2026-09-13. Those responses are not redistributed, and vote order changes over time, so
a fresh fetch gives a different silver set; the gold question ids stay fixed.

## Running the API

After the pipeline has been run (the service needs the corpus and the trained retriever):

```bash
.venv/Scripts/python -m uvicorn rag_for_pandas.api:app --port 8000
```

Interactive documentation is at `http://127.0.0.1:8000/docs`. Models load once at startup.

| Endpoint | Request | Returns |
|---|---|---|
| `GET /health` | | status, number of documents, whether answers are enabled |
| `POST /search` | `{"query": "...", "k": 5}` | top k documents with score, excerpt and source file |
| `POST /answer` | `{"question": "..."}` | an answer, whether it abstained, and the 3 sources it could cite |

Queries must be 1-500 characters after trimming whitespace, and `k` must be 1-20; other
requests get HTTP 422. Only one answer is generated at a time; searches do not wait for it.

```bash
curl -X POST http://127.0.0.1:8000/answer -H "Content-Type: application/json" -d '{"question": "How do I delete rows that contain missing values?"}'
```

A response from the service on the development machine:

```json
{
  "question": "How do I delete rows that contain missing values?",
  "answer": "To remove rows containing missing values, use `DataFrame.dropna` with the parameter `how='any'`. This will remove any row where at least one value is missing.",
  "abstained": false,
  "sources": [
    {"number": 1, "name": "DataFrame.dropna", "cited": false},
    {"number": 2, "name": "Categorical.isnull", "cited": false},
    {"number": 3, "name": "Categorical.isna", "cited": false}
  ],
  "invalid_citations": []
}
```

The answer names the right function but cites nothing, which is typical of the small
generator (see the cited-answers results above).

Environment variables:

- `RAG_FOR_PANDAS_RETRIEVER`: retriever model folder or name (default `models/retriever-minilm-ft-hn`)
- `RAG_FOR_PANDAS_GENERATOR`: generator model name (default `Qwen/Qwen2.5-1.5B-Instruct`), or `none`
  to serve search only; `/answer` then returns HTTP 503

`scripts/smoke_test_api.py` starts the real server, checks every endpoint and stops it.

## Web page

A React page (Vite, TypeScript) in `frontend/` asks questions or searches the documentation.
It shows each answer with its sources, marks which sources were cited, and warns when an answer
cites nothing, cites a source that does not exist, or when the documentation did not answer the
question. Search results link to the pandas source file at the pinned commit.

Requires Node.js 24 (developed with 24.16.0):

```bash
cd frontend
npm ci
npm test
npm run build
```

`npm test` runs component tests against a fake API. `npm run build` type-checks the code and
writes the page to `frontend/dist`; when that folder exists, the API server above also serves the
page at `http://127.0.0.1:8000/`, from the same origin as the API.

Browser tests drive the whole system with Playwright: they build the page, start the real API
server with the trained retriever, and use the page in the Chrome installed on the machine. They
need the pipeline outputs, so they are run separately from `npm test`:

```bash
npm run e2e
```

Answers are turned off by default so the tests do not load the generator; set `E2E_ANSWERS=1` to
also check a real answer and its sources.

## Running with Docker

The `Dockerfile` builds the web page with Node, then runs the API and the page in a slim Python
image with a CPU build of PyTorch (2.3 GB). The corpus and the trained retriever are not in the
image: run the pipeline first, then mount them read-only when starting the container.

```bash
docker build -t rag-for-pandas .
```

In PowerShell on Windows:

```powershell
docker run -p 8000:8000 -v "${PWD}/models:/app/models:ro" -v "${PWD}/data/interim:/app/data/interim:ro" rag-for-pandas
```

On Linux or macOS the same command uses `$(pwd)` in place of `${PWD}`. The page is then at
`http://127.0.0.1:8000/`, and Docker reports the container healthy once `/health` answers.

Answers are off by default, because on CPU the generator is slow and memory-hungry. To turn them
on, set the generator and mount a Hugging Face cache that already holds it:

```powershell
docker run -p 8000:8000 -e RAG_FOR_PANDAS_GENERATOR=Qwen/Qwen2.5-1.5B-Instruct -e HF_HUB_OFFLINE=1 -v "${PWD}/models:/app/models:ro" -v "${PWD}/data/interim:/app/data/interim:ro" -v "${HOME}/.cache/huggingface:/app/.cache/huggingface:ro" rag-for-pandas
```

Measured on the development machine (Windows 11, Docker Desktop 29.8 with 7.4 GB of memory):

| | Search only | With answers |
|---|---|---|
| Ready after starting | 33 s | 49 s |
| Memory | 0.6 GB | 6.1 GB |
| Time per answer | | 24 s (2-5 s on the GPU without Docker) |

Search results in the container matched the GPU setup to 4 decimal places. Answers can be worded
differently: the GPU runs the generator in bfloat16 and the CPU in float32. The commands above were
tested in PowerShell; the Linux and macOS form was not run. The container runs as a non-root user
and cannot write to the mounted folders.

## Repository layout

```
src/rag_for_pandas/  pipeline logic, one module per step, and the API
scripts/             command-line entry points for each step
tests/               Python unit and API tests
frontend/            React web page, component tests and browser tests
annotations/         hand-made gold labels (tracked; everything under data/ is generated)
docs/                experiment log
.github/workflows/   continuous integration
Dockerfile           container image for the API and the page
```

## Limitations

- Gold labels were made by a single annotator.
- Gold was scored after several stages of the project, and those scores guided what to try
  next (for example, hard negatives). No setting was chosen by its gold score; seeds, prompts
  and reranking were decided on validation data.
- The corpus is API docstrings only, from an unreleased development snapshot of pandas
  (commit `a183ef5`, September 2026, after release 3.0.5). The user guide is not included, and
  2 gold questions are unanswerable because pandas has no docstring for `GroupBy.ngroups`.
- Silver labels are loose, which limits what training can learn; a cross-encoder reranker
  trained on them did not beat the retriever.
- The retriever is small (22M parameters) and was trained on 1,663 question-document pairs
  from 984 questions (training questions with more than 3 labels are skipped).
- Each configuration was trained once (seed 3). Retraining after a small data change moved gold
  R@5 by 3-4 points, so single-run differences of that size are not meaningful. Mining hard
  negatives with the fine-tuned model instead of the base model was not tried.
- The answer generator is a 1.5B-parameter model chosen to fit a 4 GB GPU. Its answers are
  checked only for citations and refusals, not for whether every statement is true, and
  they often name functions without citing them or add details not in the documentation.

## Data and licences

- The code in this repository is released under the [MIT License](LICENSE). The data below
  keeps its own licences.
- pandas documentation text is from the pandas source code, BSD 3-Clause License.
- Stack Overflow question titles in `annotations/` are licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each row links to its question.
