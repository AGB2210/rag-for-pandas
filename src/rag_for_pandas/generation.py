"""Answer generation: turn retrieved documentation into a short answer with citations.

The generator is an interface, so a local model and an API model are
interchangeable. Answers must cite the excerpts they use as [1], [2], ... and
must say so when the excerpts do not answer the question. Both behaviours can
be checked automatically, without a second model judging the answer.

The prompt was chosen with scripts/compare_prompts.py on 40 validation
questions: putting the rules after the question and showing two worked
examples raised the share of answers with a citation from 2/40 to 12/40.
Prompts that forced citations harder (experiment 14 in docs/EXPERIMENTS.md)
cited more but refused more often, even with the answer in an excerpt, so this
prompt was kept.

Most answers still name a function without citing its excerpt. Those excerpts
are linked afterwards by code (`named_indices`), which leaves the answer's
text untouched.
"""

from __future__ import annotations

import builtins
import re
from typing import Protocol

from rag_for_pandas.corpus import final_name
from rag_for_pandas.labels import FOREIGN_ROOTS

# The local model the prompt below was chosen with. Defined here rather than in
# local_generator.py so it can be read without loading PyTorch.
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_DOCS = 3
# The median docstring is 190 words, so about half the excerpts are complete; longer ones are cut.
MAX_EXCERPT_WORDS = 200
NOT_FOUND = "I could not find this in the pandas documentation."

SYSTEM_PROMPT = "You are a pandas documentation assistant."
# Placed after the question, closest to where the model starts writing: a
# small model follows those instructions more reliably than ones read earlier.
RULES = (
    "Rules:\n"
    "- Use only the excerpts above. Do not rely on anything else you know.\n"
    "- Answer in at most 4 sentences and name the pandas function to use.\n"
    "- End every sentence that uses an excerpt with its number in square brackets, like [2].\n"
    "- Do not invent example output.\n"
    f'- If no excerpt answers the question, reply exactly: "{NOT_FOUND}"'
)

# Worked examples shown before the real question (few-shot prompting). The
# second one teaches abstention: its excerpts do not answer it.
EXAMPLE_EXCERPTS = (
    "[1] DataFrame.dropna: Remove missing values.\n\n"
    "[2] DataFrame.fillna: Fill NA/NaN values using the specified method."
)
EXAMPLES = [
    ("How do I delete rows with empty cells?", "Use DataFrame.dropna, which removes rows that contain missing values [1]."),
    ("How do I read an Excel file?", NOT_FOUND),
]

CITATION = re.compile(r"\[(\d+)\]")
NON_SPACE = re.compile(r"\S+")


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


def excerpt(doc: dict, max_words: int = MAX_EXCERPT_WORDS) -> str:
    """The document's name and docstring, cut after `max_words` words with line breaks kept."""
    text = doc["docstring"]
    words = list(NON_SPACE.finditer(text))
    if len(words) > max_words:
        text = text[: words[max_words - 1].end()] + " ..."
    return f"{doc['qualname']}: {text}"


def numbered_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[{number}] {excerpt(doc)}" for number, doc in enumerate(docs, start=1))


def user_message(question: str, context: str, rules: str = RULES) -> str:
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\n{rules}"


def build_messages(
    question: str, docs: list[dict], rules: str = RULES, examples: list[tuple[str, str]] = EXAMPLES
) -> list[dict[str, str]]:
    """Chat messages: the worked examples, then the real question with its numbered excerpts."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, example_answer in examples:
        messages.append({"role": "user", "content": user_message(example_question, EXAMPLE_EXCERPTS, rules)})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": user_message(question, numbered_context(docs), rules)})
    return messages


def cited_indices(answer: str, doc_count: int) -> list[int]:
    """0-based indices of the excerpts an answer cites, in order of first citation."""
    cited: list[int] = []
    for match in CITATION.finditer(answer):
        index = int(match.group(1)) - 1
        if 0 <= index < doc_count and index not in cited:
            cited.append(index)
    return cited


def invalid_citations(answer: str, doc_count: int) -> list[int]:
    """Citation numbers that point at no excerpt, such as [4] when only 3 were given."""
    return [int(n) for n in CITATION.findall(answer) if not 1 <= int(n) <= doc_count]


def _names_as_code(answer: str, name: str) -> bool:
    """Whether `name` is written as a pandas name in code: after a dot, as a call, or in backticks."""
    escaped = re.escape(name)
    # `np.mean` is NumPy's function, not the pandas one.
    after_dot = any(m.group(1) not in FOREIGN_ROOTS for m in re.finditer(rf"(\w*)\.{escaped}(?!\w)", answer))
    # A bare `sum(x)` is Python's own function when Python has one by that name.
    as_call = not hasattr(builtins, name) and re.search(rf"(?<![\w.]){escaped}\(", answer) is not None
    # "the `mode` parameter" names an argument, not the function `mode`.
    in_backticks = re.search(rf"(?<!parameter )(?<!argument )`{escaped}(?:\(\))?`(?! parameter| argument)", answer) is not None
    return after_dot or as_call or in_backticks


def _names_in_full(answer: str, qualname: str) -> bool:
    """Whether a dotted name such as `DataFrame.dropna` appears whole."""
    return "." in qualname and re.search(rf"(?<![\w.]){re.escape(qualname)}(?!\w)", answer) is not None


def named_indices(answer: str, docs: list[dict]) -> list[int]:
    """0-based indices of the excerpts whose function the answer names, cited or not.

    The model often names the right function without citing its excerpt, so
    this link is made by code instead. A name counts only when written as code
    (`DataFrame.dropna`, `df.dropna`, `concat(` or in backticks): many pandas
    names are ordinary words, such as `all`, `count` and `merge`, so a bare
    word does not count. Names are compared by their last part, as everywhere
    else, so `df.dropna` links `Series.dropna` too; but when the answer writes
    one excerpt's name in full, only that excerpt is linked.

    The rule was shaped on validation questions and then checked on unseen
    ones (experiment 17 in docs/EXPERIMENTS.md).
    """
    in_full = [i for i, doc in enumerate(docs) if _names_in_full(answer, doc["qualname"])]
    full_names = {final_name(docs[i]["qualname"]) for i in in_full}
    return [
        i
        for i, doc in enumerate(docs)
        if i in in_full or (final_name(doc["qualname"]) not in full_names and _names_as_code(answer, final_name(doc["qualname"])))
    ]


def is_abstention(answer: str) -> bool:
    """Whether the answer says the documentation does not answer the question."""
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def cites_correct_document(answer: str, docs: list[dict], labels: list[str]) -> bool:
    """Whether any cited excerpt is a document carrying a correct name."""
    return any(final_name(docs[i]["qualname"]) in labels for i in cited_indices(answer, len(docs)))


def pointed_indices(answer: str, docs: list[dict]) -> list[int]:
    """0-based indices of the excerpts a reader is sent to: those the answer cites or names."""
    return sorted({*cited_indices(answer, len(docs)), *named_indices(answer, docs)})


def points_to_correct_document(answer: str, docs: list[dict], labels: list[str]) -> bool:
    """Whether any excerpt the answer cites or names is a document carrying a correct name."""
    return any(final_name(docs[i]["qualname"]) in labels for i in pointed_indices(answer, docs))


def names_correct_function(answer: str, labels: list[str]) -> bool:
    """Whether the answer text names a correct function as a whole word, cited or not.

    `dropna` matches "DataFrame.dropna" and "`dropna()`" but not "dropna_all".
    """
    return any(re.search(rf"(?<!\w){re.escape(label)}(?!\w)", answer) for label in labels)
