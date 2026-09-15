from docsearch.generation import (
    NOT_FOUND,
    RULES,
    SYSTEM_PROMPT,
    build_messages,
    cited_indices,
    cites_correct_document,
    excerpt,
    invalid_citations,
    is_abstention,
)

DOCS = [
    {"qualname": "DataFrame.dropna", "docstring": "Remove missing values.\n\nParameters\n----------\naxis : int"},
    {"qualname": "DataFrame.fillna", "docstring": "Fill NA/NaN values."},
    {"qualname": "read_csv", "docstring": "Read a comma-separated values file."},
]


def test_excerpt_cuts_long_docstrings_but_keeps_line_breaks():
    doc = {"qualname": "f", "docstring": "one two\nthree four five"}
    assert excerpt(doc, max_words=3) == "f: one two\nthree ..."
    assert excerpt(doc, max_words=10) == "f: one two\nthree four five"


def test_build_messages_shows_examples_then_the_real_question():
    messages = build_messages("delete empty rows", DOCS)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "[1]" in messages[2]["content"]  # the first example answer is cited
    assert messages[4]["content"] == NOT_FOUND  # the second example teaches abstention


def test_real_question_has_numbered_excerpts_and_rules_last():
    question = build_messages("delete empty rows", DOCS)[-1]["content"]

    assert "[1] DataFrame.dropna: Remove missing values." in question
    assert "[3] read_csv:" in question
    assert "Question: delete empty rows" in question
    assert question.endswith(RULES)
    assert NOT_FOUND in RULES


def test_cited_indices_are_distinct_valid_and_in_order():
    answer = "Use dropna [1]. Not fillna [2], see again [1]; ignore [7] and [0]."
    assert cited_indices(answer, 3) == [0, 1]
    assert invalid_citations(answer, 3) == [7, 0]


def test_is_abstention_ignores_case_and_final_full_stop():
    assert is_abstention(NOT_FOUND)
    assert is_abstention("i could not find this in the pandas documentation")
    assert not is_abstention("Use dropna [1].")


def test_cites_correct_document_requires_a_cited_correct_name():
    assert cites_correct_document("Use dropna [1].", DOCS, ["dropna"])
    assert not cites_correct_document("Use fillna [2].", DOCS, ["dropna"])
    assert not cites_correct_document("Use dropna.", DOCS, ["dropna"])  # named but not cited
