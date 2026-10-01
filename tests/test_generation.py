from rag_for_pandas.generation import (
    NOT_FOUND,
    RULES,
    SYSTEM_PROMPT,
    build_messages,
    cited_indices,
    cites_correct_document,
    excerpt,
    invalid_citations,
    is_abstention,
    named_indices,
    names_correct_function,
    pointed_indices,
    points_to_correct_document,
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


def docs_named(*qualnames):
    return [{"qualname": qualname} for qualname in qualnames]


def test_named_indices_links_names_written_as_code():
    assert named_indices("Use `DataFrame.dropna()` here.", DOCS) == [0]
    assert named_indices("Call df.fillna(0).", DOCS) == [1]
    assert named_indices("Read it with read_csv('a.csv').", DOCS) == [2]
    assert named_indices("Use the `dropna` method, then `fillna()`.", DOCS) == [0, 1]
    assert named_indices("Use dropna_all.", DOCS) == []


def test_named_indices_ignores_ordinary_words():
    docs = docs_named("DataFrame.all", "DataFrame.count", "merge")
    assert named_indices("Count all rows, then merge the tables.", docs) == []
    assert named_indices("Use df.all() and `merge`.", docs) == [0, 2]


def test_named_indices_ignores_other_libraries_python_and_parameters():
    docs = docs_named("DataFrame.mean", "DataFrame.sum", "Series.mode")
    assert named_indices("Pass np.mean to it.", docs) == []
    assert named_indices("Add them with sum(values).", docs) == []  # Python's own sum
    assert named_indices("Set the `mode` parameter to 'a'.", docs) == []
    assert named_indices("Use the parameter `mode` here.", docs) == []
    assert named_indices("Use df.mean(), `sum()` and `mode`.", docs) == [0, 1, 2]


def test_named_indices_compares_last_names_unless_one_is_written_in_full():
    docs = docs_named("DataFrame.dropna", "Series.dropna", "Index.dropna")
    assert named_indices("Use df.dropna().", docs) == [0, 1, 2]
    assert named_indices("Use Series.dropna.", docs) == [1]
    assert named_indices("Use MultiIndex.dropna.", docs) == [0, 1, 2]  # no excerpt has that full name


def test_pointed_indices_join_cited_and_named_excerpts():
    assert pointed_indices("Use DataFrame.fillna. See [3].", DOCS) == [1, 2]
    assert pointed_indices("Nothing here.", DOCS) == []


def test_points_to_correct_document_accepts_a_named_or_cited_correct_name():
    assert points_to_correct_document("Use DataFrame.dropna.", DOCS, ["dropna"])  # named, not cited
    assert points_to_correct_document("Use it [1].", DOCS, ["dropna"])  # cited, not named
    assert not points_to_correct_document("Drop them with dropna.", DOCS, ["dropna"])  # a bare word is not a link
    assert not points_to_correct_document("Use DataFrame.fillna [2].", DOCS, ["dropna"])


def test_names_correct_function_matches_whole_names_only():
    assert names_correct_function("Use `DataFrame.dropna()` here.", ["fillna", "dropna"])
    assert not names_correct_function("Use dropna_all.", ["dropna"])
    assert not names_correct_function("Use fillna.", ["dropna"])
    assert not names_correct_function("Use dropna.", [])  # unanswerable questions have no labels


def test_build_messages_uses_the_rules_and_examples_it_is_given():
    messages = build_messages("q", DOCS, rules="Other rules", examples=[("example q", "example a")])

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"].endswith("Other rules")
    assert messages[-1]["content"].endswith("Other rules")
