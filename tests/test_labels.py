import pytest

from docsearch.labels import build_silver_queries, chain_before, is_pandas_call, names_in_answer

OWNERS = {
    "astype": {"NDFrame"},
    "lower": {"StringMethods"},
    "format": {"StylerRenderer"},
    "sum": {"DataFrame"},
    "dropna": {"DataFrame"},
    "set_option": {""},
    "dtype": {"Series"},
    "DataFrame": {""},
}


def chain_for(code: str, name: str) -> list[str]:
    return chain_before(code, code.rindex("." + name))


@pytest.mark.parametrize(
    ("code", "name", "expected"),
    [
        ("np.datetime64(x).astype(datetime)", "astype", ["datetime64", "np"]),
        ("HTML('<style>{}</style>'.format(CSS))", "format", ["<str>"]),
        ('df["a"].str.lower()', "lower", ["str", "df"]),
        ('df.groupby(["a", "b"])["c"].sum()', "sum", ["groupby", "df"]),
        ('(df["a"] > 0).any()', "any", [""]),
    ],
)
def test_chain_before_walks_back_to_the_root(code, name, expected):
    assert chain_for(code, name) == expected


@pytest.mark.parametrize(
    ("name", "chain", "expected"),
    [
        ("astype", ["datetime64", "np"], False),  # NumPy method
        ("format", ["<str>"], False),  # Python string method
        ("lower", ["c"], False),  # string-only name without the .str accessor
        ("lower", ["str", "df"], True),
        ("sum", ["groupby", "df"], True),
        ("append", ["rows"], False),  # a Python list in practice
    ],
)
def test_is_pandas_call(name, chain, expected):
    assert is_pandas_call(name, chain, OWNERS) is expected


def test_names_in_answer_reads_code_spans_and_ignores_other_libraries():
    body = (
        "<pre><code>df = df.dropna()\nnp.dtype(&quot;x&quot;)\n</code></pre>"
        "<p>Then call <code>pd.set_option</code> for display.</p>"
    )
    assert names_in_answer(body, OWNERS) == {"dropna", "set_option"}


def test_build_silver_queries_labels_deduplicates_and_skips():
    questions = [
        {"question_id": 1, "title": "How to use dropna", "accepted_answer_id": 10, "link": "u1"},
        {"question_id": 1, "title": "How to use dropna", "accepted_answer_id": 10, "link": "u1"},
        {"question_id": 2, "title": "Make a frame", "accepted_answer_id": 20, "link": "u2"},
        {"question_id": 3, "title": "No accepted answer", "link": "u3"},
    ]
    answers = {10: "<code>df.dropna()</code>", 20: "<code>pd.DataFrame(x)</code>"}

    silver = build_silver_queries(questions, answers, OWNERS)

    assert silver.candidates == 2
    assert silver.skipped_no_label == 1  # only the generic name DataFrame
    assert silver.records == [
        {"question_id": 1, "query": "How to use dropna", "labels": ["dropna"], "label_in_query": True, "url": "u1"}
    ]
