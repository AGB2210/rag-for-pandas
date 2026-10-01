from evaluate_generation import grade, rate, summary_lines


def record(
    answerable=True, retrieved=True, cited=False, named=False, invalid=False, correct=False, points=False, names=False,
    outside=False, abstained=False,
):
    return {
        "answerable": answerable,
        "context_has_correct": retrieved,
        "cited": ["DataFrame.dropna"] if cited else [],
        "named": ["DataFrame.dropna"] if named else [],
        "invalid_citations": [4] if invalid else [],
        "cites_correct": correct,
        "points_to_correct": points,
        "names_correct": names,
        "outside": ["concat"] if outside else [],
        "abstained": abstained,
    }


RECORDS = [
    record(cited=True, correct=True, points=True, names=True),
    record(named=True, points=True, names=True),
    record(abstained=True),
    record(retrieved=False, names=True, outside=True),
    record(retrieved=False, abstained=True, invalid=True),
    record(answerable=False, retrieved=False, abstained=True),
    record(answerable=False, retrieved=False, cited=True, outside=True),
]


def value(lines, section, label):
    """The rate printed for `label` inside the section whose heading starts with `section`."""
    start = next(i for i, line in enumerate(lines) if line.startswith(section))
    end = next((i for i in range(start + 1, len(lines)) if lines[i] == ""), len(lines))
    return next(line.split(":")[-1].strip() for line in lines[start:end] if line.strip().startswith(label))


def test_summary_counts_each_group_separately():
    lines = summary_lines(RECORDS)

    assert lines[0] == "gold questions: 7"
    assert value(lines, "answerable (5)", "retriever put a correct document") == "3/5 (60%)"
    assert value(lines, "answerable (5)", "answer cites a correct document") == "1/5 (20%)"
    assert value(lines, "answerable (5)", "answer points to an excerpt") == "2/5 (40%)"
    assert value(lines, "answerable (5)", "answer points to a correct document") == "2/5 (40%)"
    assert value(lines, "answerable, correct document retrieved (3)", "answer points to a correct document") == "2/3 (67%)"
    assert value(lines, "answerable (5)", "answer cites a number with no excerpt") == "1/5 (20%)"
    assert value(lines, "answerable (5)", "answer names a correct function") == "3/5 (60%)"
    assert value(lines, "answerable, correct document retrieved (3)", "answer wrongly abstains") == "1/3 (33%)"
    assert value(lines, "answerable, no correct document retrieved (2)", "answer names a correct function anyway") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer abstains") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer cites an excerpt anyway") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer points to an excerpt anyway") == "1/2 (50%)"
    assert value(lines, "answerable (5)", "answer is warned") == "1/5 (20%)"
    assert value(lines, "answerable, correct document retrieved (3)", "answer is warned") == "0/3 (0%)"
    assert value(lines, "answerable, no correct document retrieved (2)", "... and is warned") == "1/1 (100%)"
    assert value(lines, "answerable, no correct document retrieved (2)", "answer is warned") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer is warned") == "1/2 (50%)"


def test_grade_adds_every_check_from_the_saved_answer():
    saved = {
        "answerable": True,
        "labels": ["dropna"],
        "context": ["DataFrame.fillna", "DataFrame.dropna", "read_csv"],
        "answer": "Use DataFrame.dropna to remove them [1], or `pd.concat()`. See also [7].",
    }

    graded = grade(saved, {"dropna", "fillna", "read_csv", "concat"})

    assert graded["answer"] == saved["answer"]
    assert graded["context_has_correct"] is True
    assert graded["cited"] == ["DataFrame.fillna"]
    assert graded["named"] == ["DataFrame.dropna"]
    assert graded["invalid_citations"] == [7]
    assert graded["cites_correct"] is False  # the citation points at fillna
    assert graded["points_to_correct"] is True  # the name points at dropna
    assert graded["names_correct"] is True
    assert graded["outside"] == ["concat"]
    assert graded["abstained"] is False


def test_rate_handles_an_empty_group():
    assert rate(0, 0) == "0/0"
