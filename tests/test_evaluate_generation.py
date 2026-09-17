from evaluate_generation import rate, summary_lines


def record(answerable=True, retrieved=True, cited=False, invalid=False, correct=False, names=False, abstained=False):
    return {
        "answerable": answerable,
        "context_has_correct": retrieved,
        "cited": ["DataFrame.dropna"] if cited else [],
        "invalid_citations": [4] if invalid else [],
        "cites_correct": correct,
        "names_correct": names,
        "abstained": abstained,
    }


RECORDS = [
    record(cited=True, correct=True, names=True),
    record(names=True),
    record(abstained=True),
    record(retrieved=False, names=True),
    record(retrieved=False, abstained=True, invalid=True),
    record(answerable=False, retrieved=False, abstained=True),
    record(answerable=False, retrieved=False, cited=True),
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
    assert value(lines, "answerable (5)", "answer cites a number with no excerpt") == "1/5 (20%)"
    assert value(lines, "answerable (5)", "answer names a correct function") == "3/5 (60%)"
    assert value(lines, "answerable, correct document retrieved (3)", "answer wrongly abstains") == "1/3 (33%)"
    assert value(lines, "answerable, no correct document retrieved (2)", "answer names a correct function anyway") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer abstains") == "1/2 (50%)"
    assert value(lines, "unanswerable (2)", "answer cites an excerpt anyway") == "1/2 (50%)"


def test_rate_handles_an_empty_group():
    assert rate(0, 0) == "0/0"
