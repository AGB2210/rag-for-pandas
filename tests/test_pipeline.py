import numpy as np
import pytest
import sentence_transformers

from rag_for_pandas import local_generator, paths
from rag_for_pandas.jsonl import write_jsonl
from rag_for_pandas.pipeline import GENERATOR_ENV, RETRIEVER_ENV, load_pipeline


class FakeSentenceTransformer:
    loaded: list[str] = []

    def __init__(self, name):
        FakeSentenceTransformer.loaded.append(name)

    def encode(self, sentences, normalize_embeddings=True, batch_size=64):
        return np.ones((len(sentences), 2)) / np.sqrt(2)


class FakeLocalGenerator:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    """Replace the model classes load_pipeline imports, and point it at a two-document corpus."""
    corpus = tmp_path / "docstrings.jsonl"
    write_jsonl(corpus, [{"qualname": "DataFrame.dropna", "docstring": "Remove."}, {"qualname": "read_csv", "docstring": "Read."}])
    monkeypatch.setattr(paths, "CORPUS", corpus)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(local_generator, "LocalGenerator", FakeLocalGenerator)
    monkeypatch.delenv(RETRIEVER_ENV, raising=False)
    monkeypatch.delenv(GENERATOR_ENV, raising=False)
    FakeSentenceTransformer.loaded = []
    return monkeypatch


def test_defaults_load_the_trained_retriever_and_the_local_generator(fake_models):
    pipeline = load_pipeline()

    assert len(pipeline.docs) == 2
    assert FakeSentenceTransformer.loaded == [str(paths.HARD_NEGATIVE_MODEL)]
    assert pipeline.can_answer
    assert pipeline.generator.name == local_generator.LOCAL_MODEL


@pytest.mark.parametrize("disabled", ["none", "NONE"])
def test_environment_variables_choose_the_retriever_and_can_disable_answers(fake_models, disabled):
    fake_models.setenv(RETRIEVER_ENV, "some/other-retriever")
    fake_models.setenv(GENERATOR_ENV, disabled)

    pipeline = load_pipeline()

    assert FakeSentenceTransformer.loaded == ["some/other-retriever"]
    assert pipeline.generator is None
    assert not pipeline.can_answer
