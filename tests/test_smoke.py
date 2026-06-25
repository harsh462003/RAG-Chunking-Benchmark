"""Basic smoke tests for package imports."""

from rag_chunking_benchmark import CHUNKING_METHODS, count_words


def test_chunking_methods_exist():
    assert len(CHUNKING_METHODS) == 8


def test_count_words():
    assert count_words("hello world") == 2
