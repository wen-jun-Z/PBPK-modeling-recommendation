from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest

from pbpk_kb.recommendation import embedding_benchmark
from pbpk_kb.recommendation.embedding_benchmark import (
    MedCPTEncoder,
    SentenceTransformerEncoder,
)
from pbpk_kb.recommendation.reranker_benchmark import Qwen3Reranker
from pbpk_kb.recommendation.online_agent import _single_online_generation, online_generation_status


def test_online_generation_coalesces_identical_requests():
    entered, release = Event(), Event()
    calls = []

    @_single_online_generation
    def work():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {"result": ["completed"]}

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(work)
        try:
            assert entered.wait(5)
            duplicate = pool.submit(work)
            deadline = time.monotonic() + 3
            while online_generation_status()['shared_waiters'] != 1 and time.monotonic() < deadline:
                time.sleep(.01)
            assert online_generation_status()['shared_waiters'] == 1
        finally:
            release.set()
        first, second = pending.result(timeout=5), duplicate.result(timeout=5)
    first['result'].append('caller mutation')
    assert second == work() == {"result": ["completed"]}
    assert calls == [1]


def test_distinct_online_requests_wait_without_overlapping():
    entered, release, second_entered = Event(), Event(), Event()

    @_single_online_generation
    def work(name):
        if name == 'first':
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return name

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(work, 'first')
        try:
            assert entered.wait(5)
            second = pool.submit(work, 'second')
            deadline = time.monotonic() + 3
            while online_generation_status()['queued'] != 1 and time.monotonic() < deadline:
                time.sleep(.01)
            assert online_generation_status()['queued'] == 1
            assert not second_entered.is_set()
        finally:
            release.set()
        assert first.result(timeout=5) == 'first'
        assert second.result(timeout=5) == 'second'
    assert online_generation_status() == {'running':0,'queued':0,'shared_waiters':0}


def test_online_generation_failure_releases_slot():
    @_single_online_generation
    def failed():
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        failed()

    @_single_online_generation
    def next_request():
        return "available"

    assert next_request() == "available"


class _MovableModel:
    def __init__(self) -> None:
        self.destinations: list[str] = []

    def to(self, destination: str) -> _MovableModel:
        self.destinations.append(destination)
        return self


def test_sentence_transformer_release_moves_model_to_cpu() -> None:
    encoder = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    model = _MovableModel()
    encoder.model = model

    encoder.release()
    encoder.release()

    assert model.destinations == ["cpu"]
    assert encoder.model is None


def test_medcpt_release_moves_both_models_to_cpu() -> None:
    encoder = MedCPTEncoder.__new__(MedCPTEncoder)
    query_model = _MovableModel()
    article_model = _MovableModel()
    encoder.query_model = query_model
    encoder.article_model = article_model
    encoder.query_tokenizer = object()
    encoder.article_tokenizer = object()

    encoder.release()

    assert query_model.destinations == ["cpu"]
    assert article_model.destinations == ["cpu"]
    assert encoder.query_model is None
    assert encoder.article_model is None
    assert encoder.query_tokenizer is None
    assert encoder.article_tokenizer is None


def test_reranker_release_moves_model_to_cpu() -> None:
    reranker = Qwen3Reranker.__new__(Qwen3Reranker)
    model = _MovableModel()
    reranker.model = model
    reranker.tokenizer = object()

    reranker.release()

    assert model.destinations == ["cpu"]
    assert reranker.model is None
    assert reranker.tokenizer is None


def test_online_medcpt_factory_requests_query_only_encoder(monkeypatch) -> None:
    captured = {}

    class _FakeMedCPT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(embedding_benchmark, "MedCPTEncoder", _FakeMedCPT)

    result = embedding_benchmark._model_factory(
        "ncbi/MedCPT",
        cache_dir=Path("/tmp/cache"),
        device="cuda:0",
        batch_size=4,
        query_only=True,
    )

    assert isinstance(result, _FakeMedCPT)
    assert captured["query_only"] is True
