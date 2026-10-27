import json
from urllib.parse import parse_qs, urlparse

from pbpk_kb.ui import api_client


class DummyResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_review_queue_omits_blank_query_params(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        return DummyResponse([])

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.review_queue(
        status="needs_review",
        evidence_type="",
        source_id="",
        compound="",
        q="",
        created_from="",
        created_to="",
        duplicate_risk="all",
        limit=50,
    )

    parsed = urlparse(captured["url"])
    params = parse_qs(parsed.query)

    assert parsed.path == "/evidence/review-queue"
    assert "created_from" not in params
    assert "created_to" not in params
    assert "compound" not in params
    assert params["status"] == ["needs_review"]
    assert params["duplicate_risk"] == ["all"]


def test_ask_question_posts_json_without_real_service(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"answer": "source-backed"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    result = api_client.ask_question(question="What evidence exists?", compound="")

    assert result == {"answer": "source-backed"}
    assert captured["url"] == "http://example.test/qa/ask"
    assert captured["method"] == "POST"
    assert captured["body"]["compound"] is None


def test_export_evidence_posts_expected_path(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"output_path": "exports/evidence.csv"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.export_evidence(
        compound="midazolam",
        curation_status="auto_validated",
        evidence_layer="auto_validated",
    )

    assert captured["url"] == "http://example.test/exports/evidence"
    assert captured["body"] == {
        "compound": "midazolam",
        "curation_status": "auto_validated",
        "evidence_layer": "auto_validated",
    }


def test_update_source_uses_patch_json_without_real_service(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"source_id": "hpa", "enabled": False})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.update_source(source_id="hpa", enabled=False, owner_curator="curator")

    assert captured["url"] == "http://example.test/sources/hpa"
    assert captured["method"] == "PATCH"
    assert captured["body"]["enabled"] is False
    assert captured["body"]["owner_curator"] == "curator"


def test_retry_ingestion_run_posts_expected_path(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"source_id": "pubchem"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.retry_ingestion_run(ingestion_run_id="run-1", query="midazolam", retmax=2)

    assert captured["url"] == "http://example.test/ingestion-runs/run-1/retry"
    assert captured["method"] == "POST"
    assert captured["body"]["query"] == "midazolam"
    assert captured["body"]["retmax"] == 2


def test_graph_client_uses_expected_paths(monkeypatch) -> None:
    captured = []

    def fake_urlopen(request, timeout=30):
        captured.append(request.full_url)
        return DummyResponse({"ok": True})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    assert api_client.graph_summary() == {"ok": True}
    assert api_client.compound_graph("compound-1") == {"ok": True}
    assert api_client.graph_centers("protein_target") == {"ok": True}
    assert api_client.target_graph("target-1") == {"ok": True}
    assert api_client.source_graph("pubmed") == {"ok": True}
    assert api_client.model_graph("model-1") == {"ok": True}

    assert captured == [
        "http://example.test/graph/summary",
        "http://example.test/graph/compound/compound-1",
        "http://example.test/graph/centers/protein_target",
        "http://example.test/graph/target/target-1",
        "http://example.test/graph/source/pubmed",
        "http://example.test/graph/model/model-1",
    ]


def test_validated_recommendation_client_uses_api_layer(monkeypatch) -> None:
    captured = []

    def fake_urlopen(request, timeout=30):
        captured.append(request.full_url)
        return DummyResponse([])

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.validated_recommendations(
        compound="midazolam",
        objective="base_pk",
        limit=25,
    )
    api_client.validated_recommendation_detail("result-1")

    first = urlparse(captured[0])
    assert first.path == "/recommendation/validated"
    assert parse_qs(first.query) == {
        "compound": ["midazolam"],
        "objective": ["base_pk"],
        "limit": ["25"],
    }
    assert captured[1] == (
        "http://example.test/recommendation/validated/result-1"
    )


def test_ingest_source_posts_query_params(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        return DummyResponse({"source_id": "pubchem"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    api_client.ingest_source(source_id="pubchem", query="warfarin", retmax=3, allow_network=True)

    parsed = urlparse(captured["url"])
    params = parse_qs(parsed.query)
    assert parsed.path == "/sources/pubchem/ingest"
    assert captured["method"] == "POST"
    assert params["query"] == ["warfarin"]
    assert params["retmax"] == ["3"]
    assert params["allow_network"] == ["True"]


def test_recommendation_request_posts_complete_payload(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["request"] = request
        captured["timeout"] = timeout
        return DummyResponse({"status": "recommendation_validated"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")
    payload = {
        "compound": "midazolam",
        "objective": "ddi_victim",
        "species": "human",
        "perpetrator_drug": "ketoconazole",
        "prediction_endpoints": ["DDI_AUC_ratio"],
    }

    api_client.recommend_pbpk_request(payload)

    assert captured["request"].full_url == "http://example.test/recommendation"
    assert captured["request"].get_method() == "POST"
    assert json.loads(captured["request"].data) == payload
    assert captured["timeout"] == 1800


def test_public_model_ranking_uses_dedicated_non_gpu_endpoint(
    monkeypatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return DummyResponse({"decision": "reuse"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    result = api_client.rank_public_model(
        {
            "compound": "Rosuvastatin",
            "objective": "public_model_reuse",
            "species": "human",
            "route": "oral",
        }
    )

    assert result == {"decision": "reuse"}
    assert captured["url"] == (
        "http://example.test/recommendation/public-model-ranking"
    )
    assert captured["method"] == "POST"
    assert captured["payload"]["compound"] == "Rosuvastatin"
    assert captured["timeout"] == 120


def test_public_model_catalog_summary_uses_dedicated_endpoint(
    monkeypatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return DummyResponse({"model_version_count": 49})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")

    result = api_client.public_model_catalog_summary()

    assert result == {"model_version_count": 49}
    assert captured["url"] == (
        "http://example.test/recommendation/public-model-catalog-summary"
    )
    assert captured["method"] == "GET"
    assert captured["timeout"] == 30


def test_blueprint_client_uses_registered_api_contract(monkeypatch) -> None:
    captured = []

    def fake_urlopen(request, timeout=30):
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": (
                    json.loads(request.data)
                    if request.data is not None
                    else None
                ),
                "timeout": timeout,
            }
        )
        return DummyResponse({"status": "ok"})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("API_BASE_URL", "http://example.test")
    payload = {
        "compound": "Caffeine",
        "objective": "public_model_reuse",
        "species": "human",
        "route": "oral",
    }

    api_client.create_blueprint(payload)
    api_client.blueprint_detail("blueprint-1")
    api_client.blueprint_components("blueprint-1", "process")
    api_client.validate_blueprint("blueprint-1")
    api_client.compile_blueprint("blueprint-1")
    api_client.run_blueprint("blueprint-1")

    assert captured[0] == {
        "url": "http://example.test/blueprints",
        "method": "POST",
        "body": payload,
        "timeout": 180,
    }
    assert captured[1]["url"] == (
        "http://example.test/blueprints/blueprint-1"
    )
    component_url = urlparse(captured[2]["url"])
    assert component_url.path == "/blueprints/blueprint-1/components"
    assert parse_qs(component_url.query) == {
        "component_kind": ["process"],
        "limit": ["2000"],
    }
    assert [row["method"] for row in captured[3:]] == [
        "POST",
        "POST",
        "POST",
    ]
    assert api_client.blueprint_package_url("blueprint-1") == (
        "http://example.test/blueprints/blueprint-1/package"
    )
