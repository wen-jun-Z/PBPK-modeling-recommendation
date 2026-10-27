import json
from hashlib import sha256
from pathlib import Path

import pytest

from pbpk_kb.acquisition import source_recovery as recovery
from pbpk_kb.acquisition.comprehensive_raw_retriever import url_with_params
from pbpk_kb.acquisition.precision import classify_openfda_label


@pytest.fixture
def recovered_case(tmp_path, monkeypatch):
    raw = tmp_path / "run"
    raw.mkdir()
    source, compound = "openfda_drug_label", "sildenafil"
    assets, selections = [], []
    for skip in [0, 100]:
        key = f"label_skip_{skip}"
        url = url_with_params("https://api.fda.gov/drug/label.json", {
            "search": f'openfda.generic_name:"{compound}"', "limit": 100, "skip": skip})
        request_id = source + ":" + key + ":" + sha256(f"{source}|{compound}|{key}|{url}".encode()).hexdigest()[:16]
        labels = [{"id": f"label-{i}", "set_id": f"set-{i}",
                   "openfda": {"generic_name": [compound]}, "effective_time": "20260101"}
                  for i in range(skip, min(skip + 100, 102))]
        payload = {"meta": {"results": {"total": 102, "skip": skip, "limit": 100}}, "results": labels}
        path = raw / (key + ".json")
        path.write_text(json.dumps(payload))
        asset = {"asset_id": key, "request_id": request_id, "source_id": source, "compound": compound,
                 "request_key": key, "url": url, "status_code": 200, "storage_path": str(path),
                 "checksum": sha256(path.read_bytes()).hexdigest(), "byte_count": path.stat().st_size}
        assets.append(asset)
        for label in labels:
            selections.append({"compound": compound, "source_id": source, "source_record_id": label["id"],
                "request_id": request_id, "source_url": url, "source_location": "label_id:" + label["id"],
                "record": {"label_id": label["id"], "set_id": label["set_id"],
                           "openfda": label["openfda"], "effective_time": label["effective_time"]},
                **{key: value for key, value in classify_openfda_label(compound, label).items()
                   if key in {"decision", "selection_reason", "identity_match_type"}}})
    selection_path = raw / "compounds/sildenafil/selection/openfda_drug_label/candidate_records.jsonl"
    recovery.atomic_rows(selection_path, selections)
    recovery.atomic_rows(raw / "assets.jsonl", assets)
    failure = {key: assets[0][key] for key in ("request_id", "source_id", "compound", "request_key", "url")}
    failure.update(status="failed", error="HTTP Error 500: Internal Server Error")
    recovery.atomic_rows(raw / "failures.jsonl", [failure])
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", lambda *a, **k: pytest.fail("Unexpected network request"))
    return raw, failure, assets, selection_path


def verify(case, tmp_path):
    raw, failure, assets, _ = case
    return recovery.verified_openfda_download_recovery(raw, failure, assets, root=tmp_path)


def test_complete_recovery_is_idempotent_and_preserves_original_failure(recovered_case, tmp_path):
    raw, _, _, _ = recovered_case
    original = (raw / "failures.jsonl").read_bytes()
    proof = verify(recovered_case, tmp_path)
    assert proof["status"] == "resolved" and proof["page_count"] == 2
    assert proof["label_count"] == proof["reported_total"] == 102
    assert proof["raw_lookup_recovered"] and not proof["scientific_evidence_validated"]
    assert verify(recovered_case, tmp_path) == proof
    assert len(recovery.rows(raw / "finalization/failure_resolutions.jsonl")) == 1
    assert (raw / "failures.jsonl").read_bytes() == original


def test_first_page_alone_cannot_resolve_failure(recovered_case, tmp_path):
    recovered_case[2].pop()
    assert verify(recovered_case, tmp_path) is None
    assert not (recovered_case[0] / "finalization/failure_resolutions.jsonl").exists()


@pytest.mark.parametrize("field,value", [
    ("request_id", "unrelated-request"), ("source_id", "pubmed"), ("compound", "caffeine"),
    ("request_key", "label_skip_99"), ("url", "https://unrelated.example/drug/label.json"),
])
def test_failure_identity_must_match(recovered_case, tmp_path, field, value):
    recovered_case[1][field] = value
    assert verify(recovered_case, tmp_path) is None


@pytest.mark.parametrize("change", ["missing_row", "wrong_decision", "wrong_request", "wrong_record"])
def test_every_label_requires_matching_selection(recovered_case, tmp_path, change):
    path = recovered_case[3]
    items = recovery.rows(path)
    if change == "missing_row":
        items.pop()
    elif change == "wrong_decision":
        items[-1]["decision"] = "excluded"
    elif change == "wrong_request":
        items[-1]["request_id"] = "unrelated-request"
    else:
        items[-1]["record"]["openfda"] = {"generic_name": ["caffeine"]}
    recovery.atomic_rows(path, items)
    assert verify(recovered_case, tmp_path) is None


@pytest.mark.parametrize("change", ["truncated", "total_changed", "duplicate_id", "wrong_skip"])
def test_inconsistent_pagination_cannot_resolve_failure(recovered_case, tmp_path, change):
    asset = recovered_case[2][-1]
    path = Path(asset["storage_path"])
    payload = json.loads(path.read_text())
    if change == "truncated":
        payload["results"].pop()
    elif change == "total_changed":
        payload["meta"]["results"]["total"] = 103
    elif change == "duplicate_id":
        payload["results"][0]["id"] = "label-0"
    else:
        payload["meta"]["results"]["skip"] = 0
    path.write_text(json.dumps(payload))
    asset.update(checksum=sha256(path.read_bytes()).hexdigest(), byte_count=path.stat().st_size)
    assert verify(recovered_case, tmp_path) is None


@pytest.mark.parametrize("change", ["checksum", "bytes"])
def test_asset_integrity_corruption_blocks_recovery(recovered_case, tmp_path, change):
    asset = recovered_case[2][-1]
    if change == "checksum":
        asset["checksum"] = "0" * 64
    else:
        asset["byte_count"] += 1
    with pytest.raises(RuntimeError, match="integrity"):
        verify(recovered_case, tmp_path)


def test_asset_path_cannot_escape_run(recovered_case, tmp_path):
    recovered_case[2][-1]["storage_path"] = str(tmp_path / "outside-run.json")
    with pytest.raises(ValueError, match="escaped"):
        verify(recovered_case, tmp_path)


def test_nonfirst_page_failure_also_requires_all_pages(recovered_case, tmp_path):
    failure, asset = recovered_case[1], recovered_case[2][-1]
    for field in ["request_id", "source_id", "compound", "request_key", "url"]:
        failure[field] = asset[field]
    assert verify(recovered_case, tmp_path)["page_count"] == 2


def test_500_asset_is_not_a_successful_recovery(recovered_case, tmp_path):
    recovered_case[2][0]["status_code"] = 500
    assert verify(recovered_case, tmp_path) is None
