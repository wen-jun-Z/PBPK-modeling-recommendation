import json
from hashlib import sha256
from pathlib import Path

from pbpk_kb.acquisition import source_recovery as recovery
from pbpk_kb.acquisition.reconcile_final_acquisition import completed_run_osp_snapshot


def test_recovery_preserves_invalid_index_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "PROJECT_ROOT", tmp_path)
    cache = tmp_path / "artifacts/pbpk-model-assets/coverage_expansion_20260907/Example_Caffeine"
    source = cache / "source"
    source.mkdir(parents=True)
    (source / "model.pksim5").write_bytes(b"cached-model-test")
    (cache / "source.tar.gz").write_bytes(b"pinned-test-archive")
    manifest = {"compound": "caffeine", "repository": "Open-Systems-Pharmacology/Example_Caffeine",
        "commit": "a" * 40, "source_url": "https://github.com/Open-Systems-Pharmacology/Example_Caffeine/tree/" + "a" * 40,
        "source_directory": str(source.relative_to(tmp_path)),
        "source_archive_sha256": sha256((cache / "source.tar.gz").read_bytes()).hexdigest(),
        "files": [{"path": "model.pksim5", "bytes": (source / "model.pksim5").stat().st_size,
                   "sha256": sha256((source / "model.pksim5").read_bytes()).hexdigest()}]}
    (cache / "asset_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "raw"
    run = output / "compound_runs/run_test"
    run.mkdir(parents=True)
    old = run / "old_snapshot.json"
    old.write_text('{"changed": true}')
    asset = {"asset_id": "old", "source_id": "osp_pbpk_model_library",
             "storage_path": str(old), "checksum": "invalid-original-checksum", "group": "shared/snapshot_manifest"}
    original = json.dumps(asset) + "\n"
    (run / "assets.jsonl").write_text(original)
    result = recovery.repair_model_snapshot(run, output, ["caffeine"])
    assert result["asset_count"] == 1
    assert (run / "finalization/source_recovery/assets_before_repair.jsonl").read_text() == original
    exclusions = recovery.rows(run / "finalization/source_recovery/excluded_asset_index_rows.jsonl")
    assert exclusions[0]["original_asset"]["checksum"] == "invalid-original-checksum"
    current = (run / "assets.jsonl").read_bytes()
    recovery.repair_model_snapshot(run, output, ["caffeine"])
    assert (run / "assets.jsonl").read_bytes() == current
    assert completed_run_osp_snapshot(output, run.name).name == "verified"


def test_verified_openfda_empty_lookup_is_audited_and_not_refetched(tmp_path, monkeypatch):
    raw = tmp_path / "run"
    raw.mkdir()
    (raw / "assets.jsonl").write_text("")
    url = 'https://api.fda.gov/drug/label.json?search=openfda.generic_name%3A%22paracetamol%22&limit=100&skip=0'
    failure = {"request_id": "failed-search", "source_id": "openfda_drug_label", "compound": "paracetamol",
               "request_key": "label_skip_0", "url": url}
    calls = []
    body = b'{"error":{"code":"NOT_FOUND","message":"No matches found!"}}'
    def fetch(value):
        calls.append(value)
        return 404, body
    monkeypatch.setattr(recovery, "fetch_openfda_lookup_response", fetch)
    assets = []
    proof = recovery.verified_openfda_empty_lookup(raw, failure, assets, root=tmp_path)
    assert proof["accepted_for_observation_processing"] is False
    assert recovery.verified_openfda_empty_lookup(raw, failure, assets, root=tmp_path) == proof
    assert calls == [url]
    assert len(recovery.rows(raw / "assets.jsonl")) == 1
    assert len(recovery.rows(raw / "finalization/failure_resolutions.jsonl")) == 1
    assert assets[0]["processing_policy"] == "metadata_only"
    assert Path(assets[0]["storage_path"]).read_bytes() == body


def test_generic_http_404_and_other_pages_do_not_become_proven_absence(tmp_path, monkeypatch):
    import pytest
    raw = tmp_path / "run"
    raw.mkdir()
    url = 'https://api.fda.gov/drug/label.json?search=openfda.generic_name%3A%22paracetamol%22&limit=100&skip=0'
    failure = {"request_id": "failed-search", "source_id": "openfda_drug_label", "compound": "paracetamol",
               "request_key": "label_skip_0", "url": url}
    for status, body in [(404, b'<html>Not Found</html>'), (500, b'{"error":{"code":"NOT_FOUND","message":"No matches found!"}}'),
                         (200, b'{"results":[{"id":"one"}]}')]:
        monkeypatch.setattr(recovery, "fetch_openfda_lookup_response", lambda value: (status, body))
        assert recovery.verified_openfda_empty_lookup(raw, failure, [], root=tmp_path) is None
    monkeypatch.setattr(recovery, "fetch_openfda_lookup_response", lambda value: pytest.fail("Unexpected lookup"))
    for changed in [{"request_key": "label_skip_100"}, {"url": url.replace("skip=0", "skip=100")},
                    {"url": url.replace("api.fda.gov", "example.invalid")}, {"compound": "caffeine"}]:
        assert recovery.verified_openfda_empty_lookup(raw, {**failure, **changed}, [], root=tmp_path) is None
    assert not list(raw.iterdir())


def chembl_outage_case(tmp_path):
    raw = tmp_path / "run"
    raw.mkdir()
    assets = []
    for source, key, url, payload in [
        ("pubchem", "synonyms_by_name", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/quinidine/synonyms/JSON",
         {"InformationList": {"Information": [{"CID": 441074, "Synonym": ["QUINIDINE", "CHEMBL1294"]}]}}),
        ("pubchem", "properties_by_name", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/quinidine/property/InChIKey/JSON",
         {"PropertyTable": {"Properties": [{"CID": 441074, "InChIKey": "LOUPRKONTZGTKE-LHHVKLHASA-N"}]}}),
        ("rxnorm", "rxcui_by_name", "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=quinidine&search=2",
         {"idGroup": {"rxnormId": ["9068"]}}),
    ]:
        path = raw / (key + ".json")
        path.write_text(json.dumps(payload))
        assets.append({"asset_id": key, "source_id": source, "compound": "quinidine", "request_key": key,
                       "url": url, "storage_path": str(path), "status_code": 200,
                       "checksum": sha256(path.read_bytes()).hexdigest()})
    recovery.atomic_rows(raw / "assets.jsonl", assets)
    return raw, assets, {"request_id": "chembl-failure", "source_id": "chembl", "compound": "quinidine",
                         "request_key": "molecule_search", "status": "failed", "error": "HTTP Error 500: Internal Server Error",
                         "url": "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=quinidine&limit=25&offset=0"}


def test_chembl_outage_keeps_raw_error_excluded_and_does_not_repeat_lookup(tmp_path, monkeypatch):
    raw, assets, failure = chembl_outage_case(tmp_path)
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return 500, b"<html>Service error</html>"
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", fetch)
    original = (raw / "assets.jsonl").read_bytes()
    proof = recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path)
    assert proof["observations_recovered"] is False
    assert proof["retryable_in_future_run"] is True
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) == proof
    assert calls == [failure["url"]]
    assert (raw / "assets.jsonl").read_bytes().startswith(original)
    assert assets[-1]["processing_policy"] == "metadata_only"
    assert len(recovery.rows(raw / "finalization/failure_resolutions.jsonl")) == 1
    Path(assets[-1]["storage_path"]).write_bytes(b"changed")
    import pytest
    with pytest.raises(RuntimeError, match="integrity"):
        recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path)


def test_chembl_outage_requires_verified_alternative_identity_and_official_url(tmp_path, monkeypatch):
    import pytest
    raw, assets, failure = chembl_outage_case(tmp_path)
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", lambda url, **kwargs: pytest.fail("Unexpected network request"))
    for changed in [{"compound": "quinine"}, {"url": failure["url"].replace("www.ebi.ac.uk", "example.invalid")},
                    {"error": "HTTP Error 404: Not Found"}]:
        assert recovery.verified_chembl_service_gap(raw, {**failure, **changed}, assets, root=tmp_path) is None
    assert recovery.verified_chembl_service_gap(raw, failure, assets[:-1], root=tmp_path) is None
    p = Path(assets[0]["storage_path"])
    p.write_text(json.dumps({"InformationList": {"Information": [{"CID": 441074, "Synonym": ["quinine"]}]}}))
    assets[0]["checksum"] = sha256(p.read_bytes()).hexdigest()
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) is None
    p.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path)


def test_chembl_service_recheck_is_bounded_and_404_is_not_provider_outage(tmp_path, monkeypatch):
    raw, assets, failure = chembl_outage_case(tmp_path)
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return 404, b"Not found"
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", fetch)
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) is None
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) is None
    assert calls == [failure["url"]]
    assert not (raw / "finalization/failure_resolutions.jsonl").exists()


def test_chembl_successful_recheck_is_recovered_raw_and_keeps_identity_gate(tmp_path, monkeypatch):
    raw, assets, failure = chembl_outage_case(tmp_path)
    body = json.dumps({"molecules": [{"pref_name": "QUINIDINE", "molecule_chembl_id": "CHEMBL1294",
                       "molecule_structures": {"standard_inchi_key": "LOUPRKONTZGTKE-LHHVKLHASA-N"}}]}).encode()
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return 200, body
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", fetch)
    proof = recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path)
    assert proof["status"] == "resolved"
    assert proof["raw_lookup_recovered"] is True
    assert proof["scientific_evidence_validated"] is False
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) == proof
    assert calls == [failure["url"]]
    assert assets[-1]["group"] == "compound_api"
    assert assets[-1].get("processing_policy") != "metadata_only"
    assert len(recovery.rows(raw / "finalization/failure_resolutions.jsonl")) == 1
    from pbpk_kb.parsers.file_types import extension_for, file_family_for
    p = Path(assets[-1]["storage_path"])
    ext = extension_for(p, assets[-1]["content_type"], assets[-1]["url"])
    assert file_family_for(p, ext, assets[-1]["content_type"], "chembl") == "json"
    import pytest
    p.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="integrity"):
        recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path)


def test_chembl_success_with_wrong_stereochemistry_is_not_recovered(tmp_path, monkeypatch):
    raw, assets, failure = chembl_outage_case(tmp_path)
    body = json.dumps({"molecules": [{"pref_name": "QUINIDINE", "molecule_chembl_id": "CHEMBL1294",
                       "molecule_structures": {"standard_inchi_key": "different-stereoisomer"}}]}).encode()
    monkeypatch.setattr(recovery, "fetch_bounded_public_response", lambda url, **kwargs: (200, body))
    assert recovery.verified_chembl_service_gap(raw, failure, assets, root=tmp_path) is None
    assert len(recovery.rows(raw / "assets.jsonl")) == 3
    assert not (raw / "finalization/failure_resolutions.jsonl").exists()
