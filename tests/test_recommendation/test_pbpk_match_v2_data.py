"""Leakage, condition visibility, label masks and immutable round-trip boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match_v2 import data as d


def query(i, **extra):
    return {"query_id": f"q{i}", "compounds": [f"drug{i}"], "task_type": "plasma_concentration_time_prediction",
            "text": f"Find PBPK for drug{i}.\nTarget endpoints: Cmax", "features": {"species": ["human"]},
            "variants": {"reordered": {"text": f"Cmax; drug{i}; PBPK"},
                         "missing_evidence": {"text": f"PBPK drug{i}", "features": {}}}, **extra}


def candidate(i):
    return {"candidate_id": f"c{i}", "family_id": "shared-family", "compounds": [f"drug{i}"],
            "text": "Public model catalogue only", "features": {"species": ["human"], "formulations": ["tablet"]}}


def judgment(q, c, grade=3, app=True, status="judged"):
    return {"query_id": q, "candidate_id": c, "judge": "primary", "protocol_sha256": "p",
            "status": status, "grade": grade, "documented_applicability": app,
            "support": [{"quote": "HIDDEN TARGET EVIDENCE"}]}


@pytest.fixture
def assembled():
    queries, candidates = [query(i) for i in range(10)], [candidate(i) for i in range(2)]
    core, clinical = queries[:8], queries[8:]
    scenarios = [{"scenario_id": q["query_id"], "compound": q["compounds"][0], "study_id": f"study-{i}",
                  "publication_cluster_id": f"study-{i}", "normalized_design": {"dose": {"value": 50, "unit": "mg"}},
                  "sealed_references": [{"path": "DO NOT OPEN"}]} for i, q in enumerate(clinical)]
    labels = [judgment(q["query_id"], c["candidate_id"]) for q in core for c in candidates]
    scalar = {"state": "valid", "endpoint": "Cmax", "endpoint_key": "Cmax_reference_vs_group", "predicted": 2.,
              "observed": 1., "absolute_log_error": float(np.log(2)), "signed_log_error": float(np.log(2)), "unit": "ng/mL"}
    report = {"candidate_results": [{"scenario_id": "q8", "candidate_id": "c0", "compound": "drug8",
                                     "publication_cluster_id": "study-0", "identity_match": True, "scalars": [scalar]}],
              "selection": [{"scenario_id": f"q{i}", "stratum": "reference_diagnostic",
                             "required_endpoints": ["Cmax_reference_vs_group", "AUC_infinity"]} for i in (8, 9)]}
    return d.assemble(queries[:8], core, clinical, candidates, scenarios, labels, report, "p")


def test_labels_unknown_and_invalid_are_distinct():
    rows = [judgment("q", "yes"), judgment("q", "no", 0, False), judgment("q", "unknown", 1, None),
            judgment("q", "unresolved", None, None, "insufficient_evidence"),
            judgment("q", "invalid", None, None, "invalid_output")]
    grades, apps = d.document_labels(rows, ["q", "other"], ["yes", "no", "unknown", "unresolved", "invalid"], ["q"], "p")
    assert grades[0].tolist() == [3, 0, 1, -1, -1]
    assert apps[0].tolist() == [0, 1, 2, 2, -1]
    assert (grades[1] == -1).all() and (apps[1] == -1).all()


@pytest.mark.parametrize("change", ["duplicate", "wrong_protocol", "missing", "unknown_candidate"])
def test_document_cartesian_boundary(change):
    rows = [judgment("q", "c")]
    if change == "duplicate": rows *= 2
    if change == "wrong_protocol": rows[0]["protocol_sha256"] = "wrong"
    if change == "missing": rows = []
    if change == "unknown_candidate": rows[0]["candidate_id"] = "elsewhere"
    with pytest.raises(ValueError):
        d.document_labels(rows, ["q"], ["c"], ["q"], "p")


def test_three_clinical_condition_pairs_are_distinguishable_without_outcomes():
    q, c = query(0), candidate(0)
    pairs = [({"dose": {"value": 250, "unit": "mg"}}, {"dose": {"value": 50, "unit": "mg"}}),
             ({"food": "fasted"}, {"food": "fed_30_minutes_after_meal"}),
             ({"dose": {"value": 4, "unit": "mg/kg"}, "regimen": {"infusion_duration_h": 4 / 3}},
              {"dose": {"value": 6, "unit": "mg/kg"}, "regimen": {"infusion_duration_h": 2}})]
    for a, b in pairs:
        ar, br = d.pair_feature_row(q, c, design=a), d.pair_feature_row(q, c, design=b)
        assert np.array_equal(ar[:60], br[:60])
        assert not np.array_equal(ar, br)


def test_missing_is_not_zero_dose_and_perkg_is_not_absolute():
    q = query(0)
    missing = d.condition_values(q)
    zero = d.condition_values(q, design={"dose": {"value": 0, "unit": "mg"}})
    perkg = d.condition_values(q, design={"dose": {"value": 4, "unit": "mg/kg"}})
    assert missing["numeric"]["dose_mg"] is None
    assert zero["numeric"]["dose_mg"] == 0
    assert perkg["numeric"]["dose_mg"] is None and perkg["numeric"]["dose_mg_per_kg"] == 4


def test_exact_endpoint_windows_are_distinct():
    a = d.condition_values(query(0, text="Target endpoints: AUC_0_24h, Cmax"))
    b = d.condition_values(query(0, text="Target endpoints: AUC_0_72h, Cmax"))
    assert a["numeric"]["auc_end_h"] == 24 and b["numeric"]["auc_end_h"] == 72
    tau = d.condition_values(query(0, text="Target endpoints: AUC_tau_ss"), design={"regimen": {"dose_times_h": [0, 12, 24]}})
    assert tau["numeric"]["auc_end_h"] == 12 and tau["categorical"]["endpoint_window"] == ["steady_state", "tau"]


def test_missing_evidence_has_its_own_features_and_original_texts_unchanged(assembled):
    assert assembled["query_texts"][0][0] == query(0)["text"]
    assert assembled["query_texts"][1][0] == query(0)["variants"]["reordered"]["text"]
    assert assembled["query_texts"][2][0] == query(0)["variants"]["missing_evidence"]["text"]
    species_present = assembled["pair_feature_names"].index("species.query_present")
    assert assembled["pair_features_by_variant"][0, 0, 0, species_present] == 1
    assert assembled["pair_features_by_variant"][2, 0, 0, species_present] == 0
    assert np.array_equal(assembled["pair_features_by_variant"][0], assembled["pair_features"])


def test_condition_ablation_removes_text_and_structural_formulation(assembled):
    text = 'Find caffeine.\nAdministration: {"route":"oral","value":250,"unit":"mg","formulation":"tablet"}\nFood context: fed\nRegimen and observation origin: {"dose_times_h":[0,12]}\nTarget endpoints: AUC_0_24h, Cmax'
    masked = d.without_conditions(text)
    assert "caffeine" in masked and "oral" in masked and "Cmax" in masked
    assert not any(x in masked for x in ("250", "tablet", "fed", "24", "12"))
    names = assembled["pair_feature_names"]
    assert all(i in assembled["numeric_feature_indices"] for i, name in enumerate(names) if name.startswith(("condition.", "formulations.")))


def test_no_label_support_or_scalar_in_feature_records(assembled):
    feature_inputs = {k: assembled[k] for k in ("query_records", "candidate_records", "query_texts", "document_texts")}
    raw = json.dumps(feature_inputs)
    assert "HIDDEN TARGET EVIDENCE" not in raw
    assert all(key not in raw for key in ('"observed"', '"predicted"', '"absolute_log_error"', '"grade"'))
    assert assembled["error_records"][0]["observed"] == 1
    assert "sealed_references" not in assembled["execution_scenarios"][0]
    assert assembled["execution_requested_endpoints"][9] == ["Cmax_reference_vs_group", "AUC_infinity"]


def test_transitive_groups_prevent_all_three_leakage_modes():
    keys = [["compound:a", "evidence:shared"], ["compound:b", "evidence:shared"],
            ["compound:b", "study:c"], ["compound:d", "study:c"]] + [[f"compound:{i}"] for i in range(10)]
    groups = d.connected_groups(keys)
    assert len(set(groups[:4])) == 1
    assert d.nested_folds(groups) == d.nested_folds(groups)
    for fold in d.nested_folds(groups):
        for a, b in [(fold["outer_train"], fold["outer_test"])] + [(f["train"], f["validation"]) for f in fold["inner"]]:
            assert not {groups[i] for i in a} & {groups[i] for i in b}


def test_coarse_database_not_a_study_and_article_indices_are_preserved():
    a = query(0, provenance={"evidence": [{"source_id": "pubmed", "source_location": "pubmed:batch.xml#xml_path=//PubmedArticle[1]/Abstract", "evidence_text_sha256": "a"}]})
    b = query(1, provenance={"evidence": [{"source_id": "pubmed", "source_location": "pubmed:batch.xml#xml_path=//PubmedArticle[2]/Abstract", "evidence_text_sha256": "b"}]})
    assert not set(d.research_keys(a)) & set(d.research_keys(b))
    b["provenance"]["evidence"][0]["source_location"] = "pubmed:batch.xml#xml_path=//PubmedArticle[1]/Title"
    assert set(d.research_keys(a)) & set(d.research_keys(b))


def test_equal_structured_zero_is_not_shared_research_but_equal_abstract_is():
    zero_hash = hashlib.sha256(b"0").hexdigest()
    evidence = {"source_id": "chembl", "source_location": "chembl:a.json#json_path=$.molecules[0].molecule_properties.hbd",
                "evidence_text_sha256": zero_hash}
    a, b = query(0, provenance={"evidence": [evidence]}), query(1, provenance={"evidence": [copy.deepcopy(evidence)]})
    assert d.atomic_property(evidence) == ("chembl", "hbd")
    assert not set(d.research_keys(a)) & set(d.research_keys(b))
    narrative = {"source_id": "pubmed", "source_location": "pubmed:a.xml#xml_path=//PubmedArticle[1]/Abstract", "evidence_text_sha256": "exact-abstract"}
    a["provenance"]["evidence"].append(narrative)
    b["provenance"]["evidence"].append({**narrative, "source_location": "pubmed:other.xml#xml_path=//PubmedArticle[9]/Abstract"})
    assert "evidence:exact-abstract" in set(d.research_keys(a)) & set(d.research_keys(b))
    assert d.connected_groups([d.research_keys(a), d.research_keys(b)])[0] == d.connected_groups([d.research_keys(a), d.research_keys(b)])[1]


def test_labels_cannot_change_fold_assignment(assembled):
    groups = assembled["groups"]
    before = d.nested_folds(groups, core_indices=assembled["core_indices"], execution_indices=assembled["execution_indices"])
    assembled["grades"][:] = 0
    assembled["error_records"][0]["abs_log_error"] = 100
    assert before == d.nested_folds(groups, core_indices=assembled["core_indices"], execution_indices=assembled["execution_indices"])


def test_roundtrip_hash_binding_and_no_overwrite(assembled, monkeypatch):
    base = d.ROOT / "runtime/tmp"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pbpk-v2-test-", dir=base) as folder:
        out = Path(folder)
        monkeypatch.setattr(d, "PINNED", {})
        np.savez_compressed(out / "arrays.npz", **{k: assembled[k] for k in d.ARRAY_NAMES})
        d._write_exclusive(out / "dataset.json", d.json_bytes({k: v for k, v in assembled.items() if k not in d.ARRAY_NAMES}))
        manifest = {"schema_version": d.SCHEMA, "completed": True, "inputs": [],
                    "producer": {"path": str(Path(d.__file__).relative_to(d.ROOT)), "sha256": d.digest(d.__file__)},
                    "outputs": [{"path": str((out / name).relative_to(d.ROOT)), "sha256": d.digest(out / name)} for name in ("dataset.json", "arrays.npz")]}
        d._write_exclusive(out / "manifest.json", d.json_bytes(manifest))
        loaded = d.load_dataset(out)
        assert np.array_equal(loaded["pair_features_by_variant"], assembled["pair_features_by_variant"])
        assert loaded["execution_requested_endpoints"] == assembled["execution_requested_endpoints"]
        with pytest.raises(FileExistsError): d._write_exclusive(out / "dataset.json", b"replacement")
        (out / "dataset.json").write_text("{}")
        with pytest.raises(ValueError, match="binding changed"): d.load_dataset(out)


def test_reject_sealed_and_outside_project_paths():
    with pytest.raises(ValueError): d.scoped(d.ROOT / "experiments/sealed/target.json")
    with pytest.raises(ValueError): d.scoped(Path("/tmp/outside-project"))


def test_validation_rejects_group_split_and_nan(assembled):
    wrong = copy.deepcopy(assembled)
    wrong["pair_features"][0, 0, 0] = np.nan
    with pytest.raises(ValueError): d.validate_dataset(wrong)
    wrong = copy.deepcopy(assembled)
    train, test = wrong["folds"][0]["outer_train"][0], wrong["folds"][0]["outer_test"][0]
    wrong["groups"][train] = wrong["groups"][test]
    with pytest.raises(ValueError, match="leakage"): d.validate_dataset(wrong)
