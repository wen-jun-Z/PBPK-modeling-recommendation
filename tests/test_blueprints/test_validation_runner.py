"""Validation runner contracts using synthetic project-local files only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_blueprint_validation as runner
from pbpk_kb.blueprints import evidence_blueprint


def source_ref(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return source_ref(path)


def development():
    return {"compound_id": "synthetic", "compound_name": "Synthetic",
            "study_id": "synthetic-development", "role": "calibration_data_reconstruction",
            "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
            "molecular_weight_g_mol": 400.,
            "dosing": {"route": "intravenous", "dose_mg": 10., "start_h": 0., "infusion_minutes": 30.},
            "human": {"weight_kg": 70.},
            "observations": [{"time_min": t, "concentration_umol_per_l": 7-i*.5}
                             for i, t in enumerate([10., 30., 60., 90., 150., 300., 600.])],
            "raw_observations": [{"time_min": 1., "concentration_umol_per_l": 1e9}],
            "endpoint_values": {"auc": 100000}, "old_full_fit": {"CL": 1234.}}


def test_five_selection_uses_times_only_and_subset_drops_hidden_labels():
    doc = development()
    selected = runner.select_five(doc)
    altered = deepcopy(doc)
    for i, row in enumerate(altered["observations"]):
        row["concentration_umol_per_l"] = 10.**(i-4)
    assert runner.select_five(altered) == selected
    assert len(selected) == 5 and selected[0] == 0 and selected[-1] == 6
    assert 1 in selected
    assert sum(doc["observations"][i]["time_min"] > 30 for i in selected) >= 3
    subset = runner.make_subset(doc, {"path": "old_full.json", "sha256": "0"*64})
    assert not {"raw_observations", "endpoint_values", "old_full_fit"} & subset.keys()
    assert subset["observations"] == [doc["observations"][i] for i in selected]
    assert subset["selection"]["concentrations_used_for_selection"] is False
    assert subset["selection"]["full_curve_initialization_allowed"] is False
    subset["observations"][0]["concentration_umol_per_l"] *= 2
    assert doc == development()


@pytest.mark.parametrize("times", [[0, 10, 20, 40, 50], [1, 2, 3, 4, 5],
                                   [10, 30, 60, 60, 90], [10, 30, 60, 90]])
def test_five_selection_rejects_nonqualifying_sampling(times):
    doc = development()
    doc["observations"] = [{"time_min": t, "concentration_umol_per_l": 1.} for t in times]
    with pytest.raises(ValueError):
        runner.select_five(doc)


@pytest.fixture
def synthetic_batch(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp under the project runtime directory")
    current, base, output = tmp_path/"current", tmp_path/"base", tmp_path/"new_run"
    names = ["Treprostinil", "Ertapenem", "Ceftriaxone", "Cefepime", "Dexmedetomidine",
             "Edaravone", "Midazolam", "Moxifloxacin", "Nemonoxacin", "Remdesivir"]
    updated = {"Dexmedetomidine", "Moxifloxacin", "Nemonoxacin", "Remdesivir"}
    cases = []
    clinical = tmp_path/"do_not_decode_target.json"
    clinical.write_text("NOT JSON: outcome contents must not be decoded by prepare")
    for name in names:
        cid = name.lower()
        doc = development()
        doc.update(compound_id=cid, study_id=cid+"-development")
        doc_ref = save(tmp_path/"sources"/(cid+".json"), doc)
        request = {"compound_id": cid, "human": doc["human"], "dosing": doc["dosing"],
                   "evidence": [], "target": {"window_start_h": 0., "window_end_h": 12.}}
        evidence_request = deepcopy(request)
        if name in updated:
            evidence_request["evidence"] = [{"synthetic_independent_update": True}]
        for phase in ("development", "target"):
            save(current/"inputs"/cid/(phase+".json"), evidence_request)
            save(base/"inputs"/cid/(phase+"_request.json"), request)
        cases.append({"case_id": cid, "compound_id": cid, "drug": name, "study_id": cid+"-target",
                      "development": doc_ref["path"], "development_source": {
                          **doc_ref, "role": "calibration_data_reconstruction"},
                      "reference_source": source_ref(clinical),
                      "updated_inputs": {"fraction_unbound": .5} if name in updated else {},
                      "target_request_metadata": {"human": doc["human"], "dosing": doc["dosing"],
                                                  "window_end_h": 12.},
                      "nca_policy": {"sampling_time_h": [0, .5, 1, 2, 4, 12]}})
    save(current/"protocol.json", {"cases": cases, "source_pins": {
        c["development_source"]["path"]: c["development_source"]["sha256"] for c in cases}})
    monkeypatch.setattr(runner, "CURRENT", current)
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(evidence_blueprint, "assemble_evidence_blueprint",
                        lambda request, root: {"status": "assembled"})
    return current, base, output, cases


def test_prepare_pins_subsets_and_complete_design_without_decoding_targets(synthetic_batch):
    current, base, output, cases = synthetic_batch
    previous = {str(p): p.read_bytes() for folder in (current, base) for p in folder.rglob("*") if p.is_file()}
    result = runner.prepare(output, workers=14)
    protocol = json.loads((output/"protocol.json").read_text())
    assert result["status"] == "prepared"
    assert protocol["workers"] == 14 and protocol["native_threads"] == 2
    assert len(runner.jobs(protocol)) == 76
    assert sum(job[-1] for job in runner.jobs(protocol)) == 2304
    assert len([c for c in protocol["cases"] if c["effective_evidence_ablation"]]) == 4
    assert len([c for c in protocol["cases"] if c["transfer_group"] == "primary"]) == 4
    for case in protocol["cases"]:
        subset = runner.checked(case["sampling_sources"]["five"])
        assert len(subset["observations"]) == 5
        assert not {"raw_observations", "endpoint_values", "old_full_fit"} & subset.keys()
        assert case["sampling_sources"]["five"]["path"] in protocol["source_pins"]
        assert case["sampling_sources"]["full"] == case["development_source"]
    assert next(c for c in protocol["cases"] if c["drug"] == "Ertapenem")["cluster_id"] == "Pletz2004"
    assert next(c for c in protocol["cases"] if c["drug"] == "Ceftriaxone")["cluster_id"] == "Pletz2004"
    assert {str(p): p.read_bytes() for folder in (current, base) for p in folder.rglob("*") if p.is_file()} == previous
    altered = ROOT/protocol["cases"][0]["sampling_sources"]["five"]["path"]
    altered.write_text('{"altered": true}')
    with pytest.raises(ValueError):
        runner.verify(protocol["source_pins"])
    with pytest.raises(FileExistsError):
        runner.prepare(output)


def test_prepare_rejects_nonidentical_alias_and_report_is_gated(synthetic_batch, monkeypatch):
    current, base, output, cases = synthetic_batch
    path = current/"inputs"/"treprostinil"/"development.json"
    doc = json.loads(path.read_text()); doc["changed"] = True; path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Alias requires"):
        runner.prepare(output)
    stub = output/"protocol_stub.json"
    save(stub, {})
    save(output/"summary.json", {"protocol": source_ref(stub), "units": {},
                                "all_attempts_locked_before_scoring": False})
    with pytest.raises(RuntimeError, match="100"):
        runner.reporting(output)


def test_cleanup_deletes_only_unselected_success_and_preserves_failure(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp under the project runtime directory")
    folder = tmp_path/"fit"
    refs = {}
    for index, status in ((1, "predicted"), (2, "predicted"), (3, "native_failed")):
        execution = folder/f"attempt_{index:02d}"/"execution_0"
        result = save(execution/"result.json", {"status": status})
        save(execution/"completion.json", {"result": result, "infrastructure_failure": False})
        refs[index] = result
        (execution/"native").mkdir()
        (execution/"native"/"simulation.pkml").write_text("synthetic native")
    save(folder/"mapping_locked.json", {"trial_artifact_pins": {
        refs[1]["path"]: refs[1]["sha256"], refs[3]["path"]: refs[3]["sha256"]}})
    assert runner.cleanup_fit(folder) == 1
    assert (folder/"attempt_01"/"execution_0").is_dir()
    assert not (folder/"attempt_02"/"execution_0").exists()
    assert (folder/"attempt_03"/"execution_0").is_dir()
    assert runner.cleanup_fit(folder) == 0

def test_directory_scan_tolerates_disappearing_files_but_not_permission_failure(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp under the project runtime directory")
    disappeared = tmp_path/"atomic_output.tmp"
    stable = tmp_path/"stable.json"
    disappeared.write_bytes(b"temporary")
    stable.write_bytes(b"kept")
    original = Path.stat
    calls = 0
    def changing_stat(self, *args, **kwargs):
        nonlocal calls
        if self == disappeared:
            calls += 1
            if calls >= 2:
                raise FileNotFoundError("removed after listing by another worker")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", changing_stat)
    assert runner.directory_bytes(tmp_path) == 4
    def forbidden_stat(self, *args, **kwargs):
        if self == stable:
            raise PermissionError("real filesystem access failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", forbidden_stat)
    with pytest.raises(PermissionError):
        runner.directory_bytes(tmp_path)

