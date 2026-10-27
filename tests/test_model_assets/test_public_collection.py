from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from pbpk_kb.model_assets.public_collection import CollectionBudget, CollectionSafetyError, safe_extract
from pbpk_kb.model_assets.snapshot_catalog import describe_snapshot, resolve_target_names


def snapshot():
    return {"Version": 80, "Compounds": [{"Name": "Drug"}, {"Name": "Metabolite"}],
            "Individuals": [{"Name": "adult", "OriginData": {"Species": "Human", "Population": "European"}}],
            "Protocols": [{"Name": "oral", "Schemas": [{"SchemaItems": [{"ApplicationType": "Oral"}]}]}],
            "Simulations": [{"Name": "run", "Model": "4Comp", "Individual": "adult",
                             "Compounds": [{"Name": "Drug", "Protocol": {"Name": "oral"},
                                            "Processes": [{"MetaboliteName": "Metabolite"}]}, {"Name": "Metabolite"}]}]}


def test_snapshot_distinguishes_dosed_parent_from_metabolite():
    result = describe_snapshot(snapshot())
    assert result["administered_compounds"] == ["Drug"]
    assert result["metabolites"] == ["Metabolite"]
    assert result["routes"] == ["Oral"]
    assert result["species"] == ["Human"]
    assert result["simulation_count"] == 1


@pytest.mark.parametrize("mutation", ["missing_compound", "missing_individual", "missing_protocol", "duplicate_simulation", "empty_simulations"])
def test_snapshot_rejects_unresolved_or_empty_models(mutation):
    data = snapshot()
    if mutation == "missing_compound":
        data["Compounds"].pop()
    elif mutation == "missing_individual":
        data["Individuals"] = []
    elif mutation == "missing_protocol":
        data["Protocols"] = []
    elif mutation == "duplicate_simulation":
        data["Simulations"] *= 2
    else:
        data["Simulations"] = []
    with pytest.raises(ValueError):
        describe_snapshot(data)


def test_format_only_snapshot_changes_are_not_new_models():
    data = snapshot()
    reformatted = json.loads(json.dumps(data, sort_keys=True, indent=4))
    assert describe_snapshot(data)["semantic_sha256"] == describe_snapshot(reformatted)["semantic_sha256"]


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate", "expansion_limit"])
def test_archive_extraction_blocks_unsafe_or_unbounded_content(tmp_path, kind):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        name = "repo/../../escape" if kind == "traversal" else "repo/model.json"
        member = tarfile.TarInfo(name)
        member.size = 4
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
            member.size = 0
        writer.addfile(member, io.BytesIO(b"data"))
        if kind == "duplicate":
            writer.addfile(member, io.BytesIO(b"data"))
    with pytest.raises(CollectionSafetyError):
        safe_extract(archive, tmp_path / "source", CollectionBudget(tmp_path, min_free=0),
                     max_bytes=3 if kind == "expansion_limit" else 100)


def test_disk_budget_stops_all_following_writes(tmp_path):
    budget = CollectionBudget(tmp_path, max_bytes=3, min_free=0)
    budget.reserve(3)
    with pytest.raises(CollectionSafetyError):
        budget.reserve(1)
    with pytest.raises(CollectionSafetyError):
        budget.reserve(0)


def test_parameter_set_suffix_is_not_an_independent_compound():
    rows = resolve_target_names(["Bupropion", "Bupropion_FM"], ["Bupropion"], "Bupropion-Model")
    assert {r["canonical_name"] for r in rows} == {"Bupropion"}


def test_stereoisomers_are_not_collapsed_into_a_racemate():
    rows = resolve_target_names(["R-Metoprolol", "S-Metoprolol"], ["Metoprolol"], "Metoprolol-Model")
    assert {r["canonical_name"] for r in rows} == {"R-Metoprolol", "S-Metoprolol"}


def test_isotope_and_unreviewed_abbreviation_do_not_create_parent_identity():
    assert resolve_target_names(["[15N]-Carbamazepine"], ["Carbamazepine"], "Carbamazepine-Model") == []
    assert resolve_target_names(["CsA"], ["Cyclosporine"], "Unreviewed-Model") == []
    assert resolve_target_names(["CsA"], ["Cyclosporine"], "Cyclosporine-Model")[0]["canonical_name"] == "Cyclosporine"


@pytest.mark.parametrize("corrupt", [False, True])
def test_selected_sources_check_git_blob_and_never_leave_partial_assets(tmp_path, monkeypatch, corrupt):
    from pbpk_kb.model_assets import public_collection as collector
    payload = b'{"model": "test"}'
    tree = {"truncated": False, "tree": [{"path": "Model.json", "type": "blob", "mode": "100644", "size": len(payload),
              "sha": "0" * 40 if corrupt else hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}]}
    monkeypatch.setattr(collector, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collector, "_open", lambda *args, **kwargs: io.BytesIO(payload))
    if corrupt:
        with pytest.raises(CollectionSafetyError, match="Git blob"):
            collector.collect_source_files("Open-Systems-Pharmacology/Test-Model", "a" * 40, tree, ["Model.json"], tmp_path / "assets")
        assert not list((tmp_path / "assets").glob("*.partial"))
    else:
        result = collector.collect_source_files("Open-Systems-Pharmacology/Test-Model", "a" * 40, tree, ["Model.json"], tmp_path / "assets")
        collector.verify_manifest(result)
        assert result["collection_scope"] == "selected_sources_with_git_blob_verification"
