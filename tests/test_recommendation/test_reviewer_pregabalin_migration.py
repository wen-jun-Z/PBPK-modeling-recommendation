from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("reviewer_pregabalin", ROOT / "scripts/reviewer_revalidate_pregabalin_snapshot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair():
    source = {"Individuals": [{"Name": "adult"}],
              "Compounds": [{"Name": "drug", "Processes": [{"InternalName": "GlomerularFiltration", "DataSource": "ref", "Parameters": [{"Name": "GFRfraction", "Value": 1.0}]}]}],
              "Simulations": [{"Name": "oral", "Compounds": [{"Name": "drug", "Processes": [{"Name": "Glomerular Filtration-ref"}]}]}]}
    converted = deepcopy(source)
    converted["Individuals"][0]["ExpressionProfiles"] = []
    converted["Simulations"][0]["Compounds"][0]["Processes"][0]["SystemicProcessType"] = "GFR"
    return source, converted


def test_scoped_metadata_completion_accepted():
    counts, comparison = module.review(*pair())
    assert counts == {"empty_expression_profile_added": 1, "explicit_gfr_process_type_added": 1}
    assert comparison["accepted"]


@pytest.mark.parametrize("mutation", ["new_expression", "parameter_changed", "wrong_process"])
def test_real_or_unestablished_changes_rejected(mutation):
    source, converted = pair()
    if mutation == "new_expression":
        converted["Individuals"][0]["ExpressionProfiles"] = [{"Molecule": "new"}]
    elif mutation == "parameter_changed":
        converted["Compounds"][0]["Processes"][0]["Parameters"][0]["Value"] = 1.5
    else:
        source["Compounds"][0]["Processes"][0]["InternalName"] = "HepaticClearance"
    with pytest.raises(ValueError):
        module.review(source, converted)


def test_resolver_rejects_other_candidates_before_access():
    import reviewer_external_migration as resolver
    with pytest.raises(ValueError, match="restricted"):
        resolver.resolve_reviewed({"source_sha256": "0" * 64, "source_path": "other.json"})


def test_exact_source_resolver_rechecks_bound_native_chain():
    import reviewer_external_migration as resolver
    got = resolver.resolve_reviewed({"source_sha256": resolver.SOURCE_SHA, "source_path": resolver.SOURCE_REL})
    assert got["sha256"] == resolver.CONVERTED_SHA
    assert got["operation"] == "reviewed_exact_source_metadata_completion"
    assert got["reviewed_sidecar"]["sha256"] == resolver.SIDECAR_SHA
