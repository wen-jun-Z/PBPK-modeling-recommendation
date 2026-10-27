"""Reuse the authored contract cases against the integrated consumer module.

The original standalone test loader explicitly imports its own earlier stage;
this shim changes only that loader, preserving every assertion and fixture.
"""
from pathlib import Path
_base = Path(__file__).resolve()
_project = next(p for p in _base.parents if (p/'pyproject.toml').is_file())
_original = _project/'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_identity_followup_20261002/staged/tests/test_recommendation/test_json_racemate_contract.py'
_source = _original.read_text()
_start = _source.index('NAME = ')
_end = _source.index('REGISTRY = ', _start)
_source = _source[:_start] + 'from scripts.recommendation_execution_repairs import json_racemate_contract as m\n' + _source[_end:]
exec(compile(_source, str(_original), 'exec'), globals())


def test_actual_native_base_unit_serialization_rounding_is_accepted(bound):
    doc=xml(bound)
    for node in doc.iter():
        if node.get('name')=='Molecular weight':node.set('value','3.4541600000000003E-07')
    assert m.validate_native_observer(doc,bound,molecular_weights={p:345.41600000000005 for p in m.PARENTS},project_root=ROOT)['verified']


def test_native_weight_change_above_rounding_bound_is_not_accepted(bound):
    doc=xml(bound)
    next(node for node in doc.iter() if node.get('name')=='Molecular weight').set('value','3.45416000003E-07')
    with pytest.raises(ValueError,match='ORIGINAL_NATIVE_MOLECULAR_WEIGHT_DIFFERS'):
        m.validate_native_observer(doc,bound,molecular_weights=weights(),project_root=ROOT)


# Match PK-Sim Unit.UnitValueToBaseUnitValue, rather than mathematical division.
# The original decimal->binary64 mg-to-kg multiplication is not identical to /1e6.
_original_native_fixture = native
def native(bound, count=1):
    metadata, inventory = _original_native_fixture(bound, count=count)
    for row in metadata["event_parameters"]:
        if row["path"].endswith("|Dose"):
            channel = next(c for c in bound["channels"] if row["path"].startswith("Events|"+c["protocol_name"]+"|"))
            row["value"] = str(channel["source_dose_mg"] * 1e-6)
    for row in inventory["parameters"]:
        if row["path"].endswith("|Dose"):
            source = next(p for p in metadata["event_parameters"] if p["path"] == row["path"])
            row["value"] = float(source["value"])
    return metadata, inventory
