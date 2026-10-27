from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
import curate_legacy_evidence as c
from pbpk_kb.processing.run_llm_extraction import _numeric_parameter_owner_error


def test_bounded_recovery_keeps_unmatched_concentration_evidence():
    from recover_bounded_evidence import parameter_foci
    foci = parameter_foci("Sildenafil was excreted in saliva; plasma concentration was 72.4 ng/mL at 1.5 h.")
    assert "excretion" in foci
    assert foci[-1].startswith("other explicit PBPK evidence")
    assert "exclude these separately assessed families: excretion" in foci[-1]
    assert parameter_foci("Unlisted but explicit fact") == ["other explicit PBPK evidence"]


def test_bounded_recovery_retries_only_current_failed_requests():
    from recover_bounded_evidence import latest_failed
    records = [
        {"request_id": "recovered", "status": "failed_local_llm", "prompt_template_version": "current"},
        {"request_id": "recovered", "status": "ok", "prompt_template_version": "current"},
        {"request_id": "remaining", "status": "failed_local_llm", "prompt_template_version": "current"},
        {"request_id": "obsolete", "status": "failed_local_llm", "prompt_template_version": "old"},
        {"request_id": "reused", "status": "reused_same_target_input_hash", "prompt_template_version": "current"},
    ]
    assert latest_failed(records, "current") == {"remaining"}


def test_bounded_recovery_stops_before_low_space_write(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import recover_bounded_evidence as bounded
    monkeypatch.setattr(bounded.shutil, "disk_usage", lambda _: SimpleNamespace(free=11 * bounded.ctl.GIB))
    with pytest.raises(RuntimeError, match="12 GiB reserve"):
        bounded.check_resources(tmp_path)


def test_separately_assigned_percentages_are_not_a_truncated_shared_value():
    quote = ("Caspofungin reduced the blood AUC 0 to 12 of tacrolimus by approximately 20%, "
             "peak blood concentration (C max ) by 16%, and 12-hour blood concentration (C 12hr ) by 26%")
    assert _numeric_parameter_owner_error({"parameter": "AUC 0 to 12", "value": 20, "unit": "%", "source_text": quote}, "tacrolimus") is None
    truncated = "The C max and AUC of voriconazole were reduced by 34%"
    assert _numeric_parameter_owner_error({"parameter": "AUC", "value": 34, "unit": "%", "source_text": truncated}, "voriconazole") == "truncated_multi_endpoint_percent_assignment"


def test_table_verification_binds_endpoint_unit_and_row(tmp_path):
    source = tmp_path / "table.xml"
    source.write_text('<root><table><tr><th>Population</th><th>Renal Clearance (mL/min)</th><th>T max (hrs)</th></tr><tr><td>Elderly</td><td>412 (±98)</td><td>2.71</td></tr></table></root>')
    content = c.source_content(source)
    row = {"extraction_method": "source_structured_table", "metadata_json": {"parameter_header": "Renal Clearance (mL/min)"},
           "source_text": "Elderly | 412 (±98) | 2.71", "parameter": "clearance", "value": 412, "unit": "mL/min"}
    assert c.table_support(row, content)
    assert not c.table_support({**row, "value": 2.71}, content)
    assert not c.table_support({**row, "unit": "L/h"}, content)
    assert not c.table_support({**row, "parameter": "Tmax"}, content)


def test_json_source_is_not_overwritten_by_receipt(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(c, "BASE", tmp_path)
    payload = b'{"molecule":"example","value":2}'

    class Response:
        is_redirect = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield payload

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs): return Response()

    monkeypatch.setattr(requests, "Session", Session)
    item = ("historical/source.json", {"source_id": "chembl", "source_url": "https://www.ebi.ac.uk/chembl/api/data/molecule.json", "checksum": "old"}, None)
    receipt = c.recover_one(item)
    assert Path(receipt["path"]).read_bytes() == payload
    assert receipt["sha256"] == sha256(payload).hexdigest()
    assert receipt["status"] == "new_source_version"
    assert c.recover_one(item) == receipt


def test_recovery_refuses_unregistered_hosts_or_credentials():
    assert not c.safe_url("https://example.com/source")
    assert not c.safe_url("https://user:secret@eutils.ncbi.nlm.nih.gov/source")
    assert not c.safe_url("https://eutils.ncbi.nlm.nih.gov/source?api_key=secret")
    assert c.safe_url("https://eutils.ncbi.nlm.nih.gov/source?db=pubmed&id=1")


def test_source_correction_requires_the_complete_statement():
    row = {"evidence_id": "91f4dfbe-32e9-55b3-95fe-6302a1d05e01", "metadata_json": {}}
    with pytest.raises(ValueError):
        c.known_corrections(row, {"texts": ["Cmax and AUC reduced by 34%"], "tables": []})
    quote = "When multiple doses of voriconazole are administered with high-fat meals, Cmax and AUC are reduced by 34% and 24%, respectively when administered as a tablet and by 58% and 37% respectively when administered as the oral suspension"
    fixed, _ = c.known_corrections(row, {"texts": [quote], "tables": []})
    assert fixed["value"] == 24 and fixed["parameter"] == "AUC_tau"
    assert fixed["source_text_hash"] == sha256(quote.encode()).hexdigest()


def test_source_typography_relocation_does_not_change_values_or_units():
    from repair_legacy_source_gaps import locate_quote
    original = "The plasma half-life is prolonged with impaired hep- atic function; clearance is 35.64 L/h."
    source = "Background. The plasma half-life is prolonged with impaired hepatic function; clearance is 35.64 L/h. Next."
    found = locate_quote(original, [source])
    assert found and "hepatic" in found and "35.64 L/h" in found
    assert locate_quote(original.replace("35.64", "67"), [source]) is None
    assert locate_quote(original.replace("L/h", "mL/min"), [source]) is None
    assert locate_quote(original.replace("35.64", "-35.64"), [source]) is None
    assert locate_quote(original, [source + source]) is None


def test_source_statistic_is_not_an_absolute_pk_parameter():
    from repair_legacy_source_gaps import endpoint_semantics_blockers
    cv = {"parameter": "clearance", "value": 20, "unit": "%", "source_text": "The clearance coefficient of variation (CV) < 20% in healthy volunteers."}
    assert endpoint_semantics_blockers(cv) == ["variability_statistic_not_parameter_value"]
    assert not endpoint_semantics_blockers({**cv, "parameter": "clearance coefficient of variation"})
    assert not endpoint_semantics_blockers({**cv, "value": 15})
    increase = {"parameter": "fraction_unbound", "value": 20, "unit": "%", "source_text": "The obstetric model added a 20% increase in fraction unbound."}
    assert endpoint_semantics_blockers(increase) == ["relative_model_assumption_not_absolute_parameter"]
