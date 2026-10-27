"""Engineering output checks; these fixtures never provide scientific PK labels."""
import json
import runpy
from pathlib import Path

import pytest

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[2]/"scripts/run_snapshot_model_pilot.py"))
validate = HELPERS["validate_snapshot_outputs"]


def fixture_outputs(tmp_path):
    output = tmp_path / "execution"; (output/"outputs").mkdir(parents=True)
    (output/"execution_summary.json").write_text(json.dumps({"source_sha256":"source", "source_unchanged":True,"returncode":0,"stop_reason":None}))
    (output/"outputs/outputs.csv").write_text('"Sim_id"\nsimulation\n')
    (output/"outputs/model-simulation.pkml").write_text("<SimulationTransfer/>")
    (output/"outputs/model-simulation-Results.csv").write_text('IndividualId,Time [min],Concentration\n0,0,0\n0,1,2\n')
    return output,{"Simulations":[{"Name":"simulation"}]}


def test_export_index_is_not_misclassified_as_a_concentration_curve(tmp_path):
    output,snapshot=fixture_outputs(tmp_path)
    result=validate(output,snapshot,"source")
    assert result["engineering_accepted"] and result["csv_result_count"]==1
    assert result["scientific_accepted"] is False
    assert result["independent_observations_verified"] is False
    assert result["registry_modified"] is False


@pytest.mark.parametrize("fault",["nan","no_signal","time_reversal","missing_pkml","wrong_index","source_drift","runner_failure"])
def test_incomplete_or_invalid_execution_is_rejected(tmp_path,fault):
    output,snapshot=fixture_outputs(tmp_path)
    curve=output/"outputs/model-simulation-Results.csv"
    if fault=="nan":curve.write_text(curve.read_text().replace("0,1,2","0,1,NaN"))
    if fault=="no_signal":curve.write_text(curve.read_text().replace("0,1,2","0,1,0"))
    if fault=="time_reversal":curve.write_text(curve.read_text()+"0,0.5,1\n")
    if fault=="missing_pkml":(output/"outputs/model-simulation.pkml").unlink()
    if fault=="wrong_index":(output/"outputs/outputs.csv").write_text('"Sim_id"\nother\n')
    if fault=="runner_failure":
        path=output/"execution_summary.json";row=json.loads(path.read_text());row["returncode"]=1;path.write_text(json.dumps(row))
    result=validate(output,snapshot,"changed" if fault=="source_drift" else "source")
    assert result["engineering_accepted"] is False
