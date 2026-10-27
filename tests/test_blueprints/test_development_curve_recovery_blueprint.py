"""Recovery policy persistence and older locked mapping compatibility."""
import json

import pytest

from pbpk_kb.blueprints import development_curve_blueprint as dc
from pbpk_kb.blueprints import development_curve_optimizer as optimizer
from test_pk_anchored_blueprint import evidence_case
from test_development_curve_blueprint import development_case, _native_success, _mapping


@pytest.mark.parametrize("enabled", [False, True])
def test_fit_passes_explicit_policy_and_persists_version(development_case, monkeypatch, enabled):
    root, request, anchor, _, source = development_case
    actual = optimizer.optimize_development_curve
    passed = []
    def spy(*args, **kwargs):
        passed.append(kwargs["recover_failed_trials"])
        return actual(*args, **kwargs)
    monkeypatch.setattr(optimizer, "optimize_development_curve", spy)
    monkeypatch.setattr(dc, "_run", _native_success)
    fit = dc.fit_development_mapping(reference_request=request, training_source=source,
        vss_anchor=anchor, output_dir=root/"fit", project_root=root, max_evaluations=1,
        recover_failed_trials=enabled)
    assert passed == [enabled]
    assert fit["version"] == "development-curve-pbpk-1.2"
    assert fit["failure_recovery"]["version"] == "failed-search-backtrack-1.0"
    assert fit["failure_recovery"]["enabled"] is enabled
    saved = json.loads((root/"fit/mapping_locked.json").read_text())
    assert saved["failure_recovery"] == fit["failure_recovery"]
    assert json.loads((root/"fit/input.json").read_text())["recover_failed_trials"] is enabled


@pytest.mark.parametrize("version", ["development-curve-pbpk-1.0", "development-curve-pbpk-1.1", "development-curve-pbpk-1.2"])
def test_prediction_accepts_old_and_new_mapping_versions(development_case, monkeypatch, version):
    root, request, _, _, _ = development_case
    mapping = _mapping(development_case)
    mapping["version"] = version
    path = root/"mapping.json"
    path.write_text(json.dumps(mapping))
    before = path.read_bytes()
    monkeypatch.setattr(dc, "_run", _native_success)
    result = dc.predict_with_development_mapping(request=request, mapping_path=path,
        output_dir=root/"prediction", project_root=root)
    assert result["status"] == "predicted"
    assert path.read_bytes() == before
    assert json.loads((root/"prediction/construction.json").read_text())["mapping_version"] == version


def test_bad_recovery_policy_rejected_before_creating_output(development_case):
    root, request, anchor, _, source = development_case
    with pytest.raises(ValueError, match="boolean"):
        dc.fit_development_mapping(reference_request=request, training_source=source,
            vss_anchor=anchor, output_dir=root/"fit", project_root=root, recover_failed_trials="true")
    assert not (root/"fit").exists()
