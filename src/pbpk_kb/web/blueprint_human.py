"""Rebuild a web scenario's physiology through the existing native audit chain.

Every call owns a new workspace job directory. The three native stages already
enforce two CPU threads and 180 seconds each, with output thresholds of 128,
64, and 128 MiB. The native construction solve is an engineering prerequisite;
its concentrations are not used for fitting or for the web prediction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Callable

from pbpk_kb.blueprint_policy.perfusion_calibration import _physiology
from pbpk_kb.blueprint_policy.perfusion_export import ORGANS
from pbpk_kb.blueprints.design_runtime import _directory_bytes
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.evidence_preserving_calibration import InfrastructurePause
from pbpk_kb.blueprints.source_workflow import inside, sha256


BUILD_TIMEOUT_SECONDS = 600
MAX_OUTPUT_BYTES = 384 * 1024**2
WORKSPACE_NATIVE_LIMIT_BYTES = 2 * 1024**3


def _native_workspace_bytes(workspace: Path, output: Path) -> int:
    """Count retained native job artifacts without traversing other resources."""
    jobs = workspace / 'blueprints'
    if jobs.is_symlink():
        raise ValueError('Blueprint job directory must not be a symbolic link')
    total, included = 0, False
    if jobs.exists():
        for job in jobs.iterdir():
            if job.is_symlink():
                raise ValueError('Blueprint job directory must not be a symbolic link')
            if not job.is_dir():
                continue
            native = job / 'human'
            if native.is_symlink():
                raise ValueError('Human output directory must not be a symbolic link')
            if native.exists():
                total += _directory_bytes(native)
                included = included or native == output
    # Include a caller-owned workspace verification directory as well.
    if not included and output.exists():
        total += _directory_bytes(output)
    return total


def rebuild_physiology(
    request: dict, output: Path, root: Path,
    before_trial: Callable[[], None] | None = None,
) -> dict:
    """Return audited native physiology for the supplied human and formula.

    The caller must set ``request['partition_method']`` to the saved calibration
    formula *before* calling, and must use the same human in fitting/prediction.
    ``manifest`` in the returned document points to the newly exported JSON;
    no duplicate physiology file or frozen experimental input is written.
    """
    root = Path(root).resolve(strict=True)
    raw = Path(output)
    if not raw.is_absolute():
        raw = root / raw
    workspace = root / 'runtime/deployment/workspace'
    if '..' in raw.parts or raw == workspace or not raw.is_relative_to(workspace):
        raise ValueError('Human reconstruction output must use a separate job directory within this project workspace')
    output = inside(raw, root)
    if not output.is_relative_to(workspace):
        raise ValueError('Human reconstruction output must not escape the workspace through links')
    if output.exists():
        raise ValueError('Human reconstruction requires a new job directory and cannot reuse old physiology results')

    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
    if req['dosing']['route'] != 'intravenous':
        raise ValueError('Human reconstruction currently supports adults, one drug, and a single intravenous dose')
    started = time.monotonic()

    def guard():
        if before_trial is not None:
            before_trial()
        if time.monotonic() - started > BUILD_TIMEOUT_SECONDS:
            raise TimeoutError('Human reconstruction exceeded the 600-second limit; the job stopped without automatic retries')
        # Native writers independently check their bounded directory every
        # 0.2 seconds; this adds a combined check between stages and at return.
        if output.exists() and _directory_bytes(output) > MAX_OUTPUT_BYTES:
            raise ValueError('Human reconstruction output exceeded its size limit; the job stopped')
        if _native_workspace_bytes(workspace, output) > WORKSPACE_NATIVE_LIMIT_BYTES:
            raise ValueError('Human job files reached the 2 GiB limit. Download and organize saved jobs before generating another; old results will not be deleted automatically')

    guard()
    try:
        physiology = _physiology(req, output, root, guard)
    except InfrastructurePause as exc:
        raise ValueError('Native human construction or validation failed. Old physiology was not substituted; job diagnostics were preserved') from exc
    guard()

    # Bind returned physiology to this exact new native execution, not an
    # arbitrary cached or substituted document with a matching display weight.
    completion = json.loads((output / 'base/completion.json').read_text())
    if completion['request'] != req or completion.get('actual_native_solver_calls') != 1:
        raise ValueError('Human reconstruction record does not match the submitted conditions')
    reference = completion['execution_manifest']
    manifest_path = inside(reference['path'], root, exists=True)
    if (manifest_path.parent != output / 'base/native'
            or sha256(manifest_path) != reference['sha256']
            or physiology.get('source_execution_manifest_sha256') != reference['sha256']):
        raise ValueError('Native execution source verification failed for the new human conditions')
    execution = json.loads(manifest_path.read_text())
    snapshot_path = manifest_path.parent / 'snapshot.json'
    if (execution.get('engineering_accepted') is not True
            or sha256(snapshot_path) != execution.get('source_sha256')):
        raise ValueError('Native execution or snapshot verification failed for the new human conditions')
    snapshot = json.loads(snapshot_path.read_text())
    human = req['human']
    expected_origin = {
        'Species': 'Human', 'Population': human['population'], 'Gender': human['sex'],
        'Age': {'Value': human['age_years'], 'Unit': 'year(s)'},
        'Weight': {'Value': human['weight_kg'], 'Unit': 'kg'},
        'Height': {'Value': human['height_cm'], 'Unit': 'cm'},
    }
    individuals = snapshot.get('Individuals', [])
    if (len(individuals) != 1 or individuals[0].get('OriginData') != expected_origin
            or individuals[0].get('Seed') != human['seed']
            or snapshot['Simulations'][0].get('Individual') != individuals[0].get('Name')):
        raise ValueError('Native human conditions do not match the submitted age, sex, height, and weight')
    method = 'PK-Sim Standard' if req['partition_method'] == 'PKSimStandard' else 'Rodgers and Rowland'
    expected_method = 'Cellular partition coefficient method - ' + method
    if expected_method not in snapshot['Compounds'][0].get('CalculationMethods', []):
        raise ValueError('The new human partition method differs from the saved calibration protocol')
    if physiology.get('compound') != req['compound_name']:
        raise ValueError('The native drug readback differs from the submitted conditions')
    # Native parameter serialization introduces small floating-point differences;
    # use the same relative tolerance as the existing native physiology audits.
    if not math.isclose(float(physiology['weight_kg']), human['weight_kg'], rel_tol=1e-6):
        raise ValueError('The native body-weight readback differs from the submitted conditions')
    if (len(physiology['organs']) != len(ORGANS)
            or {organ['name'] for organ in physiology['organs']} != set(ORGANS)):
        raise ValueError('The native organ set does not match the expected scope')
    exported = physiology['manifest']
    exported_path = inside(exported['path'], root, exists=True)
    if (exported_path != output / 'physiology/perfusion_physiology.json'
            or sha256(exported_path) != exported['sha256']):
        raise ValueError('New human physiology export verification failed')
    guard()
    return physiology
