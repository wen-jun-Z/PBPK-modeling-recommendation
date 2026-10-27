"""Structure-only RF parameters with scaffold-held-out empirical stress ranges.

Calibration labels determine a residual scale only. They are not an independent
performance test, and the resulting ranges are not clinical confidence intervals.
The frozen cohort is excluded by standardized identity and scaffold from both
RF training and range calibration. No clinical concentration curve is consumed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping

import joblib
import numpy as np

from pbpk_kb.blueprints import pk_parameter_learning as base

VERSION = 'scaffold-ranged-parameter-prior-1'
SEED = 20260930
CALIBRATION_LIMIT = 100
QUANTILE = .9
PARAMETERS = {
    'total_clearance': ('human_CL_mL_min_kg', 'ml/min/kg'),
    'fraction_unbound': ('human_fup', 'fraction'),
}


def _inside(path, root, *, exists=True):
    p = (root / path).resolve(strict=exists)
    if p == root or not p.is_relative_to(root):
        raise ValueError('All sources and outputs must remain in this project')
    return p


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _ref(path, root):
    return {'path': str(path.relative_to(root)), 'sha256': _sha(path)}


def _write_new(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _verify(pins, root):
    for name, expected in pins.items():
        if _sha(_inside(name, root)) != expected:
            raise ValueError('Frozen source or artifact changed: ' + name)


def _residual_quantile(observed_log10, predicted_log10):
    observed = np.asarray(observed_log10, dtype=float)
    predicted = np.asarray(predicted_log10, dtype=float)
    if (observed.ndim != 1 or not len(observed) or observed.shape != predicted.shape
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()):
        raise ValueError('Range calibration requires matched nonempty finite log10 labels and predictions')
    errors = np.abs(predicted - observed)
    value = float(np.quantile(errors, QUANTILE, method='higher'))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid empirical calibration residual quantile')
    return value, errors.tolist()


def _stress_range(log10_nominal, residual_q90, parameter, calibration_n):
    if parameter not in PARAMETERS:
        raise ValueError('Unsupported parameter stress range')
    values = (log10_nominal, residual_q90)
    if any(isinstance(v, bool) or not isinstance(v, (int, float, np.floating)) or not math.isfinite(v) for v in values):
        raise ValueError('Stress range values must be finite real numbers')
    if residual_q90 < 0 or isinstance(calibration_n, bool) or not isinstance(calibration_n, int) or calibration_n < 1:
        raise ValueError('Stress ranges require nonnegative residual width and positive calibration count')
    try:
        lo, nominal, upper_unbounded = (10. ** float(log10_nominal + delta)
                                        for delta in (-residual_q90, 0., residual_q90))
    except OverflowError as exc:
        raise ValueError('Nonfinite physical stress range') from exc
    if not all(math.isfinite(v) and v > 0 for v in (lo, nominal, upper_unbounded)):
        raise ValueError('Nonpositive or nonfinite physical stress range')
    if parameter == 'fraction_unbound' and nominal > 1.:
        raise ValueError('Nominal fraction unbound exceeds its physical bound')
    clipped = parameter == 'fraction_unbound' and upper_unbounded > 1.
    upper = min(1., upper_unbounded) if parameter == 'fraction_unbound' else upper_unbounded
    return {'lower': lo, 'central': nominal, 'upper': upper,
        'unit': PARAMETERS[parameter][1], 'log10_residual_q90': float(residual_q90),
        'calibration_n': calibration_n, 'quantile': QUANTILE, 'quantile_method': 'higher',
        'upper_clipped_to_one': clipped, 'upper_before_physical_bound': upper_unbounded,
        'degenerate': lo == upper, 'limited_calibration_count': calibration_n < 20,
        'renal_clearance_used_for_clipping': False, 'clinical_confidence_interval': False,
        'basis': 'empirical_absolute_log10_residuals_on_scaffold_held_out_calibration',
        'interpretation': 'Marginal engineering stress range; not a calibrated clinical prediction interval or a joint parameter distribution.'}


def _validate_split(dataset, split, exclusions):
    train, calibration = split['train_indices'], split['test_indices']
    if set(train) & set(calibration):
        raise ValueError('Training/calibration row leakage')
    if set(split['external_exclusions']) != set(exclusions):
        raise ValueError('Frozen cohort exclusion keys changed')
    expected = {name: base.normalize_structure(smiles) for name, smiles in exclusions.items()}
    for name, identity in expected.items():
        if any(split['external_exclusions'][name][key] != identity[key] for key in identity):
            raise ValueError('Cohort identity differs from frozen scaffold exclusion')
    protected_identities = {x['identity_smiles'] for x in expected.values()}
    protected_scaffolds = {x['scaffold_group'] for x in expected.values()}
    for index in train + calibration:
        row = dataset.rows[index]
        if row['identity_smiles'] in protected_identities or row['scaffold_group'] in protected_scaffolds:
            raise ValueError('External cohort labels would enter training/calibration')
    train_scaffolds = {dataset.rows[i]['scaffold_group'] for i in train}
    if train_scaffolds & {dataset.rows[i]['scaffold_group'] for i in calibration}:
        raise ValueError('Training/calibration scaffold leakage')
    return expected


def fit_ranged_prior(wheel, *, exclude_smiles: Mapping[str, str], output_dir: Path,
                     project_root: Path, n_jobs: int = 4):
    """Freeze three artifacts, or verify and reuse the identical completed fit.

    The inherited ``test_indices`` are explicitly re-labelled calibration here:
    they set the empirical 90% residual width and never tune RF hyperparameters.
    A partial output is retained and rejected; a different request never replaces
    existing models. Input may be the PKSmart wheel or its same-schema local CSV.
    """
    root = Path(project_root).resolve(strict=True)
    source = _inside(Path(wheel), root)
    output = _inside(Path(output_dir), root, exists=False)
    if not isinstance(exclude_smiles, Mapping) or not exclude_smiles:
        raise ValueError('A nonempty frozen cohort SMILES mapping is required')
    if any(not isinstance(name, str) or not name for name in exclude_smiles):
        raise ValueError('Each cohort identifier must be a nonempty string')
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 4:
        raise ValueError('n_jobs must be an integer between 1 and 4')
    exclusions = dict(sorted(exclude_smiles.items()))
    for smiles in exclusions.values():
        base.normalize_structure(smiles)
    code = [Path(__file__).resolve(), Path(base.__file__).resolve()]
    pins = {str(source.relative_to(root)): _sha(source),
            **{str(_inside(p, root).relative_to(root)): _sha(p) for p in code}}
    frozen = {'version': VERSION, 'source_pins': pins, 'exclude_smiles': exclusions,
        'seed': SEED, 'maximum_calibration_rows': CALIBRATION_LIMIT,
        'quantile': QUANTILE, 'quantile_method': 'higher', 'n_jobs': n_jobs,
        'forest': {'n_estimators': 200, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
        'requested_parameters': list(PARAMETERS), 'calibration_is_independent_test': False}
    fingerprint = _digest(frozen)
    if output.exists():
        bundle_path = output / 'predictions.json'
        if not bundle_path.is_file():
            raise FileExistsError('Incomplete existing range-prior output is retained; use a new directory')
        bundle = json.loads(bundle_path.read_text())
        if bundle.get('input_fingerprint') != fingerprint:
            raise ValueError('Existing range prior has different frozen inputs, sources or code')
        _verify(pins, root)
        for key in ('model', 'audit'):
            item = bundle['artifacts'][key]
            _verify({item['path']: item['sha256']}, root)
        audit = json.loads(_inside(bundle['artifacts']['audit']['path'], root).read_text())
        if _digest(bundle['predictions']) != audit['predictions_digest']:
            raise ValueError('Frozen cohort predictions changed')
        return {'predictions': bundle['predictions'], 'audit': audit,
            'artifacts': {**bundle['artifacts'], 'predictions': _ref(bundle_path, root)},
            'input_fingerprint': fingerprint, 'reused': True}
    dataset = base.load_parameter_dataset(source)
    split = base.make_scaffold_split(dataset, exclude_smiles=exclusions,
        train_limit=len(dataset.rows), test_limit=CALIBRATION_LIMIT, seed=SEED)
    identities = _validate_split(dataset, split, exclusions)
    learned = base.fit_parameter_pilot(dataset, split, n_jobs=n_jobs)
    calibration, widths, label_audit = {}, {}, {}
    for parameter, (endpoint, unit) in PARAMETERS.items():
        if endpoint not in learned.models:
            raise ValueError('No RF training labels for ' + parameter)
        labels = learned.audit['label_audit'][endpoint]
        if not set(labels['train_indices']) <= set(split['train_indices']) or not set(labels['test_indices']) <= set(split['test_indices']):
            raise ValueError('Endpoint label membership escaped frozen split')
        indices = labels['test_indices']
        if not indices:
            raise ValueError('No independent scaffold calibration labels for ' + parameter)
        features = base._features([dataset.rows[i]['identity_smiles'] for i in indices])
        predicted = learned.models[endpoint].predict(features)
        truth = [dataset.rows[i]['labels_log10'][endpoint] for i in indices]
        q90, residuals = _residual_quantile(truth, predicted)
        widths[parameter] = q90
        calibration[parameter] = {'endpoint': endpoint, 'unit': unit,
            'row_indices': indices, 'absolute_log10_residuals': residuals,
            'log10_residual_q90': q90, 'n': len(indices),
            'role': 'range calibration only; not independent test performance'}
        label_audit[parameter] = {'train_indices': labels['train_indices'],
            'calibration_indices': indices, 'train_count': labels['train_count'],
            'calibration_count': len(indices),
            'train_missing_indices': labels['train_missing_indices'],
            'calibration_missing_indices': labels['test_missing_indices']}
    names = list(exclusions)
    external_features = base._features([identities[name]['identity_smiles'] for name in names])
    log_predictions = {p: learned.models[ep].predict(external_features) for p, (ep, _) in PARAMETERS.items()}
    predictions = {}
    for i, name in enumerate(names):
        ranges = {p: _stress_range(float(log_predictions[p][i]), widths[p], p, calibration[p]['n']) for p in PARAMETERS}
        predictions[name] = {'input_smiles': exclusions[name], **identities[name],
            'predict_nominal': {p: r['central'] for p, r in ranges.items()}, 'ranges': ranges,
            'source_kind': 'predicted', 'eligible_as_measured_evidence': False,
            'training_structure_overlap': False, 'training_scaffold_overlap': False,
            'calibration_structure_overlap': False, 'calibration_scaffold_overlap': False,
            'cohort_labels_used': False}
    rows = deepcopy(split['rows'])
    for row in rows:
        if row['status'] == 'test': row['status'] = 'calibration'
        if row.get('reason') in {'external_test_identity', 'external_test_scaffold'}:
            row['reason'] = row['reason'].replace('external_test_', 'external_cohort_')
    audit = {'version': VERSION, 'input_fingerprint': fingerprint, 'frozen_input': frozen,
        'source': dataset.provenance, 'source_pins': pins,
        'cohort_count': len(exclusions), 'cohort_unique_connectivities': len({r['identity_smiles'] for r in identities.values()}),
        'external_exclusions': split['external_exclusions'], 'rows': rows,
        'train_indices': split['train_indices'], 'calibration_indices': split['test_indices'],
        'train_count': split['train_count'], 'calibration_count': split['test_count'],
        'train_scaffold_count': split['train_scaffold_count'], 'calibration_scaffold_count': split['test_scaffold_count'],
        'training_calibration_scaffold_overlap': [], 'selection_uses_labels': False,
        'label_audit': label_audit, 'calibration': calibration,
        'feature_specification': learned.audit['feature_specification'],
        'forest_parameters': learned.audit['forest_parameters'], 'software': learned.audit['software'],
        'hyperparameter_search': False, 'label_imputation': False, 'pretrained_weights_used': False,
        'base_helper_trained_endpoints': list(learned.models), 'retained_model_endpoints': [v[0] for v in PARAMETERS.values()],
        'calibration_is_independent_test': False, 'clinical_curves_used': False,
        'renal_clearance_used': False, 'predictions_digest': _digest(predictions),
        'range_interpretation': 'Empirical marginal stress ranges from held-out scaffold residuals; no clinical or conformal coverage guarantee.',
        'deduplication': split['deduplication'], 'acyclic_policy': split['acyclic_policy']}
    _verify(pins, root)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output/'parameter_models.joblib'
    joblib.dump({endpoint: learned.models[endpoint] for endpoint, _ in PARAMETERS.values()}, model_path, compress=3)
    audit['model'] = _ref(model_path, root)
    _write_new(output/'audit.json', audit)
    artifacts = {'model': audit['model'], 'audit': _ref(output/'audit.json', root)}
    bundle = {'version': VERSION, 'input_fingerprint': fingerprint, 'predictions': predictions, 'artifacts': artifacts}
    _write_new(output/'predictions.json', bundle)
    _verify(pins, root)
    return {'predictions': predictions, 'audit': audit,
        'artifacts': {**artifacts, 'predictions': _ref(output/'predictions.json', root)},
        'input_fingerprint': fingerprint, 'reused': False}
