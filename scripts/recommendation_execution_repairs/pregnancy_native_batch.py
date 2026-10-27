"""Native serialization of all saved simulations in the pregnancy inventory.

Use the official PK-Sim reader plus OSPSuite's ModelCoreSimulation mapper and
serializer. Run one process at a time, never an ODE, and retain every outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from recommendation_execution_repairs import source_pkml
from recommendation_execution_repairs import model_semantics
from recommendation_execution_repairs.pregnancy_source_assets import inside, sha_file, write_json
from pbpk_kb.blueprints.design_runtime import run_bounded_r

INVENTORY = ROOT / 'artifacts/pbpk-model-assets/execution_repairs_20261001/pregnancy_exports/pregnancy_export_manifest.json'
WORKER = ROOT / 'src/pbpk_kb/model_runtime/R/recommendation_saved_model_core_export.R'
OUTPUT = ROOT / 'runtime/deployment/pregnancy_native_export_20261001'


def run_all(output=OUTPUT, *, max_simulations=None):
    output = inside(output)
    if output.exists():
        raise FileExistsError('Use a new native batch directory')
    inventory = json.loads(INVENTORY.read_text())
    inputs = inventory['records']
    if len(inputs) != 16 or inventory['counts']['blocked']:
        raise ValueError('Expected the complete 16-simulation pregnancy inventory')
    if max_simulations is not None:
        if not isinstance(max_simulations, int) or not 1 <= max_simulations <= len(inputs):
            raise ValueError('Invalid bounded engineering probe count')
        inputs = inputs[:max_simulations]
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise OSError('At least 3 GiB free required for bounded source copies and native outputs')
    output.mkdir(parents=True)
    code = {str(p.relative_to(ROOT)): sha_file(p) for p in [Path(__file__), WORKER, Path(source_pkml.__file__), Path(model_semantics.__file__)]}
    # Keep the exact unfrozen engineering implementation with this attempt.
    provenance = output / 'implementation'; provenance.mkdir()
    for path in code:
        copied = provenance / path
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, copied)
    rows = []
    for source_row in inputs:
        simulation = source_row['simulation']
        entry = output / source_row['compound'] / hashlib.sha256(simulation['simulation_id'].encode()).hexdigest()[:16]
        row = {k: source_row[k] for k in ['candidate_id', 'compound', 'simulation', 'repository_commit', 'structure']}
        try:
            if any(sha_file(ROOT / name) != digest for name, digest in code.items()):
                raise ValueError('Native batch implementation changed; stop before using mixed code')
            entry.mkdir(parents=True)
            original = source_row['extraction']['source']
            prepared = source_pkml.prepare_native_export_copy(original['path'], original['sha256'],
                                                             simulation['simulation_id'], entry / 'isolated_source')
            worker_dir = entry / 'worker'; worker_dir.mkdir()
            target = worker_dir / 'preserved_native_model.pkml'
            process = run_bounded_r(script=WORKER,
                args=[str(ROOT / prepared['working_copy']['path']), simulation['simulation_id'], simulation['name'], str(target)],
                output_dir=worker_dir, project_root=ROOT, timeout_seconds=180, cpu_threads=1,
                max_output_bytes=128 * 1024**2)
            row.update(working_copy=prepared, process=process)
            if process['status'] != 'completed' or not target.exists():
                raise ValueError('Native source serialization failed; see bounded worker log')
            native_audit = json.loads(Path(str(target) + '.native.json').read_text())
            if not native_audit.get('same_model_object') or not native_audit.get('same_settings_object') or native_audit.get('model_rebuilt') is not False or native_audit.get('solver_calls') != 0:
                raise ValueError('Native source-object preservation contract failed')
            document = ET.parse(target).getroot()
            if document.tag != 'SimulationTransfer' or document.find('./Simulation/Model') is None:
                raise ValueError('Complete OSPSuite simulation transfer missing')
            if any(n.tag in {'DataRepository', 'ObservedData', 'IndividualResults', 'SimulationResults'} for n in document.iter()):
                raise ValueError('Outcome payload in native export')
            model = document.find('./Simulation/Model')
            original_xml = source_row['extraction']['pkml']
            original_xml_path = inside(ROOT / original_xml['path'])
            if sha_file(original_xml_path) != original_xml['sha256']:
                raise ValueError('Original extracted XML changed')
            saved_document = ET.parse(original_xml_path).getroot()
            model_check = model_semantics.compare_saved_model_xml(saved_document.find('.//Model'), model)
            settings_check = model_semantics.compare_saved_settings_xml(saved_document.find('.//SimulationSettings'), document.find('.//SimulationSettings'))
            row.update(model_semantics=model_check, settings_semantics=settings_check)
            if not model_check['accepted'] or not settings_check['accepted']:
                raise ValueError('Complete model or settings semantics changed; differences retained in audit')
            markers = sorted({n.get('name') for n in model.iter('Container')} & {'Fetus', 'PlacentaMaternal', 'PlacentaFetal', 'AmnioticFluid'})
            if markers != source_row['structure']['pregnancy_markers']:
                raise ValueError('Pregnancy structures changed during native export')
            row.update(status='native_serialization_passed', pkml={'path': str(target.relative_to(ROOT)), 'sha256': sha_file(target)},
                       native_audit=native_audit, pregnancy_markers_preserved=markers,
                       native_import_and_solver_validation_pending=True,
                       population_covariates_applied=False, scientific_applicability_inferred=False)
        except Exception as exc:
            row.update(status='failed', reason=type(exc).__name__ + ':' + str(exc))
        rows.append(row)
        if entry.exists():
            write_json(entry / 'serialization_result.json', row)
        progress = {'complete': len(rows), 'requested': len(inputs), 'catalog_inventory': len(inventory['records']),
                    'passed': sum(x['status'] == 'native_serialization_passed' for x in rows),
                    'failed': sum(x['status'] == 'failed' for x in rows), 'last': {'compound': row['compound'], 'simulation': simulation['name'], 'status': row['status']}}
        temporary = output / 'progress.new.json'
        write_json(temporary, progress); temporary.replace(output / 'progress.json')
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if row['status'] == 'failed':
            # Preserve this real failure, fix the concrete issue, and resume
            # in a new attempt; do not repeat the same failing native call 16x.
            break
    result = {'schema': 'pregnancy-native-serialization-batch-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'input_inventory': {'path': str(INVENTORY.relative_to(ROOT)), 'sha256': sha_file(INVENTORY)},
        'implementation_sha256': code, 'records': rows,
        'summary': {'catalog_simulations': 16, 'attempted': len(rows),
                    'passed': sum(x['status'] == 'native_serialization_passed' for x in rows),
                    'failed': sum(x['status'] == 'failed' for x in rows)},
        'solver_calls': 0, 'frozen_experiment_modified': False, 'original_sources_modified': False,
        'limitations': ['Serialization alone is not a native solver or scientific prediction validation.',
                       'Original population distributions are retained separately, not sampled in this operation.']}
    write_json(output / 'native_batch_manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--max-simulations', type=int)
    args = parser.parse_args()
    result = run_all(args.output, max_simulations=args.max_simulations)
    print(json.dumps(result['summary']))
    raise SystemExit(0 if result['summary']['failed'] == 0 else 1)
