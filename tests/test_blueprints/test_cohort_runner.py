"""Queue isolation and global holdout barrier, using tiny fake child processes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cohort_cli',ROOT/'scripts/run_blueprint_cohort.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    cases=[]
    for name in ('first','second','third'):
        path=tmp_path/(name+'.json')
        path.write_text(json.dumps({'case':{'case_id':name,'compound_id':name,'tier':'reconstruction'},
                                    'external_scenarios':[{'scenario_id':'external'}]}))
        cases.append({'case_id':name,'compound_id':name,'tier':'reconstruction','status':'ready',
                      'bundle_path':path.name,'bundle_sha256':cli.file_sha(path)})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':cases}))
    return manifest,cases


def test_mutated_bundle_rejected_before_any_child_starts(cohort):
    manifest,cases=cohort
    (manifest.parent/cases[0]['bundle_path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):cli.load_manifest(manifest)


def test_failed_drug_does_not_prevent_later_drug_and_scoring_waits_for_all(cohort,monkeypatch):
    manifest,cases=cohort;root=manifest.parent;out=root/'run'
    script=root/'scripts/run_blueprint_experiments.py';script.parent.mkdir()
    script.write_text('''import sys,json
from pathlib import Path
arguments=sys.argv
out=Path(arguments[arguments.index('--output')+1]);out.mkdir(parents=True)
case=json.loads(Path(arguments[arguments.index('--case-file')+1]).read_text())['case']
if case['case_id']=='first':
 (out/'progress.json').write_text(json.dumps({'status':'failed','error':'unsupported drug process'}))
 raise SystemExit(1)
(out/'completion.json').write_text(json.dumps({'status':'completed'}))
(out/'external_prediction_locks.json').write_text(json.dumps({'locks':[]}))
''')
    for name in ['scripts/run_blueprint_cohort.py','src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
                 'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R']:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen fixture\n')
    (root/'src/pbpk_kb/blueprints').mkdir()
    def load_case(path):
        payload=json.loads((root/path).read_text());return payload['case'],payload['external_scenarios']
    scored=[]
    def score(case,external,records,directory,update,*,device):
        barrier=json.loads((out/'cohort_prediction_barrier.json').read_text())
        assert barrier['all_eligible_drug_queues_terminal']
        assert {r['status'] for r in barrier['states']}=={'failed','predictions_locked'}
        scored.append(case['case_id'])
        (directory/'external_results.json').write_text(json.dumps({'evaluations':[]}))
    monkeypatch.setattr(cli,'single_runner',lambda:SimpleNamespace(load_case_bundle=load_case,score_external=score))
    original_sleep=time.sleep
    monkeypatch.setattr(cli.time,'sleep',lambda duration:original_sleep(.02))
    assert cli.main(['--input-manifest',str(manifest),'--output',str(out),'--workers','1',
                     '--cpu-threads','1','--scenario-workers','1','--device','cpu'])==0
    queue=json.loads((out/'queue.json').read_text())['cases']
    assert [r['status'] for r in queue]==['failed','evaluated','evaluated']
    assert scored==['second','third']
    assert len(json.loads((out/'configuration.json').read_text())['jobs'])==81
    barrier=(out/'cohort_prediction_barrier.json').read_bytes()
    # Simulate a completed supervisor identity; resume must not launch another
    # attempt or replace the original pre-scoring time record.
    (out/'runner.json').write_text(json.dumps({'pid':2147483647,'start_ticks':'0'}))
    assert cli.main(['--input-manifest',str(manifest),'--output',str(out),'--workers','1',
                     '--cpu-threads','1','--scenario-workers','1','--device','cpu','--resume'])==0
    assert (out/'cohort_prediction_barrier.json').read_bytes()==barrier
    assert all(row['attempt']==1 for row in json.loads((out/'queue.json').read_text())['cases'])
    assert scored==['second','third']
