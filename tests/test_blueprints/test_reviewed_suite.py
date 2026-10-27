"""Queue boundaries, planned denominators and cross-budget holdout isolation."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('reviewed_suite_test',ROOT/'scripts/run_blueprint_reviewed_suite.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def cases():
    return [{'case_id':name,'compound_id':name,'tier':'reconstruction','parameter_count':2,
             'external_scenario_count':1,'bundle_path':name+'.json','bundle_sha256':'a'*64}
            for name in ('caffeine','fluconazole','alprazolam')]


def test_plan_preserves_budget_dimension_and_seed_denominators():
    tasks=cli.make_plan(cases())
    assert len(tasks)==12
    assert sum(len(t['jobs']) for t in tasks)==171
    for name,expected in [('main_5000',81),('budget_500',36),('budget_2000',36),('matched_1',18)]:
        assert sum(len(t['jobs']) for t in tasks if t['block_id']==name)==expected
    for task in tasks:
        if task['block_id']=='matched_1':
            assert task['matched_count']==1 and task['methods']==['M5','M6']
        else:assert task['matched_count'] is None


@pytest.mark.parametrize('identifier',['../outside','a/b','','a.b'])
def test_plan_rejects_unsafe_output_names(identifier):
    rows=cases();rows[0]['case_id']=identifier
    with pytest.raises(ValueError):cli.make_plan(rows)


def test_plan_rejects_unqualified_independent_tier_and_unavailable_matched_dimension():
    rows=cases();rows[0]['tier']='main_holdout'
    with pytest.raises(ValueError,match='unqualified'):cli.make_plan(rows)
    rows=cases();rows[0]['parameter_count']=0
    with pytest.raises(ValueError,match='dimension'):cli.make_plan(rows)


def test_failed_and_pending_jobs_remain_in_results_without_pooled_scores():
    tasks=cli.make_plan(cases())
    tasks[0].update(status='failed',reason='native adapter failure')
    report=cli.collect_results(tasks)
    assert len(report['rows'])==171
    assert report['method_status_counts']=={'not_completed':171}
    assert report['rows'][0]['reason']=='native adapter failure'
    assert report['independent_drug_count']==0


def test_all_budget_and_dimension_blocks_lock_before_any_external_scoring(tmp_path,monkeypatch):
    root=tmp_path;out=root/'run'
    monkeypatch.setattr(cli,'ROOT',root)
    def inside(path,*,exists=True):
        result=(root/path).resolve(strict=exists)
        if not result.is_relative_to(root) or result==root:raise ValueError('outside')
        return result
    monkeypatch.setattr(cli,'inside',inside)
    rows=cases()
    for row in rows:
        file=root/row['bundle_path'];file.write_text(json.dumps({'case_id':row['case_id']}))
        row['bundle_sha256']=cli.file_sha(file)
    manifest=root/'inputs.json';manifest.write_text(json.dumps({'cases':rows}))
    proposals=root/'proposals';proposals.mkdir();(proposals/'manifest.json').write_text('{}')
    monkeypatch.setattr(cli,'validate_inputs',lambda *args:({'cases':rows},rows))
    for name in ['scripts/run_blueprint_reviewed_suite.py','scripts/run_blueprint_cohort.py',
                 'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
                 'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R']:
        file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# fixture\n')
    (root/'src/pbpk_kb/blueprints').mkdir()
    script=root/'scripts/run_blueprint_experiments.py'
    script.write_text('''import sys,json
from pathlib import Path
args=sys.argv
out=Path(args[args.index('--output')+1]);out.mkdir(parents=True)
if out.name=='main_5000__caffeine':
 (out/'progress.json').write_text(json.dumps({'error':'explicit failure'}))
 raise SystemExit(1)
(out/'completion.json').write_text(json.dumps({'status':'completed'}))
(out/'external_prediction_locks.json').write_text(json.dumps({'locks':[]}))
''')
    scored=[]
    monkeypatch.setattr(cli.single,'load_case_bundle',lambda path:({'case_id':Path(path).stem},[{}]))
    def score(case,external,records,directory,update,*,device):
        barrier=json.loads((out/'prediction_barrier.json').read_text())
        assert len(barrier['task_states'])==12 and barrier['all_tasks_terminal']
        assert {r['status'] for r in barrier['task_states']}=={'failed','predictions_locked'}
        assert len(barrier['lock_sha256'])==11
        scored.append(directory.name)
    monkeypatch.setattr(cli.single,'score_external',score)
    original_sleep=time.sleep
    monkeypatch.setattr(cli.time,'sleep',lambda duration:original_sleep(.02))
    assert cli.main(['--manifest',str(manifest),'--qwen-proposals',str(proposals),'--output',str(out),
                     '--workers','2','--cpu-threads','1','--scenario-workers','1'])==0
    assert len(scored)==11
    results=json.loads((out/'results.json').read_text())
    assert len(results['rows'])==171
    assert results['rows'][0]['method_status']=='not_completed'
