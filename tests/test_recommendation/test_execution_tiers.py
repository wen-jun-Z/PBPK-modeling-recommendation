"""Declared tiers bind source identity and preserve clinical task boundaries."""
import json
import inspect
import pytest
from scripts import recommendation_execution_tiers as tiers


def test_mapping_uses_actual_endpoints_dose_route_and_regimen():
    s = {'scenario_id': 'q', 'compound': 'drug', 'sealed_references': [{'observed_auc': 987654321}],
         'normalized_design': {'species':'human','dose': {'value': 250, 'unit': 'mg', 'basis': 'label_salt', 'formulation': 'tablet'},
            'route': 'oral', 'regimen': {'dose_times_h': [0,24,48], 'infusion_duration_h': None},
            'population': {'sex': 'female', 'healthy': True},
            'endpoint_contracts': [{'endpoint': 'AUC0_24', 'source_cell': 'hidden_source'}, {'paper_target': 'Cmax'}]}}
    task = tiers.clinical_task(s)
    assert task['required_outputs'] == ['AUC0_24','Cmax']
    assert task['scenario']['dose_value'] == 250 and task['scenario']['frequency'] == 'multiple_dose_schedule'
    assert task['scenario']['end_time_value'] == 48 and task['scenario']['sex'] == 'female'
    assert '987654321' not in json.dumps(task) and 'source_cell' not in json.dumps(task)
    q = tiers.build_execution_requirement_graph(task)
    assert {n.label for n in q.nodes if n.node_type == 'Endpoint'} == {'auc','cmax'}
    s['normalized_design']['species']='rat'
    with pytest.raises(ValueError,match='human'):
        tiers.clinical_task(s)


def test_same_name_wrong_hash_is_unknown_not_rejected_or_rebound():
    c = {'candidate_id': 'c', 'family_name': 'drug', 'provenance': {'source_sha256': 'a'*64}}
    records = [({'model_family':'drug', 'project_sha256':'b'*64}, {'file':'f','row_1based':1})]
    assert tiers.metadata_match(c, records) == (None,None,'metadata_missing')
    records[0][0]['project_sha256']='a'*64
    model,ref,gap=tiers.metadata_match(c,records)
    assert model['model_version_id']=='c' and gap is None
    assert 'model_version_id' not in records[0][0]
    assert tiers.metadata_match(c,records*2)[2]=='metadata_ambiguous'


def test_rule_rejection_name_cannot_be_scientific_unusable():
    assert tiers.CLASS_NAMES == {0:'direct_reuse',2:'reference_only',3:'not_cleared_by_existing_rules'}
    model=tiers.build_model_graph({'model_version_id':'c','compounds':['drug'],
        'license':{'status':'allowed'},'pkml_parse_status':'registered_not_executable',
        'execution_probe_status':'not_run','reusable_candidate':False})
    query=tiers.build_execution_requirement_graph({'task_id':'q','active_moieties':['drug'],
        'required_outputs':['Cmax'],'scenario':{'route':'oral'}})
    assessment=tiers.assess_candidate_use(query,model)
    assert tiers.deterministic_use_class(assessment)==3
    assert any('ExecutionStatus' in x for x in assessment.direct_reuse_gate_failures)


def test_verifier_source_rehash_is_explicit_and_defaults_to_on():
    signature=inspect.signature(tiers.verify_declared_tiers)
    assert signature.parameters['check_original_sources'].default is True
    signature.bind(check_original_sources=False)
    a={'enzymes':['Cyp3a4','CYP3A4','CYP2D6']}
    b={'enzymes':['CYP2D6','CYP3A4','Cyp3a4']}
    assert tiers.stable_model_metadata(a)==tiers.stable_model_metadata(b)=={'enzymes':['CYP2D6','CYP3A4']}
    assert len(a['enzymes'])==3
