import copy
import json

import pytest

from pbpk_kb.recommendation.pbpk_match.data import (
    OUTPUT, fields_for, fitting_indices, node_specs, normalize, scoped,
)


def query():
    return {"query_id":"q", "compound_id":"drug", "compounds":["Drug A"],
            "task_type":"plasma_concentration_time_prediction",
            "text":"This text can contain unsafe outcome metadata", "features":{
                "routes":["Oral"],"enzymes":["CYP3A4"],"endpoints":["auc"]},
            "variants":{"reordered":{"text":"Same fields, different sentence order"},
                        "missing_evidence":{"features":{"routes":["Oral"],"endpoints":["auc"]}}}}


def test_outcome_and_identity_metadata_cannot_change_model_features():
    first=query();second=copy.deepcopy(first)
    second.update(run_label_grade=3,legacy_split="train",candidate_id="known_good",text="judge says Y")
    second['features']['validation_result']=['perfect'];second['graph']={'nodes':[{'node_type':'ValidationMetric','label':'1.0'}]}
    assert node_specs(first)==node_specs(second)


def test_sentence_reordering_invariant_but_removed_mechanisms_visible():
    q=query()
    assert node_specs(q)==node_specs(q,'reordered')
    assert node_specs(q)!=node_specs(q,'missing_evidence')
    assert fields_for(q,'missing_evidence')['enzymes']==[]
    assert fields_for(q,'missing_evidence')['routes']==['oral']


def test_nodes_deduplicate_and_field_order_does_not_matter():
    first=query();second=copy.deepcopy(first)
    second['features']['enzymes']=['cyp3a4','CYP3A4',' cyp3a4 ']
    assert node_specs(first)==node_specs(second)
    assert normalize([' A  B ','a b'])==['a b']


def test_fit_excludes_same_source_as_outer_and_locked_test():
    def rec(name,h):return {'query_id':name,'provenance':{'evidence':[{'evidence_text_sha256':h}]}}
    d={'query_records':[rec('train','same'),rec('outer','same'),rec('locked_overlap','different'),rec('safe','safe')],
       'snapshot':{'exact_test_source_overlap_excluded_from_fitting':['locked_overlap']}}
    assert fitting_indices(d,[0,2,3],[1])==[3]


def test_frozen_development_export_has_no_locked_test_labels():
    path=OUTPUT/'development_snapshot.json'
    if not path.exists():pytest.skip('Real development snapshot has not yet been prepared')
    d=json.loads(path.read_text())
    development=set(d['development_query_ids']);test=set(d['test_query_ids_excluded'])
    assert len(development)==43 and len(test)==102 and not development & test
    assert all(r['query_id'] in development for r in d['labels'])
    assert len(set(d['fold_by_compound'].values()))==3


def test_paths_cannot_escape_project():
    with pytest.raises(ValueError):scoped('/tmp/pbpk-outside-project')
