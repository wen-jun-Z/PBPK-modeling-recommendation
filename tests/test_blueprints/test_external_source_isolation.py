import pytest
from pbpk_kb.blueprints.external_source_isolation import audit_design_inputs


@pytest.fixture
def policy():
    return {'denied_paths':['raw/final_article.xml'], 'denied_hashes':['a'*64],
        'denied_identifiers':[{'kind':'doi','value':'10.1000/final-study'}, {'kind':'trial','value':'nct01234567'}],
        'curated_metadata_sources':{'curated/protocol.json':'b'*64},
        'protocols':[{'study_id':'study_1'}]}


@pytest.mark.parametrize('value', [
    {'source_id':'renamed/article.xml','source_checksum':'a'*64},
    {'provenance':{'source_id':'raw/final_article.xml'}},
    {'evidence':'See https://doi.org/10.1000/final-study'},
    {'parameter_prior':[{'trial':'NCT01234567'}]},
])
def test_final_study_is_rejected_across_aliases_and_nested_sources(value,policy):
    with pytest.raises(ValueError,match='External study entered'):
        audit_design_inputs(value,policy)


def test_exact_curated_protocol_provenance_is_distinct_from_final_outcomes(policy):
    reference={'source_id':'curated/protocol.json','source_checksum':'b'*64,
               'source_location':'Methods NCT01234567; dose only'}
    assert audit_design_inputs(reference,policy,allowed_metadata_sources=['curated/protocol.json'])['accepted']
    with pytest.raises(ValueError):audit_design_inputs(reference,policy)
    reference['hidden_evidence']='NCT01234567'
    with pytest.raises(ValueError):
        audit_design_inputs(reference,policy,allowed_metadata_sources=['curated/protocol.json'])


def test_denylist_declarations_do_not_count_as_used_evidence(policy):
    assert audit_design_inputs({'forbidden_source_ids':['raw/final_article.xml'],
                                'source_id':'source/development.json'},policy)['accepted']
