from copy import deepcopy
import pytest
from scripts.recommendation_execution_repairs.application_path_schema import normalize_application_paths


def test_only_leading_application_path_changes_and_all_source_values_are_retained():
    source={'Version':80,'Simulations':[{'Name':'s','Parameters':[
        {'Path':'Applications|A|Delay','Value':0.010002129,'Unit':'', 'ValueOrigin':{'Source':'Publication'}},
        {'Path':'Body|Applications|Unrelated','Value':5}]}], 'Compounds':[{'Name':'D'}]}
    before=deepcopy(source)
    mapped,changes=normalize_application_paths(source)
    assert source==before and len(changes)==1
    assert changes[0]['before']==before['Simulations'][0]['Parameters'][0]
    expected=deepcopy(before);expected['Simulations'][0]['Parameters'][0]['Path']='Events|A|Delay'
    assert mapped==expected
    assert normalize_application_paths(mapped)==(mapped,[])


@pytest.mark.parametrize('source',[
    {'Version':79},
    {'Version':80,'Simulations':[{'Name':'s'},{'Name':'s'}]},
    {'Version':80,'Simulations':[{'Name':'s','Parameters':[{'Path':'Applications|A'},{'Path':'Events|A'}]}]},
])
def test_ambiguous_or_wrong_version_is_rejected(source):
    with pytest.raises(ValueError):normalize_application_paths(source)
