"""Engineering exceptions cannot be relabelled completed unknown cases."""
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs import engineering_gate as g


def plan(reason):
    return {'status':'blocked','reasons':[reason],'sealed_observations_accessed':False,'fitting_performed':False}


@pytest.mark.parametrize('reason',[
    'ValueError:SOURCE_CHECKSUM_MISMATCH','KeyError:missing_parameter',
    'Blocked:VERIFIED_NATIVE_MIGRATION_UNAVAILABLE','Blocked:SOURCE_ADAPTER_BLOCKED:x',
    'Blocked:ABSOLUTE_MASS_DOSE_REQUIRED:mg/kg','Blocked:IMPLEMENTATION_CHANGED_DURING_PREPARATION',
    'Blocked:SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN'])
def test_engineering_problems_fail_closed(reason,monkeypatch):
    monkeypatch.setattr(g,'_context',lambda *a:({'case_id':'other','normalized_design':{},'protocol_fields':{}},{},[]))
    result=g.assess_blocked_plan(plan(reason))
    assert result['accepted'] is False and result['category']=='engineering_repair_required'


def test_exact_clinical_ambiguity_is_retained_unknown(monkeypatch):
    scenario={'case_id':'felodipine_pmc5642591_monotherapy_er5mg_bid_day7',
        'normalized_design':{'regimen':{'observation_origin_h':None}},
        'protocol_fields':{'dose':{'explicit_total_dose_count':None},'protocol_execution_ambiguities':['Last dose morning/evening unresolved']}}
    monkeypatch.setattr(g,'_context',lambda *a:(scenario,{},[{'path':'frozen','sha256':'fixed'}]))
    result=g.assess_blocked_plan(plan('Blocked:UNKNOWN_OBSERVATION_ORIGIN_REQUIRES_EXPLICIT_PROTOCOL'))
    assert result['accepted'] and result['scientific_applicability']=='unknown'
    scenario['normalized_design']['regimen']['observation_origin_h']=156
    assert not g.assess_blocked_plan(plan('Blocked:UNKNOWN_OBSERVATION_ORIGIN_REQUIRES_EXPLICIT_PROTOCOL'))['accepted']


def test_generic_no_simulation_reason_needs_full_exclusion_proof(monkeypatch):
    monkeypatch.setattr(g,'_context',lambda *a:({'case_id':'other','normalized_design':{},'protocol_fields':{}},{},[]))
    def failure(*a):raise ValueError('UNRESOLVED_ADAPTER_EXCLUSION:SOURCE_HAS_FEWER_EVENTS')
    monkeypatch.setattr(g,'_source_absence',failure)
    result=g.assess_blocked_plan(plan('Blocked:NO_COMPLETE_SOURCE_SIMULATION_MATCHES_FROZEN_CONTRACT'))
    assert not result['accepted'] and 'UNRESOLVED_ADAPTER_EXCLUSION' in result['reasons'][0]


def test_license_unknown_requires_reproducible_frozen_audit(monkeypatch):
    monkeypatch.setattr(g,'_context',lambda *a:({'case_id':'other','normalized_design':{},'protocol_fields':{}},{},[]))
    monkeypatch.setattr(g,'_license_gap',lambda *a:[{'path':'fixed_receipt','sha256':'fixed'}])
    result=g.assess_blocked_plan(plan('Blocked:CANDIDATE_LICENSE_NOT_RESOLVED'))
    assert result['accepted'] and result['category']=='original_source_license_unresolved'
    def corrupt(*a):raise ValueError('EVIDENCE_CHECKSUM_MISMATCH')
    monkeypatch.setattr(g,'_license_gap',corrupt)
    assert not g.assess_blocked_plan(plan('Blocked:CANDIDATE_LICENSE_NOT_RESOLVED'))['accepted']


def test_no_outcome_or_fitted_plan_may_pass():
    p=plan('Blocked:CANDIDATE_LICENSE_NOT_RESOLVED');p['fitting_performed']=True
    assert not g.assess_blocked_plan(p)['accepted']


@pytest.fixture
def route_case(monkeypatch):
    import json,tempfile
    from recommendation_execution_repairs import complete_candidate as c
    base=ROOT/'runtime/tmp';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='route-gate-',dir=base) as directory:
        parent=Path(directory);source=parent/'original.pksim5';source.write_bytes(b'only fixture')
        digest='fixed-original';simulations=[{'simulation_id':'one','source_blob_sha256':'blob1','compounds':['Voriconazole']},
                                          {'simulation_id':'two','source_blob_sha256':'blob2','compounds':['Fentanyl']}]
        certificate={'source':{'path':str(source.relative_to(ROOT)),'sha256':digest},'implementation':[],
          'simulations':[{'simulation_id':s['simulation_id'],'source_blob_sha256':s['source_blob_sha256'],
                         'target_compound_declared':'Voriconazole' in s['compounds'],
                         'target_administrations':[],'all_administrations_checked':1,'association_audit':{}}
                        for s in simulations]}
        cert_file=parent/'certificate.json';cert_file.write_text(json.dumps(certificate))
        monkeypatch.setattr(g,'VORICONAZOLE_ROUTE_CERTIFICATES',{digest:{'path':str(cert_file.relative_to(ROOT)),'sha256':g.sha(cert_file)}})
        def check(ref,root):
            if ref['path']==str(source.relative_to(ROOT)):return source
            path=ROOT/ref['path']
            if g.sha(path)!=ref['sha256']:raise ValueError('EVIDENCE_CHECKSUM_MISMATCH')
            return path
        monkeypatch.setattr(g,'_checked',check)
        monkeypatch.setattr(g.source_pkml,'source_simulation_metadata',lambda *a,**kw:{'simulations':simulations})
        monkeypatch.setattr(c,'pregnancy_source_policy',lambda *a,**kw:{})
        calls=[]
        def validate(cert,**kwargs):calls.append(kwargs);return {'accepted':True}
        monkeypatch.setattr(g.event_associations,'validate_route_absence_certificate',validate)
        p={'contract':{'route':'Intravenous'},'excluded_source_simulations':[
            {'simulation_id':'one','reason':'SOURCE_EVENT_COMPOUND_ASSOCIATION_INCOMPLETE'},
            {'simulation_id':'two','reason':'COADMINISTERED_SOURCE_EVENT_NOT_ALLOWED'}]}
        scenario={'compound':'voriconazole','normalized_design':{'route':'intravenous','source_route':'IV','dose':{'route':'IV'}}}
        candidate={'catalog_provenance':{'source_path':str(source.relative_to(ROOT)),'source_sha256':digest}}
        yield p,scenario,candidate,calls,certificate,cert_file


def test_full_route_proof_overrides_old_parser_and_coadmin_reasons(route_case):
    p,s,c,calls,certificate,path=route_case
    refs,facts=g._source_absence(p,s,c,ROOT)
    assert len(facts)==2 and all(x['scientific_applicability']=='unknown' for x in facts)
    assert calls[0]['compound']=='Voriconazole' and calls[0]['required_route']=='Intravenous'
    assert calls[0]['source_sha256']==c['catalog_provenance']['source_sha256']
    assert any(r['path']==str(path.relative_to(ROOT)) for r in refs)


def test_route_plan_cannot_replace_the_frozen_clinical_route(route_case):
    p,s,c,*_=route_case;s['normalized_design']['dose']['route']='Oral'
    with pytest.raises(ValueError,match='FROZEN_PROTOCOL'):g._source_absence(p,s,c,ROOT)


def test_unresolved_original_association_remains_engineering_blocked(route_case,monkeypatch):
    p,s,c,*_=route_case
    def unresolved(*a,**kw):raise ValueError('EVENT_ASSOCIATION_UNIQUE_DRUG_OUTGOING_TRANSPORT_REQUIRED')
    monkeypatch.setattr(g.event_associations,'validate_route_absence_certificate',unresolved)
    with pytest.raises(ValueError,match='EVENT_ASSOCIATION'):g._source_absence(p,s,c,ROOT)


def test_route_certificate_cannot_omit_an_original_simulation(route_case):
    import json
    p,s,c,_,certificate,path=route_case;certificate['simulations'].pop();path.write_text(json.dumps(certificate))
    g.VORICONAZOLE_ROUTE_CERTIFICATES[c['catalog_provenance']['source_sha256']]['sha256']=g.sha(path)
    with pytest.raises(ValueError,match='INVENTORY_DIFFERS'):g._source_absence(p,s,c,ROOT)


def test_route_certificate_cannot_change_during_validation(route_case,monkeypatch):
    p,s,c,_,_,path=route_case
    def mutate(*a,**kw):path.write_text('{"changed":true}');return {'accepted':True}
    monkeypatch.setattr(g.event_associations,'validate_route_absence_certificate',mutate)
    with pytest.raises(ValueError,match='CHECKSUM_MISMATCH'):g._source_absence(p,s,c,ROOT)


@pytest.mark.parametrize('case,salt',[
    ('dextromethorphan_yamazaki2016_day1_30mg','dextromethorphan hydrobromide; hydration state not specified'),
    ('paroxetine_elshafeey2021_market_tablet_25mg','salt/hydration not stated in clinical protocol')])
def test_explicit_product_ambiguity_cannot_hide_a_resolved_dose(case,salt):
    scenario={'case_id':case,'normalized_design':{'dose':{'salt':salt}},
              'protocol_fields':{'dose':{'salt':salt}}}
    reasons=['SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN']
    assert g._clinical_gap(scenario,reasons)
    scenario['normalized_design']['dose']['active_moiety_dose_mg']=20
    assert g._clinical_gap(scenario,reasons) is None
    scenario['normalized_design']['dose'].pop('active_moiety_dose_mg')
    scenario['protocol_fields']['salt_conversion']={'factor':0.8}
    assert g._clinical_gap(scenario,reasons) is None


@pytest.fixture
def joint_source(tmp_path):
    import json
    source=tmp_path/'model.json'
    document={'Version':80,'Compounds':[{'Name':'Ethinylestradiol'}],
              'Simulations':[{'Name':'original','Compounds':[{'Name':'Ethinylestradiol'}]}]}
    dose=[{'compound':'ethinylestradiol','dose_mg':0.03},{'compound':'levonorgestrel','dose_mg':0.15}]
    scenario={'case_id':'ethinylestradiol_pmc8416862_nct02845219_microgynon_alone_day8',
              'compound':'ethinylestradiol','normalized_design':{'dose':{'coadministered_drugs':dose},
              'regimen':{'coadministered_drugs':dose}},'protocol_fields':{'dose':{'coadministered_drugs':dose}}}
    def write():
        source.write_text(json.dumps(document))
        return {'catalog_provenance':{'source_path':'model.json','source_sha256':g.sha(source)}}
    return scenario,document,write,tmp_path


def test_joint_regimen_requires_an_actual_missing_original_model(joint_source):
    scenario,document,write,root=joint_source
    refs,facts=g._missing_joint_drug_model(scenario,write(),root)
    assert facts['missing_required_parent_models']==['levonorgestrel']
    assert facts['scientific_applicability']=='unknown'
    # A globally declared partner, even outside the selected simulation, needs
    # adapter repair and may not be dismissed as an absent chemical model.
    document['Compounds'].append({'Name':'Levonorgestrel'})
    with pytest.raises(ValueError,match='ADAPTER_MUST_BE_REPAIRED'):
        g._missing_joint_drug_model(scenario,write(),root)


@pytest.mark.parametrize('change', ['empty_simulation','undeclared_dependency','empty_inventory','changed_regimen'])
def test_incomplete_joint_inventory_or_changed_protocol_is_not_absence(joint_source,change):
    scenario,document,write,root=joint_source
    if change=='empty_simulation':document['Simulations'][0]['Compounds']=[]
    elif change=='undeclared_dependency':document['Simulations'][0]['Compounds'].append({'Name':'Other'})
    elif change=='empty_inventory':document['Compounds']=[]
    else:scenario['normalized_design']['regimen']={'coadministered_drugs':[]}
    with pytest.raises(ValueError):g._missing_joint_drug_model(scenario,write(),root)


def test_changed_joint_source_checksum_is_rejected(joint_source):
    scenario,document,write,root=joint_source
    candidate=write();document['Compounds'].append({'Name':'Levonorgestrel'});write()
    with pytest.raises(ValueError,match='CHECKSUM_MISMATCH'):
        g._missing_joint_drug_model(scenario,candidate,root)
