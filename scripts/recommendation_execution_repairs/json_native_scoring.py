"""Independent pre-observation checks for source-specific bridge semantics."""
from pathlib import Path
import xml.etree.ElementTree as ET


def _require(ok, message):
    if not ok:
        raise ValueError('JSON native scoring: ' + message)


def verified_inspection(f, directory, plan, registered, global_files, prediction_files):
    from . import native_pkml
    def locked(path):
        path=f.path(path)
        _require(path in global_files and path in prediction_files, 'inspection evidence missing from a lock')
        return path
    def checked(ref):return locked(f.checked(ref))
    base=directory/'inspect'
    paths={name:locked(base/name) for name in ('request.json','native_result.json','result.json','process_manifest.json','artifact_lock.json')}
    request,raw,result,process,artifact=(f.read(paths[n]) for n in paths)
    native_pkml.validate_request(request)
    _require(all(f.path(request[key])==base/name for key,name in
                 (('result_path','native_result.json'),('before_pkml_path','before.pkml'),('after_pkml_path','after.pkml'))),
             'inspection destinations differ from this execution')
    source=checked(registered['pkml'])
    _require(f.path(request['pkml_path'])==source and request['pkml_sha256']==f.sha(source)
             and raw.get('request_sha256')==f.sha(paths['request.json'])
             and raw.get('pkml_sha256')==f.sha(source) and result.get('source_sha256')==f.sha(source),
             'inspection is not bound to the actual native input')
    _require(all(r.get('version')==native_pkml.VERSION for r in (request,raw,result,artifact))
             and all(r.get('action')=='inspect' for r in (request,raw,result))
             and result.get('status')=='inspected' and result.get('native')==raw
             and type(raw.get('solver_calls')) is int and raw['solver_calls']==0
             and request.get('patches')==[] and raw.get('parameter_changes')==[]
             and all(r.get(k) is False for r in (raw,result) for k in ('sealed_observations_accessed','fitting_performed')),
             'inspection scope/results differ')
    _require(artifact.get('source')==registered['pkml'] and artifact.get('prediction_accepted') is False,
             'inspection artifact source differs')
    files={checked({'path':p,'sha256':h}) for p,h in artifact['files'].items()}
    _require({paths[n] for n in ('request.json','native_result.json','result.json','process_manifest.json')}<=files
             and all(p.is_relative_to(base) for p in files), 'inspection artifact set differs')
    worker=native_pkml.WORKER;adapter='scripts/recommendation_execution_repairs/native_pkml.py';code=plan['implementation_sha256']
    _require(result.get('worker_sha256')==code.get(worker) and result.get('python_sha256')==code.get(adapter)
             and code.get(worker) and code.get(adapter) and process.get('script_sha256')==code[worker]
             and f.path(process['script'])==f.path(worker) and f.path(result['process_manifest'])==paths['process_manifest.json']
             and process.get('status')=='completed' and type(process.get('returncode')) is int and process['returncode']==0
             and process.get('runtime_unchanged') is True and process.get('registry_modified') is False,
             'inspection implementation or runtime differs')
    command=process.get('command',[])
    _require(len(command)==5 and command[1]=='--vanilla' and f.path(command[2])==f.path(worker)
             and f.path(command[3])==paths['request.json'] and command[4]==f.sha(paths['request.json']),
             'inspection process did not receive the frozen request')
    _require(isinstance(raw.get('inventory'),dict), 'inspection inventory absent')
    return raw['inventory']


def verified_metadata(f, plan):
    """Re-derive source event IDs even when no co-drug isolation is requested."""
    from . import event_associations
    metadata=plan['source_metadata']
    if plan.get('source_event_association'):
        source=plan.get('json_native_source_proof',{}).get('native_source') or {
            'path':plan['source_path'],'sha256':plan['source_sha256']}
        proof=event_associations.enrich_source_metadata(source['path'],source['sha256'],
            plan['source_simulation_id'],plan['source_model_blob_sha256'],metadata=metadata,project_root=f.root)
        _require(proof==plan['source_event_association'], 'source event association differs')
        metadata=proof['enriched_metadata']
    return metadata


def verified_native_observer_document(f, directory, plan, registered, native_before,
                                      global_files, prediction_files):
    """Bind the actual PKML Simulation, not its serialization transfer wrapper.

    Export may assign an ID different from the saved SQLite simulation ID.
    The pre-existing bridge/export/BLOB checks authenticate that conversion;
    this consumer checks the registered PKML against its actual native readback.
    Neither XML attributes nor any scientific model content are rewritten.
    """
    directory = f.path(directory)
    source = f.checked(registered['pkml'])
    before = f.path(native_before)
    _require(source in global_files and source in prediction_files
             and before in global_files and before in prediction_files,
             'actual observer documents must be present in both locks')
    _require(source.is_relative_to(directory) and before == directory/'native/before.pkml'
             and source != before, 'actual observer documents belong to another execution')
    expected_name = plan.get('source_simulation')
    _require(isinstance(expected_name, str) and bool(expected_name.strip()),
             'frozen source simulation name is missing')
    documents = []
    identities = []
    for path in (source, before):
        transfer = ET.parse(path).getroot()
        simulations = transfer.findall('Simulation')
        _require(transfer.tag == 'SimulationTransfer' and len(simulations) == 1
                 and len(list(transfer.iter('Simulation'))) == 1,
                 'PKML requires one direct Simulation under SimulationTransfer')
        simulation = simulations[0]
        models = simulation.findall('Model')
        _require(len(models) == 1 and len(list(transfer.iter('Model'))) == 1,
                 'PKML Simulation requires one direct Model')
        model = models[0]
        _require(simulation.get('name') == model.get('name') == expected_name
                 and isinstance(simulation.get('id'), str) and bool(simulation.get('id').strip())
                 and isinstance(model.get('id'), str) and bool(model.get('id').strip()),
                 'actual observer simulation/model identity is missing or changed')
        identities.append((simulation.get('id'), simulation.get('name'), model.get('id'), model.get('name')))
        documents.append(simulation)
    _require(identities[0] == identities[1],
             'native observer readback differs from the locked registered PKML identity')
    return documents[1]


def verify_bridge_semantics(f, directory, plan, registered, native_before, global_files, prediction_files):
    """Re-derive source design, both parent doses and the original R+S observer."""
    from . import json_source_design, json_racemate_contract as racemate, json_dual_native
    def local(name,value):
        path=directory/name
        _require(path in global_files and path in prediction_files and f.read(path)==value,
                 'source-specific contract is absent or changed in prediction locks: '+name)
        _require(registered.get(name.removesuffix('.json'))=={'path':str(path.relative_to(f.root)),'sha256':f.sha(path)},
                 'actual request lacks exact source-specific contract: '+name)
    protocol=f.read(directory.parent.parent/'protocol.json')
    def evidence(refs):
        for ref in refs:
            f.checked(ref)
            _require([r for r in protocol['inputs'] if r.get('path')==ref['path']]==[ref],
                     'source-specific evidence is not uniquely frozen')
    provenance={'source_path':plan['source_path'],'source_sha256':plan['source_sha256']}
    expected_design=json_source_design.prepare_source_design_amendment(provenance,plan['contract'],project_root=f.root)
    _require(plan.get('source_design_amendment')==expected_design, 'source design amendment differs from original source')
    if expected_design is not None:
        local('source_design_amendment.json',expected_design);evidence(expected_design['evidence_refs'])
        _require(plan['source_simulation']==expected_design['source_simulation']
                 and plan['native_compound_name']==expected_design['native_name'], 'source design selection changed')
    dual=plan.get('dual_parent_contract')
    if dual is None:
        _require(plan['output_path']!=racemate.OUTPUT_PATH, 'total racemate output requires both parent contracts')
        return {'source_design_rechecked':expected_design is not None,'dual_parent_rechecked':False}
    expected=racemate.prepare_dual_parent_contract(provenance,plan['source_simulation'],plan['contract'],project_root=f.root)
    _require(dual==expected, 'dual parent contract differs from original source')
    local('dual_parent_contract.json',dual);evidence(json_dual_native.evidence_refs(dual))
    _require(not plan.get('application_expansion'), 'dual-parent expansion needs a separately registered proof')
    isolation=plan.get('coadministration_isolation')
    if isolation:
        _require(isolation.get('target')==list(racemate.PARENTS), 'dual-parent isolation must retain both original parents')
    source=plan['json_native_source_proof']['native_source']
    observer=json_dual_native.source_output_contract(source['path'],source['sha256'],plan['source_simulation_id'],
        plan['source_model_blob_sha256'],dual,project_root=f.root)
    _require(observer==plan['dual_parent_observer'] and observer==plan['source_output_contract'],
             'original native dual observer differs')
    local('dual_parent_observer.json',observer)
    inventory=verified_inspection(f,directory,plan,registered,global_files,prediction_files)
    metadata=verified_metadata(f,plan)
    patches,events=json_dual_native.build_patches(inventory,plan,metadata,project_root=f.root)
    _require(registered.get('patches')==patches and registered.get('administrations')==events,
             'actual request does not preserve both parent dose channels')
    output,mw=json_dual_native.native_output(inventory,plan)
    _require(output==plan['output_path']==registered['output_path']
             and mw==plan['molecular_weight_g_per_mol']==registered['molecular_weight_g_per_mol'],
             'total racemate output or common molecular weight differs')
    document=verified_native_observer_document(f,directory,plan,registered,native_before,
        global_files,prediction_files)
    before=racemate.validate_native_observer(document,dual,
        molecular_weights={p:dual['molecular_weight_g_per_mol'] for p in racemate.PARENTS},project_root=f.root)
    _require(before['output_path']==observer['output_path']
             and before['original_observer_sha256']==observer['original_observer_sha256']
             and before['molecular_weight_g_per_mol']==observer['molecular_weight_g_per_mol'],
             'actual native model lacks the original total racemate observer')
    return {'source_design_rechecked':expected_design is not None,'dual_parent_rechecked':True,
            'both_parent_doses_recomputed':True,'original_and_actual_total_observer_rechecked':True}


def verify_split_semantics(f, directory, plan, registered, native, global_files, prediction_files):
    """Verify the author's two absorption inputs remain one clinical dose."""
    from . import split_absorption
    policy=plan['split_absorption_contract']
    source=plan.get('json_native_source_proof',{}).get('native_source') or {
        'path':plan['source_path'],'sha256':plan['source_sha256']}
    _require(policy.get('source_original')=={'path':plan['source_path'],'sha256':plan['source_sha256']}
             and policy.get('native_source')==source and policy.get('simulation_id')==plan['source_simulation_id']
             and policy.get('source_model_blob_sha256')==plan['source_model_blob_sha256'],
             'split absorption belongs to another source or simulation')
    path=directory/'split_absorption_contract.json'
    _require(path in global_files and path in prediction_files and f.read(path)==policy,
             'split absorption contract is absent or changed in prediction locks')
    ref={'path':str(path.relative_to(f.root)),'sha256':f.sha(path)}
    _require(registered.get('split_absorption_contract')==ref, 'actual request lacks exact split absorption contract')
    protocol=f.read(directory.parent.parent/'protocol.json')
    for evidence in policy['evidence_refs']:
        f.checked(evidence)
        _require([r for r in protocol['inputs'] if r.get('path')==evidence['path']]==[evidence],
                 'split absorption source evidence is not uniquely frozen')
    _require(not any(plan.get(k) for k in ('coadministration_isolation','application_expansion',
                                         'surplus_event_suppression','dual_parent_contract'))
             and plan.get('internal_absorption_event_count')==2,
             'split absorption cannot discard or relabel an original pulse')
    metadata=verified_metadata(f,plan)
    split_absorption.revalidate_contract(policy,plan['contract'],metadata,project_root=f.root)
    inventory=verified_inspection(f,directory,plan,registered,global_files,prediction_files)
    patches,events=split_absorption.build_split_patches(inventory,plan['contract'],metadata,policy,project_root=f.root)
    _require(registered.get('patches')==patches and registered.get('administrations')==events,
             'actual native request changes the original split dose or lag')
    audit=split_absorption.verify_native_readback(native,policy)
    _require(audit.get('accepted') is True, 'actual native split absorption readback failed')
    return audit
