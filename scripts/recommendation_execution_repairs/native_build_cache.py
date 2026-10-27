"""Reuse exact-input successful builds; always repeat the new source proof.

An old readback request/process binds the saved project hash even when the
later scientific-preservation comparison failed. No predictions are reused.
"""
import json
from pathlib import Path
from . import native_partition_resume as audit


def reuse_build(input_path,directory,*,root):
    root=Path(root);parts=directory.relative_to(root).parts
    revision=next((v for v in ('snapshot_repairs_11','snapshot_repairs_10','snapshot_repairs_09','snapshot_repairs_08') if v in parts),None)
    if revision is None:return None
    i=parts.index(revision)
    if len(parts)<=i+1:return None
    prior=root.joinpath(*parts[:i],'snapshot_repairs_07',parts[i+1],'complex_source_proof')
    if not prior.is_dir():return None
    document=audit._read(input_path)
    for stage in sorted(prior.glob('batch_*/*build/stage_receipt.json')):
        record=audit._read(stage)
        try:
            original=audit._checked(record['input'],root)
            if not audit._exact(audit._read(original),document):continue
            _,_,refs=audit._stage(stage.parent,'snapshot_to_project',record['input'],root)
            name=stage.parent.name.removesuffix('_build')+'_readback'
            readback=stage.parent.parent/name
            request_path=readback/'request.json';process_path=readback/'process_manifest.json'
            request=audit._read(request_path);process=audit._read(process_path)
            if (process.get('command',[])[-1:]!=[audit.migration.sha(request_path)]
                    or Path(process.get('script','')).resolve()!=root/'src/pbpk_kb/model_runtime/R/recommendation_snapshot_readback.R'):
                continue
            project=audit._checked({'path':str(Path(request['project_path']).relative_to(root)),
                                    'sha256':request['project_sha256']},root)
            if not project.is_relative_to(stage.parent/'output'):continue
            refs += [record['input'],audit._ref(project,root),audit._ref(request_path,root),audit._ref(process_path,root)]
            directory.mkdir()
            receipt={'status':'exact_input_build_reused_pending_new_full_proof','input':audit._ref(input_path,root),
                'historical_input':record['input'],'native_project':audit._ref(project,root),
                'historical_stage':audit._ref(stage,root),'references':refs,
                'predictions_reused':False,'clinical_outcomes_accessed':False}
            audit.migration.save(directory/'build_reuse.json',receipt)
            return project,[*refs,audit._ref(directory/'build_reuse.json',root)]
        except (ValueError,KeyError,FileNotFoundError,IndexError):
            # Never accept a weak cache entry; the normal bounded build follows.
            continue
    return None
