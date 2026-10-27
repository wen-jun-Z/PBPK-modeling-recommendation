"""Conservative qualification of a protein-binding value used as plasma fu.

Only an explicit metadata flag triggers this rule. Unknown study overlap by
itself does not reject a plasma measurement. No drug identity, clinical outcome,
model error, or file content is used to select which evidence is retained.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

VERSION = 'qualified-plasma-binding-1'
REJECTION = 'general_protein_binding_does_not_establish_plasma_fu'


def _fallback(record, request):
    method=record.get('prediction_method') or {};context=record.get('context') or {}
    app=method.get('applicability') or {};source=record.get('source') or {};val=record.get('value')
    if (record.get('parameter')!='fraction_unbound' or record.get('compound_id')!=request.get('compound_id')
            or record.get('basis')!='predicted' or record.get('unit')!='fraction'
            or isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or not 0<val<=1
            or not method.get('name') or not method.get('version') or method.get('training_overlap') is not False
            or app.get('human_IV_parameter_model') is not True or app.get('target_structure_and_scaffold_excluded') is not True
            or context.get('species')!='human' or context.get('matrix')!='plasma'
            or context.get('study_role') in {'target','evaluation','test','validation','heldout'}
            or not source.get('path') or re.fullmatch('[0-9a-f]{64}',str(source.get('sha256',''))) is None):
        raise ValueError('A qualified same-compound ML plasma-fu fallback is required')
    for ref in request.get('forbidden_sources',[]):
        if source['path']==ref.get('path') or source['sha256']==ref.get('sha256'):
            raise ValueError('A target-observation source cannot supply the fu fallback')


def qualify_binding(request, fallback_fu):
    """Return a copied request and audit, changing only a rejected fu record.

    The caller must verify all source hashes and bind the resulting evidence to
    an exact ledger or an identical previously locked request before execution.
    """
    result=deepcopy(request)
    indices=[i for i,r in enumerate(result.get('evidence',[])) if r.get('parameter')=='fraction_unbound']
    if len(indices)!=1:
        raise ValueError('Exactly one selected fraction-unbound record is required')
    index=indices[0];record=result['evidence'][index];context=record.get('context') or {}
    reject=(record.get('basis') in {'measured','derived'}
            and context.get('general_protein_binding_adopted_for_plasma') is True)
    audit={'version':VERSION,'changed':reject,'parameter':'fraction_unbound',
        'criterion':'Only explicit general_protein_binding_adopted_for_plasma=True on measured/derived fu',
        'unknown_study_overlap_is_not_a_rejection_rule':True,'clinical_outcomes_used':False,
        'reason':REJECTION if reject else 'No explicit general-binding-to-plasma assumption; keep selected evidence'}
    if reject:
        _fallback(fallback_fu,request)
        audit.update(rejected_evidence=deepcopy(record),fallback_evidence=deepcopy(fallback_fu),
            source_receipts={'rejected':deepcopy(record['source']),'fallback':deepcopy(fallback_fu['source'])})
        result['evidence'][index]=deepcopy(fallback_fu)
    else:
        audit['retained_evidence']=deepcopy(record)
    return result,audit
