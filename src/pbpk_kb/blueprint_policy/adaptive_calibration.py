"""Alternative optimization using the frozen native IO in a private call namespace.

No shared-module monkeypatching and no edits to historical calibration sources.
Numerical requests, native renal protection and readback are unchanged. Each
variant and objective is frozen in the existing input/mapping source contract.
"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from uuid import uuid4
import numpy as np

from . import native_calibration as baseline
from . import adaptive_optimizer
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprint_learning.native_calibration import _score as baseline_score, _read, _ref, _BudgetExhausted
from pbpk_kb.blueprints.development_curve_blueprint import _training_document

VERSION = 'adaptive-evidence-calibration-1'
VARIANTS = ('trust_region', 'trust_region_sparse', 'trust_region_robust', 'pattern')


def objective_config(variant, n):
    if variant not in VARIANTS: raise ValueError('Unknown optimization variant')
    if isinstance(n,bool) or not isinstance(n,int) or n<2: raise ValueError('At least two development observations')
    multiplier=max(1.,10./n) if variant=='trust_region_sparse' else 1.
    return {'version':VERSION,'variant':variant,'optimizer':'pattern' if variant=='pattern' else 'trust_region',
        'development_point_count':n,'prior_multiplier':multiplier,
        'data_loss':'soft_l1' if variant=='trust_region_robust' else 'squared_log',
        'robust_scale_log':math.log(2.),'fixed_clearance_prior_weight':.1,
        'fixed_lipophilicity_prior_weight':.01,'hard_origin_screening':False,
        'interpretation':'Engineering regularization; not a calibrated posterior or clinical confidence interval'}


def score_native(result, curve, observed, lip, scale, prior, weight, config):
    if not math.isclose(weight,.1,rel_tol=0,abs_tol=1e-15):
        raise ValueError('This frozen optimization protocol uses clearance prior weight 0.1')
    row=baseline_score(result,curve,observed,lip,scale,prior,weight)
    errors=np.asarray(row['sampled_log_ratios'],dtype=float)
    if len(errors)!=config['development_point_count']: raise ValueError('Observation count differs from objective')
    if config['data_loss']=='soft_l1':
        f=config['robust_scale_log'];z=(errors/f)**2
        # Stable 2*f^2*(sqrt(1+z)-1), only for the data term.
        squared=2*f*f*z/(np.sqrt(1+z)+1)
        data=np.sign(errors)*np.sqrt(squared/len(errors))
    else:data=errors/math.sqrt(len(errors))
    k=config['prior_multiplier']
    residual=np.r_[data,math.sqrt(weight*k)*math.log(scale)/math.log(2),
                   math.sqrt(.01*k)*(lip-prior)/2.5]
    row['residual_vector']=residual.tolist()
    row['objective_components']={'development_curve_loss':float(np.dot(data,data)),
        'clearance_input_prior':float(residual[-2]**2),'lipophilicity_prior':float(residual[-1]**2)}
    row['objective']=float(np.dot(residual,residual)); row['objective_config']=deepcopy(config)
    return row


def _engine(config):
    """One independent namespace per call, safe for concurrent different variants."""
    source=Path(baseline.__file__).resolve()
    spec=importlib.util.spec_from_file_location('_pbpk_adaptive_'+uuid4().hex,source)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load frozen native adapter')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    engine.VERSION=VERSION
    original_pins=engine._code_pins
    def pins(root):
        result=original_pins(root)
        for path in (Path(__file__).resolve(),Path(adaptive_optimizer.__file__).resolve()):
            result[str(inside(path,root,exists=True).relative_to(root))]=sha256(path)
        return result
    engine._code_pins=pins
    engine._screening_policy=lambda:deepcopy(config)
    engine._score=lambda result,curve,observed,lip,scale,prior,weight:score_native(
        result,curve,observed,lip,scale,prior,weight,config)
    return engine


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                variant='trust_region', max_evaluations=24, before_trial=None):
    root=Path(project_root).resolve(strict=True)
    observed=_training_document(training_source,reference_request,root)
    values=[float(r['concentration_umol_per_l']) for r in observed['observations']]
    config=objective_config(variant,len(values));engine=_engine(config)
    def optimize(evaluate,prior,allowed,screen_parameters):
        if screen_parameters:raise ValueError('Hard origin screening is not part of this variant')
        return adaptive_optimizer.optimize(evaluate,prior,values,allowed_parameters=allowed,
            max_evaluations=max_evaluations,prior_weight=.1,method=config['optimizer'],
            budget_exceptions=(_BudgetExhausted,))
    engine._pattern=optimize
    result=engine.fit_mapping(reference_request=reference_request,training_source=training_source,
        output_dir=output_dir,project_root=root,max_evaluations=max_evaluations,
        allowed_parameters=('lipophilicity','clearance_scale'),screen_parameters=False,
        before_trial=before_trial,prior_weight=.1)
    path=inside(result['mapping_path'],root,exists=True)
    stored=_read(path)
    spec={'optimization_config':config,'objective_spec':
        'mean(data_loss(ln(P/O))) + k*0.1*(ln(clearance_scale)/ln(2))^2 + k*0.01*((logMA-logP)/2.5)^2',
        'limitations':['Native physiology, evidence selection and fixed renal inputs remain unchanged.',
            'Search/objective comparison on previously viewed development drugs, not independent generalization.',
            'Strong priors can introduce bias when the source estimate is inaccurate.',
            'Numerical convergence and a local sensitivity do not prove parameter identifiability.']}
    if any(stored.get(k)!=v for k,v in spec.items()):
        stored.update(spec);save(path,stored)
    return {**result,**spec,'mapping':_ref(path,root)}


def _mapping_engine(mapping_path,project_root):
    root=Path(project_root).resolve(strict=True);path=inside(mapping_path,root,exists=True)
    mapping=_read(path);config=mapping.get('optimization_config')
    if not isinstance(config,dict) or config!=objective_config(config.get('variant'),config.get('development_point_count')):
        raise ValueError('Unrecognized frozen optimization configuration')
    if mapping.get('screening_policy')!=config:raise ValueError('Input/mapping objective differs')
    return _engine(config)


def predict_mapping(*,request,mapping_path,output_dir,project_root):
    engine=_mapping_engine(mapping_path,project_root)
    return engine.predict_mapping(request=request,mapping_path=mapping_path,output_dir=output_dir,project_root=project_root)


def cleanup_unselected(*,output_dir,project_root):
    engine=_mapping_engine(Path(output_dir)/'mapping_locked.json',project_root)
    return engine.cleanup_unselected(output_dir=output_dir,project_root=project_root)
