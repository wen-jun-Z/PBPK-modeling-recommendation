"""Animal-label parsing and cross-species identity/scaffold isolation."""
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import clearance_animal as cl


def test_only_animal_whitelist_is_interpreted_and_raw_units_logged_once():
    data=('smiles_r,dog_CL_mL_min_kg,monkey_CL_mL_min_kg,rat_CL_mL_min_kg,human_CL_mL_min_kg\n'
          'CCO,10,100,1,THIS_IS_NOT_A_LABEL\nCCN,0,NaN,-2,12345\n').encode()
    rows=cl.animal_rows(data)
    assert rows[0]['log10_cl']=={'dog':1.,'monkey':2.,'rat':0.}
    assert rows[1]['log10_cl']=={}
    assert rows[1]['label_exclusions']=={'dog':'nonpositive_or_nonfinite','monkey':'missing','rat':'nonpositive_or_nonfinite'}
    assert all('human' not in key for row in rows for key in row['log10_cl'])
    with pytest.raises(ValueError):
        cl.animal_rows(data.replace(b'CCN',b'OCC'))


def test_exclusion_removes_identity_and_same_scaffold_before_label_use():
    def row(index,smiles):return {'index':index,**cl.shared.normalize_structure(smiles),'log10_cl':{}}
    rows=[row(0,'c1ccccc1'),row(1,'Cc1ccccc1'),row(2,'C1CCCCC1'),row(3,'C1CCCC1')]
    allowed,excluded=cl.exclude_structures(rows,[rows[0]])
    assert [r['index'] for r in allowed]==[3]
    assert excluded=={'0':'protected_identity','1':'protected_generic_scaffold','2':'protected_generic_scaffold'}
    targets=[row(9,'C1CCCC1')]
    globally,_=cl.exclude_structures(rows,targets)
    fold_allowed,_=cl.exclude_structures(globally,[rows[0]])
    assert fold_allowed==[]


def test_animal_recipe_is_fixed_and_training_imputation_is_local():
    model=cl.make_animal_estimator(1)
    assert model.named_steps['model'].n_estimators==300
    assert model.named_steps['model'].min_samples_leaf==2
    model.set_params(model__n_estimators=2)
    model.fit([[1.,np.nan],[3.,2.],[5.,4.]],[-1.,0.,1.])
    assert model.named_steps['impute'].statistics_[1]==3.
    assert np.isfinite(model.predict([[2.,np.nan]])).all()


def test_animal_blend_uses_fixed_half_log_weight():
    base=np.log10([1.,4.,100.]);aux=np.log10([4.,1.,1.])
    assert np.allclose(10**(.5*(base+aux)),[2.,2.,10.])
    y=np.array([-2.,.5,2.]);bad=y+1.;good=y+.1
    result=cl.gate.evaluate_candidates(y,{cl.gate.BASE:bad,cl.RAW:bad,cl.AUGMENTED:good,cl.BLEND:.5*(bad+good)},[0,1,2])
    assert result['selected']==cl.AUGMENTED
