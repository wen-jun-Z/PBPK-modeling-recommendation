"""Fixed complementary formulas and frozen OOF alignment, without training."""
from copy import deepcopy

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import clearance_complementary as cl


def test_three_fixed_candidates_and_mean_median_formulas():
    c=cl.complementary_predictions([0.,2.],[1.,0.],[2.,.5])
    assert list(c)==[cl.gate.BASE,cl.MEAN,cl.MEDIAN]
    assert np.allclose(c[cl.MEAN],[1.,2.5/3])
    assert np.allclose(c[cl.MEDIAN],[1.,.5])
    with pytest.raises(ValueError):cl.complementary_predictions([0.],[0.,1.],[0.])
    with pytest.raises(ValueError):cl.complementary_predictions([0.],[np.nan],[0.])


def test_frozen_component_alignment_rejects_different_rows_labels_folds_or_baseline():
    audit={'training_indices':list(range(1124)),'oof_fold_ids':[i%3 for i in range(1124)],
        'observed_training_log10':[0.]*1124,'cl_training_labels':1124,
        'candidate_oof_log10':{cl.gate.BASE:[1.]*1124,cl.animal.AUGMENTED:[2.]*1124,cl.KRR_CANDIDATE:[3.]*1124}}
    source={'audit':audit};y,folds,candidates=cl.align_component_oof(source,deepcopy(source))
    assert len(y)==1124 and len(folds)==1124 and np.all(candidates[cl.MEAN]==2.)
    for key in ('training_indices','oof_fold_ids','observed_training_log10'):
        altered=deepcopy(source);altered['audit'][key][0]+=1
        with pytest.raises(ValueError):cl.align_component_oof(source,altered)
    altered=deepcopy(source);altered['audit']['candidate_oof_log10'][cl.gate.BASE][0]+=1
    with pytest.raises(ValueError):cl.align_component_oof(source,altered)


def test_failed_combinations_keep_baseline_without_searching_other_weights():
    y=np.array([-2.,.5,2.]);base=y+.1
    c=cl.complementary_predictions(base,y+1.,y+2.)
    selection=cl.gate.evaluate_candidates(y,c,[0,1,2])
    assert selection['selected']==cl.gate.BASE
    assert selection['eligible']==[cl.gate.BASE]
