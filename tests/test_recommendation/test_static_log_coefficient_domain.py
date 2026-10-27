"""Signed source log coefficients retain exactly the domains used by the proof."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest

from scripts.recommendation_execution_repairs import coadministration_isolation as iso
from scripts.recommendation_execution_repairs import native_pkml
from scripts.recommendation_execution_repairs.event_associations import ModelIndex


LOG_PATH = 'Gemfibrozil|logP (veg.oil/water)'
FU_PATH = 'Gemfibrozil|Fraction unbound (plasma)'


def source_model():
    document = ET.Element('IndividualSimulation', name='Source', id='simulation')
    model = ET.SubElement(document, 'Model')
    root = ET.SubElement(model, 'Root', name='Source', id='root')
    formulas = ET.SubElement(ET.SubElement(model, 'FormulaCache'), 'Formulas')

    def node(parent, tag, name, identity, **attributes):
        return ET.SubElement(parent, tag, name=name, id=identity, **attributes)

    def formula(identity, expression, bindings=(), time=False):
        value = ET.SubElement(formulas, 'Formula', id=identity, formula=expression)
        paths = ET.SubElement(value, 'Paths')
        for alias, path in bindings:
            ET.SubElement(paths, 'Path', **{'as': alias, 'path': path})
        if time:
            ET.SubElement(paths, 'TimePath')
        return value

    applications = node(root, 'Container', 'Applications', 'applications')
    administrations = []
    for name in ('Target', 'Gemfibrozil'):
        group = node(applications, 'EventGroup', name, name + '-application', eventGroupType='Oral')
        schema = node(group, 'Container', 'ProtocolSchemaItem', name + '-schema')
        node(schema, 'Parameter', 'Dose', name + '-dose', dim='Mass', value='1')
        node(schema, 'Parameter', 'Start time', name + '-start', dim='Time', value='0')
        node(group, 'Event', 'Start', name + '-event', formula=name + '-condition')
        formula(name + '-condition', 'Time=StartTime',
                [('StartTime', '..|ProtocolSchemaItem|Start time')], time=True)
        administrations.append({'event_prefix': 'Applications|' + name,
                                'administered_compounds': [name],
                                'compound_association_complete': True})

    drug = node(root, 'Container', 'Gemfibrozil', 'drug')
    node(drug, 'Parameter', 'logP (veg.oil/water)', 'logp', dim='Log Units',
         negativeValuesAllowed='1', formula='logp-formula')
    node(drug, 'Parameter', 'Source logP', 'source-logp', dim='Log Units', value='-0.9288656680827935')
    formula('logp-formula', 'Value', [('Value', 'Gemfibrozil|Source logP')])
    node(drug, 'Parameter', 'Fraction unbound (plasma)', 'fu', dim='Fraction',
         negativeValuesAllowed='0', formula='fu-formula')
    node(drug, 'Parameter', 'Source fraction', 'source-fu', dim='Fraction', value='0.05')
    formula('fu-formula', 'Value', [('Value', 'Gemfibrozil|Source fraction')])

    body = node(root, 'Container', 'Body', 'body')
    node(body, 'MoleculeAmount', 'Gemfibrozil', 'amount', dim='Amount',
         quantityType='Drug', negativeValuesAllowed='0', value='0')
    reaction = node(body, 'Reaction', 'Elimination', 'reaction', formula='flux')
    ET.SubElement(ET.SubElement(reaction, 'Educts'), 'ReactionPartner', partner='amount')
    formula('flux', 'Mass / (0.004175 * 10^LogP + 0.001575 + 0.945 / Fu)',
            [('Mass', 'Body|Gemfibrozil'), ('LogP', LOG_PATH), ('Fu', FU_PATH)])
    return document, {'administrations': administrations}


@pytest.mark.parametrize('negative_allowed', [None, '1', '0'])
def test_static_sign_assumption_follows_explicit_source_metadata(negative_allowed):
    document, _ = source_model()
    index = ModelIndex(document)
    parameter = index.paths[LOG_PATH]
    if negative_allowed is None:
        parameter.attrib.pop('negativeValuesAllowed')
    else:
        parameter.set('negativeValuesAllowed', negative_allowed)
    proof = iso.FormulaProof(index, [], [])
    coefficient = proof.static_symbol(LOG_PATH)
    assert coefficient.is_real is True and coefficient.is_finite is True
    assert coefficient.is_nonnegative is (True if negative_allowed == '0' else None)


def test_real_partition_denominator_registers_fu_domain_without_signing_logp():
    document, metadata = source_model()
    before = ET.tostring(document)
    result = iso.prove_zero_manifold(document, metadata, 'Target')
    requirements = {row['path']: row for row in result['native_parameter_requirements']}
    assert result['verified'] is True and result['zero_flux_count'] == 1
    assert requirements[LOG_PATH] == {
        'path': LOG_PATH, 'id': 'logp', 'dimension': 'Log Units',
        'finite_required': True, 'nonnegative_required': False,
        'strictly_positive_required': False,
    }
    assert requirements[FU_PATH]['nonnegative_required'] is True
    assert requirements[FU_PATH]['strictly_positive_required'] is True
    assert ET.tostring(document) == before


def test_signed_exponent_domain_does_not_change_a_source_switch():
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    exponent = proof.static_symbol(LOG_PATH)
    proof.denominator(10 ** exponent)
    assert not proof.positive_paths
    expression = ET.Element('Formula', formula='LogP < 0 ? 1 : 2')
    ET.SubElement(ET.SubElement(expression, 'Paths'), 'Path', **{'as': 'LogP', 'path': LOG_PATH})
    branch = proof.expression(expression, 'Body|Gemfibrozil')
    assert branch.subs(exponent, -1) == 1
    assert branch.subs(exponent, 0) == 2


def test_bare_static_denominator_still_requires_a_checked_positive_domain():
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    coefficient = proof.static_symbol(LOG_PATH)
    proof.denominator(coefficient)
    assert proof.positive_paths == {LOG_PATH}


def test_zero_multiplication_preserves_a_true_log_argument_domain():
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    formula = ET.Element('Formula', formula='0 * log(LogP)')
    ET.SubElement(ET.SubElement(formula, 'Paths'), 'Path', **{'as': 'LogP', 'path': LOG_PATH})
    assert proof.expression(formula, 'Body|Gemfibrozil') == 0
    assert proof.positive_paths == {LOG_PATH}


def test_a_possible_zero_exponential_denominator_is_not_canceled():
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    formula = ET.Element('Formula', formula='0 / (10^LogP - 1)')
    ET.SubElement(ET.SubElement(formula, 'Paths'), 'Path', **{'as': 'LogP', 'path': LOG_PATH})
    with pytest.raises(ValueError, match='DENOMINATOR'):
        proof.expression(formula, 'Body|Gemfibrozil')


def test_exact_rational_normalization_precedes_new_positive_assumptions():
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    exponent = proof.static_symbol(LOG_PATH)
    value = (iso.sy.Rational(17, 1000) * 10 ** exponent
             - iso.sy.Rational(76, 5) / (1 + 10 ** iso.sy.Rational(23, 50))
             + iso.sy.Rational(823391, 35600))
    assert value.is_nonzero is None
    proof.denominator(value)
    assert not proof.positive_paths


def native_fixture():
    document, metadata = source_model()
    proof = iso.prove_zero_manifold(document, metadata, 'Target')
    request = iso.monitor_request({'verified': True, 'zero_manifold_proof': proof})
    rows = [{'path': row['path'], 'id': row['id'], 'dimension': row['dimension'],
             'unit': 'Log Units' if row['path'] == LOG_PATH else '',
             'value': -0.9288656680827935 if row['path'] == LOG_PATH else 0.05}
            for row in request['proof_parameter_requirements']]
    return request, {'proof_parameter_readback': rows}


def test_negative_log_readback_is_accepted_under_the_regenerated_proof():
    request, native = native_fixture()
    native_pkml.validate_proof_parameters(native, request)


@pytest.mark.parametrize('mutation', ['nonfinite_log', 'wrong_id', 'wrong_dimension', 'zero_fu', 'negative_fu', 'missing_log'])
def test_signed_log_support_preserves_independent_native_validation(mutation):
    request, native = native_fixture()
    rows = {row['path']: row for row in native['proof_parameter_readback']}
    if mutation == 'nonfinite_log':
        rows[LOG_PATH]['value'] = float('nan')
    elif mutation == 'wrong_id':
        rows[LOG_PATH]['id'] = 'foreign'
    elif mutation == 'wrong_dimension':
        rows[LOG_PATH]['dimension'] = 'Amount'
    elif mutation == 'zero_fu':
        rows[FU_PATH]['value'] = 0.0
    elif mutation == 'negative_fu':
        rows[FU_PATH]['value'] = -0.05
    else:
        native['proof_parameter_readback'].remove(rows[LOG_PATH])
    with pytest.raises(ValueError, match='Static'):
        native_pkml.validate_proof_parameters(native, request)


def test_a_frozen_positive_log_requirement_is_not_silently_relaxed():
    request, native = native_fixture()
    frozen = deepcopy(request)
    for requirement in frozen['proof_parameter_requirements']:
        if requirement['path'] == LOG_PATH:
            requirement['strictly_positive_required'] = True
    with pytest.raises(ValueError, match='Static'):
        native_pkml.validate_proof_parameters(native, frozen)


def test_uncertain_pruning_trial_never_repeats_exact_expansion(monkeypatch):
    document, _ = source_model()
    proof = iso.FormulaProof(ModelIndex(document), [], [])
    logp = proof.static_symbol(LOG_PATH)
    fu = proof.static_symbol(FU_PATH)
    def reject_expansion(expression, *args, **kwargs):
        raise AssertionError('Initial or pruning checks must not expand this simple denominator')

    monkeypatch.setattr(iso.sy, 'expand', reject_expansion)
    proof.denominator(logp * fu)
    assert proof.positive_paths == {LOG_PATH, FU_PATH}


def test_nonlog_conditional_nonnegative_domain_is_checked_despite_negative_allowed_flag():
    document, metadata = source_model()
    index = ModelIndex(document)
    index.paths[FU_PATH].set('negativeValuesAllowed', '1')
    index.formulas['flux'].set('formula', 'Mass * Fu')
    symbolic = iso.FormulaProof(index, [], [])
    assert symbolic.static_symbol(FU_PATH).is_nonnegative is True

    proof = iso.prove_zero_manifold(document, metadata, 'Target')
    request = iso.monitor_request({'verified': True, 'zero_manifold_proof': proof})
    assert len(request['proof_parameter_requirements']) == 1
    requirement = request['proof_parameter_requirements'][0]
    assert requirement['path'] == FU_PATH
    assert requirement['nonnegative_required'] is True
    assert requirement['strictly_positive_required'] is False
    native = {'proof_parameter_readback': [{
        'path': FU_PATH, 'id': 'fu', 'dimension': 'Fraction', 'unit': '', 'value': 0.0,
    }]}
    native_pkml.validate_proof_parameters(native, request)
    native['proof_parameter_readback'][0]['value'] = -0.05
    with pytest.raises(ValueError, match='Static'):
        native_pkml.validate_proof_parameters(native, request)
