"""Compare complete saved-model semantics across official XML serializers.

IDs, shared formula caches and string-table indexes are serialization identity,
not model identity. Resolve them to full entity paths and complete formula trees.
Unknown attributes/nodes stay in the comparison. No discrepancy is learned from
the comparison target, and existing NaN placeholders never become finite values.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'complete-model-semantics-v1'
ALIASES = {'Para': 'Parameter', 'DistPara': 'DistributedParameter', 'Compartment': 'Container', 'Organ': 'Container', 'Organism': 'Container'}
PARAMETERS = {'Parameter', 'DistributedParameter'}
LINKS = {'neighborhoods', 'sourceAmount', 'targetAmount', 'firstNeighbor', 'secondNeighbor',
         'changedEntity', 'partner', 'educt', 'product'}
NUMERIC = {'value', 'min', 'max', 'percentile', 'stoichiometry', 'scaleFactor'}
PRESENTATION_ATTRIBUTES = {'description', 'icon'}
SOURCE_FILES = [
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Serialization/Xml/QuantityXmlSerializer.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Serialization/Xml/Extensions/XElementExtensions.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Xml/Serializers/PKSimParameterXmlSerializer.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/Parameter.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/TimePath.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/ProjectConverter/v12/Converter11To12.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/ParameterInfo.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/DistributedParameter.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Xml/Serializers/OrganXmlSerializer.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/ProjectConverter/v7_3/Converter721To730.cs',
    'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Converters/v7_3/Converter710To730.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Model/Compartment.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Model/Organ.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Model/Organism.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Xml/Serializers/ContainerXmlSerializer.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/ProjectConverter/v7_4/Converter730To740.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Presentation/Presenters/Parameters/ParameterGroupsPresenter.cs',
]
NORMALIZATIONS = [
    'Infusion time parameters in ProtocolSchemaItem may display min or h: QuantityXmlSerializer stores Value in base units independently of DisplayUnit. Exact base value, Time dimension, formulas and every other attribute remain strict.',
    'Dimensionless Percentile children of DistributedParameter ignore legacy displayUnit annotations: official XML loading drops the invalid unit without rescaling Value. Percentile value, dimensions, parent distribution and all actual units remain strict.',
    'PK-Sim Organism/Organ/Compartment subclasses map to Container; containerType/topology is retained.',
    'PK-Sim Para/DistPara map to Parameter/DistributedParameter; rhs maps to rHSFormula.',
    'PK-Sim parameter mode maps to BuildMode; native default flags follow Parameter constructor.',
    'Official v12 conversion maps legacy buildMode Property to Global; GUI ReadOnly/Visible flags are excluded, while all other parameter flags and bounds remain compared.',
    'Entity IDs resolve to exact typed model paths; formula IDs recursively resolve to their full definition.',
    'Formula-cache string maps are expanded for paths, aliases and dimensions; cache order/duplication is not model behavior.',
    'TimePath without alias has the official default Time.',
    'Scalar parameter value and an equivalent ConstantFormula are the same serialized quantity definition.',
    'Legacy EventAssignment.objectPath attribute and native ObjectPath child are compared as the same complete path.',
    'PK-Sim organType rebuild classification and compartment visibility are retained in source metadata but excluded from an already-built model comparison; containerType, complete topology and equations remain strict.',
    'Only id, icon, description and origin provenance annotations, and parameter reset-default metadata (default/isDefault) are excluded; current values/formulas/fixed-value flags remain strict.',
    'Legacy valueDescription and empty native ValueOrigin descriptions are provenance retained in the original XML; ValueOrigin method/source classifications stay compared.',
    'Only Tablet time delay factor Info.seq 1 and 20 are equivalent: official v7.4 migration changes its UI display order to 20; the parameter value, formula, dimension and all other metadata remain strict.',
    'NaN/Infinity are explicit nonfinite tokens; equality preserves a placeholder and never validates a finite numerical prediction.',
    'All other model nodes, attributes, child multiplicities, formulas, dimensions, connections and event targets remain compared.',
]


def _number(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f):
        return ('nonfinite', 'NaN')
    if math.isinf(f):
        return ('nonfinite', '+Infinity' if f > 0 else '-Infinity')
    return ('number', f)


def _stable(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def model_semantics(model):
    """Return a stable, JSON-serializable, ID-free representation of one full Model.

    This accepts legacy saved PK-Sim Model XML and standard simulation-transfer
    Model XML. Dangling/ambiguous references and duplicate semantic paths fail.
    It deliberately does not compare simulation configuration or solver settings.
    """
    if not isinstance(model, ET.Element) or model.tag != 'Model':
        raise ValueError('Exactly one Model element is required')
    roots = model.findall('Root')
    caches = model.findall('FormulaCache')
    if len(roots) != 1 or len(caches) != 1:
        raise ValueError('Complete Model requires exactly one Root and FormulaCache')
    cache = caches[0]
    maps = cache.findall('StringMap')
    if len(maps) > 1:
        raise ValueError('Multiple formula string maps')
    string_map = {row.attrib['id']: row.attrib['s'] for row in maps[0]} if maps else {}
    if maps and len(string_map) != len(maps[0]):
        raise ValueError('Duplicate string-map IDs')
    groups = cache.findall('Formulas')
    if len(groups) != 1:
        raise ValueError('Exactly one formula definitions group is required')
    formulas = {node.get('id'): node for node in groups[0]}
    if None in formulas or len(formulas) != len(groups[0]):
        raise ValueError('Formula identities are missing or duplicated')
    entity_ids, path_keys = {}, set()
    info_owners={child:parent for parent in model.iter() for child in parent if child.tag=='Info'}
    parents={child:parent for parent in model.iter() for child in parent}
    def index(node, path):
        tag = ALIASES.get(node.tag, node.tag)
        if node.get('name') and node.tag not in {'Info', 'Origin', 'ValueOrigin'}:
            path = path + (node.get('name'),)
            key = (tag, '|'.join(path))
            if key in path_keys:
                raise ValueError('Duplicate typed entity path: ' + key[1])
            path_keys.add(key)
            if node.get('id'):
                if node.get('id') in entity_ids:
                    raise ValueError('Duplicate entity ID')
                entity_ids[node.get('id')] = ('entity',) + key
        for child in node:
            index(child, path)
    index(roots[0], ())
    used_formulas, memo, resolving = set(), {}, set()
    nonfinite = []
    def reference(identifier):
        if identifier not in formulas:
            raise ValueError('Dangling formula reference: ' + str(identifier))
        used_formulas.add(identifier)
        if identifier in memo:
            return memo[identifier]
        if identifier in resolving:
            raise ValueError('Cyclic formula definition reference')
        resolving.add(identifier)
        result = canonical(formulas[identifier], is_formula=True)
        resolving.remove(identifier); memo[identifier] = result
        return result
    def canonical(node, *, is_formula=False):
        tag = ALIASES.get(node.tag, node.tag)
        attrs = dict(node.attrib)
        attrs.pop('id', None)
        for key in PRESENTATION_ATTRIBUTES:
            attrs.pop(key, None)
        # Native model mapping updates source provenance to newly cloned IDs;
        # these annotations do not participate in a stored model's equations.
        attrs.pop('origin', None)
        children = [child for child in node if child.tag != 'Origin' and not (
            child.tag == 'ValueOrigin' and not len(child) and not (child.text or '').strip()
            and not (set(child.attrib) - {'id', 'description'}))]
        if node.tag == 'Organ':
            attrs.pop('organType', None)  # PK-Sim rebuild classification; no rebuilding here.
        if node.tag == 'Compartment':
            attrs.pop('visible', None)  # UI-only subclass property.

        if tag in PARAMETERS:
            immediate=parents.get(node);container=parents.get(immediate)
            if (tag=='Parameter' and attrs.get('name')=='Infusion time' and attrs.get('dim')=='Time'
                    and attrs.get('displayUnit') in {'min','h'} and immediate is not None
                    and immediate.tag=='Children' and container is not None
                    and container.get('name')=='ProtocolSchemaItem'):
                attrs['displayUnit']='min'  # presentation only; Value is already in native base units
            owner=parents.get(parents.get(node))
            if (tag=='Parameter' and attrs.get('name')=='Percentile' and not attrs.get('dim')
                    and owner is not None and ALIASES.get(owner.tag,owner.tag)=='DistributedParameter'):
                # The official loader reads numeric Value directly and resolves
                # displayUnit only within Dimension; no dimension => no unit.
                attrs.pop('displayUnit',None)
            attrs.pop('default', None)  # ResetToDefault is never executed here.
            attrs.pop('isDefault', None)  # Rebuild/UI provenance introduced by v7.3 conversion.
            attrs.pop('valueDescription', None)  # Moved to ValueOrigin.description by official conversion.
            if node.tag in ALIASES and 'mode' in attrs:
                attrs['buildMode'] = attrs.pop('mode')
            if 'rhs' in attrs:
                if 'rHSFormula' in attrs:
                    raise ValueError('Conflicting RHS serialization fields')
                attrs['rHSFormula'] = attrs.pop('rhs')
            if tag == 'DistributedParameter':
                attrs.setdefault('mode', 'Logical')
            attrs.setdefault('buildMode', 'Local')
            if attrs['buildMode'] == 'Property':
                attrs['buildMode'] = 'Global'
            attrs.setdefault('persistable', '0')
            attrs.setdefault('isFixedValue', '1' if 'fix' in attrs else '0')
            attrs.pop('fix', None)
            attrs.setdefault('quantityType', 'Parameter')
            attrs.setdefault('negativeValuesAllowed', '1')
            # The legacy constant is stored as value instead of a FormulaCache
            # entry. Preserve a simultaneous nonconstant formula/fixed override.
            constant = None
            if attrs.get('formula') in formulas and formulas[attrs['formula']].tag == 'ConstantFormula':
                f = formulas[attrs['formula']]
                if len(f) or set(f.attrib) - {'id', 'name', 'value', 'dim', 'description'}:
                    raise ValueError('ConstantFormula has unreviewed fields')
                used_formulas.add(attrs.pop('formula'))
                constant = _number(f.get('value'))
                if f.get('dim', attrs.get('dim')) != attrs.get('dim', f.get('dim')):
                    raise ValueError('Constant formula dimension differs from its quantity')
            elif 'formula' not in attrs and 'value' in attrs:
                constant = _number(attrs['value'])
            if constant is not None:
                if 'value' in attrs and _number(attrs['value']) != constant:
                    raise ValueError('Constant formula and explicit value conflict')
                attrs.pop('value', None)
                attrs['constant_value'] = constant
        if tag == 'TimePath':
            attrs.setdefault('alias', 'Time')
        if tag == 'Info' and 'flag' in attrs:
            attrs['flag'] = str(int(attrs['flag']) & ~12)
        if tag=='Info' and attrs.get('seq') in {'1','20'}:
            owner=info_owners.get(node)
            if (owner is not None and ALIASES.get(owner.tag,owner.tag) in PARAMETERS and
                    owner.get('name')=='Tablet time delay factor'):
                attrs['seq']='20'
        if tag in {'Path', 'ObjectPath'}:
            attrs = {k: string_map.get(v, v) if k in {'path', 'as', 'dim'} else v for k, v in attrs.items()}
        if tag == 'EventAssignment':
            object_paths = [n for n in children if n.tag == 'ObjectPath']
            if len(object_paths) > 1:
                raise ValueError('Ambiguous event assignment target path')
            if object_paths:
                obj = object_paths[0]
                if len(obj) or set(obj.attrib) != {'path'}:
                    raise ValueError('Unreviewed event target path serialization')
                value = string_map.get(obj.get('path'), obj.get('path'))
                if attrs.get('objectPath', value) != value:
                    raise ValueError('Conflicting event assignment paths')
                attrs['objectPath'] = value
                children.remove(obj)
        normalized = {}
        for key, value in attrs.items():
            if key in {'formula', 'rHSFormula'} and not is_formula:
                value = reference(value)
            elif key in LINKS:
                if value not in entity_ids:
                    raise ValueError('Dangling model entity reference: ' + key + ':' + str(value))
                value = entity_ids[value]
            elif key in NUMERIC:
                value = _number(value)
            if isinstance(value, tuple) and value and value[0] == 'nonfinite':
                nonfinite.append({'entity_id': node.get('id'), 'name': node.get('name'), 'field': key, 'token': value[1]})
            normalized[key] = value
        child_values = [canonical(child, is_formula=is_formula) for child in children]
        # XML sibling ordering of named model entities/cache maps is not model
        # topology. Formula table point order IS semantic and remains ordered.
        if tag in {'Children', 'Paths', 'Tags', 'Educts', 'Products', 'ModifierNames'} or tag in PARAMETERS:
            child_values.sort(key=lambda value: (value[0], str(dict(value[1]).get('name', '')), _stable(value)))
        return (tag, tuple(sorted(normalized.items())), (node.text or '').strip(), tuple(child_values))
    # FormulaCache itself is only an object/string interning table; every live
    # formula is expanded at the actual model entity that references it.
    model_attrs = {k: v for k, v in model.attrib.items() if k not in {'id', 'icon', 'description', 'origin'}}
    for key in LINKS & model_attrs.keys():
        if model_attrs[key] not in entity_ids:
            raise ValueError('Dangling model root reference')
        model_attrs[key] = entity_ids[model_attrs[key]]
    others = [canonical(child) for child in model if child.tag != 'FormulaCache']
    semantic = ('Model', tuple(sorted(model_attrs.items())), tuple(others))
    counts = Counter(ALIASES.get(n.tag, n.tag) for n in roots[0].iter())
    return {'version': VERSION, 'canonical': semantic,
            'sha256': hashlib.sha256(_stable(semantic).encode()).hexdigest(),
            'counts': dict(sorted(counts.items())), 'entity_paths': len(path_keys),
            'formula_definitions': len(formulas), 'referenced_formula_definitions': len(used_formulas),
            'unreferenced_formula_definitions': len(set(formulas) - used_formulas),
            'nonfinite_placeholders': nonfinite, 'nonfinite_is_numerical_success': False}


def _diff(a, b, path='', limit=50):
    out = []
    def visit(x, y, where):
        if len(out) >= limit or x == y:
            return
        if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)) and len(x) == len(y):
            for i, (left, right) in enumerate(zip(x, y)):
                visit(left, right, where + '/' + str(i))
                if len(out) >= limit:
                    break
        else:
            out.append({'path': where, 'source': str(x)[:1000], 'exported': str(y)[:1000]})
    visit(a, b, path)
    return out


def compare_saved_model_xml(source_Model, exported_Model):
    left, right = model_semantics(source_Model), model_semantics(exported_Model)
    accepted = left['canonical'] == right['canonical']
    return {'version': VERSION, 'accepted': accepted,
            'counts': {'source': {k: v for k, v in left.items() if k != 'canonical'},
                       'exported': {k: v for k, v in right.items() if k != 'canonical'}},
            'differences': [] if accepted else _diff(left['canonical'], right['canonical']),
            'normalizations': NORMALIZATIONS, 'unknown_differences_ignored': False,
            'scope': 'Complete saved Model only; simulation settings/configuration/population/source independence require separate checks.',
            'implementation': {'path': str(Path(__file__).relative_to(ROOT)),
                               'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'official_source_evidence': [{'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in SOURCE_FILES]}


compare_models = compare_saved_model_xml


def compare_saved_settings_xml(sourceSettings, exportedSettings):
    """Compare solver/output selections/schema/random seed; do not reset settings.

    Creation records and serializer IDs are provenance only. All other fields,
    including every solver parameter, output interval and randomSeed, are strict.
    Reuse the model normalizer for legacy versus native Parameter serialization.
    """
    from copy import deepcopy
    def wrap(settings):
        if not isinstance(settings, ET.Element) or settings.tag not in {'Settings','SimulationSettings'}:
            raise ValueError('Exactly one saved Settings or SimulationSettings element is required')
        model = ET.Element('Model')
        root = ET.SubElement(model, 'Root', {'name': 'SettingsComparison'})
        copy = deepcopy(settings)
        # PK-Sim SimulationXmlSerializer maps x.Settings; the core exchange
        # configuration maps SimulationSettings. Change only this root tag in
        # the comparison copy, retaining seed, solver and all unknown fields.
        copy.tag = 'SimulationSettings'
        caches = copy.findall('FormulaCache')
        if len(caches) > 1:
            raise ValueError('Multiple SimulationSettings formula caches')
        for node in list(copy):
            if node.tag == 'Creation':
                copy.remove(node)
        root.append(copy)
        if caches:
            copy.remove(caches[0])
            model.append(caches[0])
        else:
            cache = ET.SubElement(model, 'FormulaCache')
            ET.SubElement(cache, 'Formulas')
        return model
    result = compare_saved_model_xml(wrap(sourceSettings), wrap(exportedSettings))
    result['scope'] = 'All SimulationSettings including randomSeed, Solver, OutputSchema and OutputSelections; only Creation provenance and serialization identity/presentation excluded.'
    result['normalizations'] = list(result['normalizations']) + ['SimulationSettings Creation provenance is excluded; randomSeed is always retained.']
    if sourceSettings.tag=='Settings' or exportedSettings.tag=='Settings':
        result['normalizations'].append('Saved PK-Sim Settings and core SimulationSettings have the same root type; only the comparison-copy tag is normalized and original XML hashes retain the saved tag.')
        sources=(
            'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Xml/Serializers/SimulationXmlSerializer.cs',
            'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Serialization/Xml/SimulationSettingsXmlSerializer.cs')
        result['official_source_evidence'] += [{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources]
    return result
