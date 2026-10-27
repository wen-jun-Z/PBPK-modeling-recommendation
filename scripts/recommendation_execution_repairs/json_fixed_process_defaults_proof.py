"""Narrow readback proof for three registered Dabigatran build partitions.

No snapshot is rewritten. Only explicit calculation defaults whose entire
native dependency frontier is an unchanged fixed kcat may differ in export.
The ordinary exact snapshot comparator and all numeric tolerances are unchanged.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
from . import source_pkml, snapshot_repair
from .event_associations import ModelIndex

ROOT = source_pkml.ROOT
RULE = 'source_bound_fixed_process_calculation_defaults'
REGISTRATION = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_builds_20261002/fixed_process_defaults/registration_v2.json'
REGISTRATION_SHA256 = '892820a9cb9656ca1ca1fc599890e784ab466db8192210e946c76754116cc483'
TRANSPORT = 'ActiveTransportSpecific_MM'
METABOLISM = 'MetabolizationLiverMicrosomes_MM'
CONFIG = (
    ('DabiEtex', TRANSPORT, 'ABCB1', None, 'DabiEtex|ABCB1-FIT'),
    ('DabiEtex', METABOLISM, 'CES1', 'Dabigatran', 'DabiEtex-CES1-FIT'),
    ('DabiEtex', METABOLISM, 'CES2', 'Dabigatran', 'DabiEtex-CES2-FIT'),
    ('Dabigatran', METABOLISM, 'UGT2B15', 'DabiGluc', 'Dabigatran-UGT2B15-FIT'),
)
DEFAULTS = {
    TRANSPORT: (('Transporter concentration', 1, 'µmol/l', 'Concentration (molar)'),
                ('Vmax', 0, 'µmol/l/min', 'Concentration (molar) per time')),
    METABOLISM: (('In vitro Vmax for liver microsomes', 0, 'pmol/min/mg mic. protein', 'Vmax per mass protein'),),
}

def require(ok, reason):
    if not ok:
        raise ValueError('FIXED_PROCESS_DEFAULTS_' + reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked(ref, root):
    return snapshot_repair.checked(ref, root)


def unique(rows, key):
    keys = [key(x) for x in rows]
    require(len(keys) == len(set(keys)), 'DUPLICATE_IDENTITY')
    return dict(zip(keys, rows))


def processes(snapshot):
    compounds = unique(snapshot['Compounds'], lambda c: c['Name'])
    return {(name, p.get('InternalName'), p.get('Molecule'), p.get('Metabolite')): p
            for name, c in compounds.items()
            for p in unique(c.get('Processes', []), lambda p: (
                p.get('InternalName'), p.get('Molecule'), p.get('DataSource'),
                p.get('Species'), p.get('Metabolite'))).values()
            if p.get('DataSource') == 'FIT'}


def compare_documents(original, exported, *, allow_protocol_defaults=False):
    """Check original parameters verbatim; remove ONLY proved extra defaults in a copy."""
    adjusted = deepcopy(exported)
    before, after = processes(original), processes(adjusted)
    changes = []
    for compound, kind, molecule, metabolite, prefix in CONFIG:
        identity = (compound, kind, molecule, metabolite)
        require(identity in before and identity in after, 'EXACT_PROCESS_REQUIRED')
        old, new = before[identity], after[identity]
        original_params = unique(old['Parameters'], lambda p: p['Name'])
        native_params = unique(new['Parameters'], lambda p: p['Name'])
        added = {name: {'Name': name, 'Value': value, 'Unit': unit}
                 for name, value, unit, dimension in DEFAULTS[kind]}
        require(set(native_params) == set(original_params) | set(added)
                and not (set(original_params) & set(added)), 'PARAMETER_COVERAGE_DIFFERS')
        require(all(native_params[name] == value for name, value in original_params.items()),
                'ORIGINAL_PARAMETER_CHANGED')
        require(all(type(native_params[name].get('Value')) in (int, float) and native_params[name] == value for name, value in added.items()),
                'NONDEFAULT_PARAMETER_ADDED')
        kcat = original_params.get('kcat')
        require(kcat is not None and kcat.get('Unit') == '1/min'
                and type(kcat.get('Value')) in (int, float), 'EXPLICIT_SOURCE_KCAT_REQUIRED')
        new['Parameters'] = [p for p in new['Parameters'] if p['Name'] not in added]
        changes.append({'compound': compound, 'internal_name': kind, 'molecule': molecule,
                        'metabolite': metabolite, 'native_prefix': prefix,
                        'source_kcat': deepcopy(kcat), 'added_defaults': list(added.values()),
                        'original_process_sha256': digest(old)})
    implicit = implicit_protocol_defaults(original, adjusted) if allow_protocol_defaults else []
    report = snapshot_repair.exact_comparison(original, adjusted)
    require(report.get('exact_normalized_document_equality') is True, 'OTHER_SNAPSHOT_DIFFERENCES')
    return changes, report, implicit


# These are native importer defaults, not fitted or invented replacement values.
IMPLICIT_DEFAULTS = {'Lag time': (0, 'h', 'Time', '1'),
                     'Start time': (0, 'h', 'Time', '60'),
                     'Volume of water/body weight': (3.5, 'ml/kg', 'Volume per body weight', '0.001')}

def implicit_protocol_defaults(original, adjusted):
    rows = []
    def inspect(old, new, identity, allowed):
        before = unique(old['Parameters'], lambda p: p['Name'])
        after = unique(new['Parameters'], lambda p: p['Name'])
        require(all(after.get(k) == v for k, v in before.items()), 'ORIGINAL_PROTOCOL_PARAMETER_CHANGED')
        extra = set(after) - set(before)
        require(extra <= allowed, 'UNSUPPORTED_IMPLICIT_DEFAULT')
        for name in sorted(extra):
            value, unit, dimension, factor = IMPLICIT_DEFAULTS[name]
            require(type(after[name].get('Value')) in (int, float) and after[name] == {'Name': name, 'Value': value, 'Unit': unit}, 'IMPLICIT_DEFAULT_DIFFERS')
            rows.append({**identity, 'parameter': deepcopy(after[name]), 'dimension': dimension,
                         'native_value': str(Decimal(str(value)) * Decimal(factor))})
        new['Parameters'] = [p for p in new['Parameters'] if p['Name'] not in extra]
    forms = unique(adjusted['Formulations'], lambda p: p['Name'])
    for old in original['Formulations']:
        if old['Name'] == 'DabiEtex_Capsule':
            require(old['FormulationType'] == forms[old['Name']]['FormulationType'] == 'Formulation_Tablet_Weibull', 'FORMULATION_TYPE_CHANGED')
            inspect(old, forms[old['Name']], {'kind': 'Formulation', 'name': old['Name']}, {'Lag time'})
    protocols = unique(adjusted['Protocols'], lambda p: p['Name'])
    allowed_names = {'Clarithromycin DDI, Delavenne 2012 - ' + suffix for suffix in
                     ('Dabigatran (70.5h)', 'Dabigatran (0h)', 'Clarithromycin')}
    for old in original['Protocols']:
        if old['Name'] not in allowed_names:
            continue
        new = protocols[old['Name']]
        schemas = unique(new['Schemas'], lambda p: p['Name'])
        for schema in old['Schemas']:
            items = unique(schemas[schema['Name']]['SchemaItems'], lambda p: p['Name'])
            for item in schema['SchemaItems']:
                require(item['ApplicationType'] == items[item['Name']]['ApplicationType'] == 'Oral', 'ORAL_PROTOCOL_REQUIRED')
                inspect(item, items[item['Name']], {'kind': 'Protocol', 'name': old['Name'],
                        'schema': schema['Name'], 'item': item['Name']}, {'Start time', 'Volume of water/body weight'})
    require(len(rows) == 10, 'EXACT_IMPLICIT_DEFAULT_COVERAGE_REQUIRED')
    return rows


def validate_implicit_defaults(document, original, exported, changes):
    index = ModelIndex(document)
    blocks = [b for wrapper in document.findall('UsedBuildingBlocks/*') for b in wrapper]
    simulation = next(s for s in original['Simulations'] if s['Name'] == document.get('name'))
    used_protocols = {c['Protocol']['Name'] for c in simulation['Compounds'] if c.get('Protocol')}
    used_formulations = {f['Name'] for c in simulation['Compounds'] if c.get('Protocol') for f in c['Protocol'].get('Formulations', [])}
    proofs = []
    for row in changes:
        tag = 'Formulation' if row['kind'] == 'Formulation' else 'AdvancedProtocol'
        matches = [b for b in blocks if b.tag == tag and b.get('name') == row['name']]
        if row['name'] not in (used_protocols if row['kind'] == 'Protocol' else used_formulations):
            require(not matches, 'UNUSED_PROTOCOL_BINDING_DIFFERS')
            proofs.append({**row, 'used_by_simulation': False})
            continue
        require(len(matches) == 1, 'USED_DEFAULT_BUILDING_BLOCK_REQUIRED')
        block = matches[0]
        if row['kind'] == 'Formulation':
            parent = block.find('Root/Children')
        else:
            schemas = [s for s in block.findall('Root/Children/Schema') if s.get('name') == row['schema']]
            require(len(schemas) == 1, 'SCHEMA_IDENTITY_DIFFERS')
            items = [i for i in schemas[0].findall('Children/SchemaItem') if i.get('name') == row['item']]
            require(len(items) == 1, 'ITEM_IDENTITY_DIFFERS')
            parent = items[0].find('Children')
        params = [n for n in parent if n.get('name') == row['parameter']['Name']]
        require(len(params) == 1, 'DEFAULT_PARAMETER_IDENTITY_DIFFERS')
        node = params[0]
        require(node.get('dim') == row['dimension'] and node.get('displayUnit') == row['parameter']['Unit']
                and Decimal(node.get('value', 'NaN')) == Decimal(row['native_value'])
                and not any(node.get(k) for k in ('formula', 'rhs', 'rHSFormula')), 'NATIVE_IMPLICIT_DEFAULT_DIFFERS')
        proofs.append({**row, 'used_by_simulation': True, 'building_block_id': block.get('id'),
                       'native_node_sha256': source_pkml.element_sha(node)})
        if row['kind'] == 'Formulation':
            values = [(p, n) for p, n in index.paths.items() if p.endswith('|' + row['name'] + '|Lag time')]
            require(bool(values), 'NATIVE_FORMULATION_DEFAULT_REQUIRED')
            for path, n in values:
                origin = n.find('Origin')
                require(n.get('dim') == 'Time' and Decimal(n.get('value', 'NaN')) == 0
                        and not any(n.get(k) for k in ('formula', 'rhs', 'rHSFormula'))
                        and origin is not None and origin.get('bb') == block.get('id')
                        and origin.get('para') == node.get('id'), 'NATIVE_FORMULATION_DEFAULT_BINDING_DIFFERS')
    schedules = []
    for name in sorted({c['name'] for c in changes if c['kind'] == 'Protocol'} & used_protocols):
        protocol = next(p for p in exported['Protocols'] if p['Name'] == name)
        require(len(protocol['Schemas']) == 1, 'SINGLE_SCHEMA_REQUIRED')
        schema = protocol['Schemas'][0]; pars = unique(schema['Parameters'], lambda p: p['Name'])
        require(pars['NumberOfRepetitions']['Value'] == 1 and pars['TimeBetweenRepetitions']['Value'] == 0
                and pars['Start time']['Unit'] == 'h', 'SOURCE_SCHEMA_SCHEDULE_DIFFERS')
        prefixes = [(p.rsplit('|', 1)[0], n) for p, n in index.paths.items()
                    if p.startswith('Events|' + name + '|') and p.endswith('|ProtocolSchemaItem|Start time')]
        prefixes.sort(key=lambda row: int(re.search(r'\|Application_(\d+)\|ProtocolSchemaItem$', row[0]).group(1)))
        require(len(prefixes) == len(schema['SchemaItems']), 'EVENT_COUNT_DIFFERS')
        for item, (prefix, start_node) in zip(schema['SchemaItems'], prefixes):
            parameters = unique(item['Parameters'], lambda p: p['Name'])
            require(parameters['Start time']['Unit'] == 'h'
                    and parameters['Volume of water/body weight']['Unit'] == 'ml/kg', 'SOURCE_EVENT_UNITS_DIFFER')
            expected_start = (Decimal(str(pars['Start time']['Value'])) + Decimal(str(parameters['Start time']['Value']))) * 60
            expected_water = Decimal(str(parameters['Volume of water/body weight']['Value'])) / 1000
            water = index.node(prefix + '|Volume of water/body weight', {'Para', 'Parameter'}, 'Volume per body weight')
            require(start_node.get('dim') == 'Time' and Decimal(start_node.get('value', 'NaN')) == expected_start
                    and Decimal(water.get('value', 'NaN')) == expected_water
                    and not any(n.get(k) for n in (start_node, water) for k in ('formula', 'rhs', 'rHSFormula')),
                    'EVENT_SCHEDULE_OR_WATER_CHANGED')
            schedules.append({'event_prefix': prefix, 'source_schema_item': item['Name'],
                              'start_time_min': str(expected_start), 'water_l_per_kg': str(expected_water),
                              'start_node_sha256': source_pkml.element_sha(start_node),
                              'water_node_sha256': source_pkml.element_sha(water)})
    return {'implicit_default_readbacks': proofs, 'source_event_schedule_readbacks': schedules}


def validate_model(document, changes):
    """All direct consumers of added defaults must be exact source-fixed kcats."""
    index = ModelIndex(document)
    added_nodes = {}; fixed_nodes = {}; records = []
    for change in changes:
        prefix, kind = change['native_prefix'], change['internal_name']
        path = prefix + '|kcat'
        node = index.node(path, {'Para', 'Parameter'}, 'Inversed time')
        require(node.get('isFixedValue') == '1' and node.get('isDefault') == '0'
                and node.get('mode') == 'Global' and node.get('displayUnit') == '1/min'
                and not node.get('rhs') and not node.get('rHSFormula'), 'KCAT_NOT_SOURCE_FIXED')
        require(Decimal(node.get('value', 'NaN')) == Decimal(str(change['source_kcat']['Value'])),
                'KCAT_VALUE_CHANGED')
        fixed_nodes[path] = node
        if kind == TRANSPORT:
            formula = index.formula(node, 'Vmax/TransporterConcentration', {
                'Vmax': prefix + '|Vmax', 'TransporterConcentration': prefix + '|Transporter concentration'})
        else:
            formula = index.formula(node, 'InVitroVmax/CYP', {
                'InVitroVmax': prefix + '|In vitro Vmax for liver microsomes',
                'CYP': prefix + '|Content of CYP proteins in liver microsomes'})
        defaults = []
        for name, value, unit, dimension in DEFAULTS[kind]:
            p = prefix + '|' + name
            n = index.node(p, {'Para', 'Parameter'}, dimension)
            require(n.get('mode') == 'Global' and n.get('displayUnit') == unit
                    and Decimal(n.get('value', 'NaN')) == Decimal(value)
                    and not any(n.get(k) for k in ('formula', 'rhs', 'rHSFormula')),
                    'NATIVE_DEFAULT_DIFFERS')
            added_nodes[p] = n
            defaults.append({'path': p, 'id': n.get('id'), 'node_sha256': source_pkml.element_sha(n)})
        records.append({'kcat_path': path, 'kcat_id': node.get('id'),
                        'kcat_node_sha256': source_pkml.element_sha(node),
                        'fixed_value': node.get('value'), 'formula': formula, 'defaults': defaults})
    protected_ids = {n.get('id') for n in [*added_nodes.values(), *fixed_nodes.values()]}
    # No event, assignment, or other id-bound operation may replace these constants.
    for node in index.model.iter():
        for key, value in node.attrib.items():
            require(key == 'id' or value not in protected_ids, 'PROTECTED_PARAMETER_ID_REFERENCE')
    target_names = {p.rsplit('|', 1)[-1] for p in added_nodes}
    consumers = []
    # Scan every model entity/attribute referencing a formula, including RHS and
    # assignments. Object paths to any target necessarily end in its exact name.
    for node in index.model.iter():
        for attribute, formula_id in node.attrib.items():
            if attribute == 'id' or formula_id not in index.formulas:
                continue
            formula = index.formulas[formula_id]
            for pointer in formula.findall('.//Path'):
                raw = index.strings.get(pointer.get('path'), pointer.get('path'))
                require(isinstance(raw, str), 'FORMULA_PATH_REQUIRED')
                if raw.rsplit('|', 1)[-1] not in target_names:
                    continue
                context = index.node_paths.get(id(node))
                require(context is not None, 'FORMULA_CONTEXT_REQUIRED')
                # Bare paths address children; explicit ./../absolute paths use
                # the existing native model resolver.
                child = context + '|' + raw
                target = child if child in index.paths else index.resolve(raw, context)
                if target not in added_nodes:
                    continue
                require(attribute == 'formula' and context in fixed_nodes
                        and target.rsplit('|', 1)[0] == context.rsplit('|', 1)[0],
                        'UNFIXED_DEFAULT_CONSUMER')
                consumers.append({'parameter_path': target, 'consumer_path': context,
                                  'formula_attribute': attribute, 'formula_id': formula_id})
    require(len(consumers) == len(added_nodes)
            and {c['parameter_path'] for c in consumers} == set(added_nodes),
            'DEFAULT_CONSUMER_COVERAGE_DIFFERS')
    return {'simulation_name': document.get('name'), 'simulation_id': document.get('id'),
            'fixed_processes': records, 'default_consumers': sorted(consumers, key=lambda x: x['parameter_path'])}


def official_evidence(registration, root):
    for ref in registration['official_refs']:
        checked(ref, root)
    db_ref = next(r for r in registration['official_refs'] if r['path'].endswith('PKSimDB.sqlite'))
    rows = []
    with sqlite3.connect(checked(db_ref, root).as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        for kind, defaults in DEFAULTS.items():
            for name, value, unit, dimension in defaults:
                groups = db.execute('SELECT group_name FROM tab_container_parameters WHERE container_name=? AND parameter_name=?', (kind, name)).fetchall()
                require(groups == [('COMPOUNDPROCESS_CALCULATION_PARAMETERS',)], 'OFFICIAL_CALCULATION_GROUP_DIFFERS')
                rates = db.execute('SELECT calculation_method,formula_rate FROM tab_container_parameter_rates WHERE container_name=? AND parameter_name=?', (kind, name)).fetchall()
                expected_rate = 'One_Rate' if value == 1 else 'Zero_Rate'
                require(rates == [('LinksCommon', expected_rate)], 'OFFICIAL_DEFAULT_RATE_DIFFERS')
                formulas = db.execute('SELECT formula FROM tab_calculation_method_rates WHERE calculation_method=? AND formula_rate=?', rates[0]).fetchall()
                require(formulas == [(str(value),)], 'OFFICIAL_DEFAULT_VALUE_DIFFERS')
                rows.append({'process': kind, 'parameter': name, 'group': groups[0][0],
                             'rate': expected_rate, 'formula': str(value)})
        for method, rate, value in [('SchemaItemParameter', 'PARAM_Application_AmountOfWaterPerBodyweight', '0.0035'),
                                     ('SchemaItemParameter', 'PARAM_Application_StartTime', '0'),
                                     ('Formulation', 'PARAM_LagTime_Tablet', '0')]:
            actual = db.execute('SELECT formula FROM tab_calculation_method_rates WHERE calculation_method=? AND formula_rate=?', (method, rate)).fetchall()
            require(actual == [(value,)], 'OFFICIAL_IMPLICIT_DEFAULT_DIFFERS')
            rows.append({'calculation_method': method, 'rate': rate, 'formula': value})
    return rows


def derive(input_ref, export_ref, native_ref, *, project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    registration_ref = {'path': REGISTRATION, 'sha256': REGISTRATION_SHA256}
    registration = json.loads(checked(registration_ref, root).read_text())
    matches = [e for e in registration['entries'] if e['input'] == input_ref
               and e['export_snapshot'] == export_ref and e['native_source'] == native_ref]
    require(len(matches) == 1, 'REGISTERED_SOURCE_PARTITION_REQUIRED')
    entry = matches[0]
    refs = [registration_ref, registration['queue_registration'], *registration['official_refs'],
            *entry['runtime_refs'], *[entry[k] for k in ('source_original', 'source_receipt',
            'verified_snapshot', 'input', 'export_snapshot', 'native_source', 'raw_comparison',
            'result', 'build_process_manifest')]]
    for ref in refs:
        checked(ref, root)
    official = official_evidence(registration, root)
    original = json.loads(checked(input_ref, root).read_text())
    exported = json.loads(checked(export_ref, root).read_text())
    changes, remainder, implicit = compare_documents(original, exported, allow_protocol_defaults=entry['source_original']['sha256'] == 'c474e0bfebd70b93d75491fd0de58a3284e277aede13ddec0654d4370afbfc18')
    raw = snapshot_repair.exact_comparison(original, exported)
    require(raw == json.loads(checked(entry['raw_comparison'], root).read_text()), 'RAW_COMPARISON_CHANGED')
    require(raw['exact_normalized_document_equality'] is False, 'SPECIAL_PROOF_NOT_NEEDED')
    path = checked(native_ref, root)
    inventory = source_pkml.simulation_inventory(path, native_ref['sha256'], project_root=root)
    require(len(inventory) == len(entry['simulation_names'])
            and {r['name'] for r in inventory} == set(entry['simulation_names']), 'SIMULATION_COVERAGE_DIFFERS')
    models = []
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        for row in inventory:
            blob = db.execute('SELECT Data FROM CONTENTS WHERE Id=?', (row['content_id'],)).fetchone()[0]
            document = ET.fromstring(source_pkml.decompress_model_blob(blob))
            require(document.get('id') == row['simulation_id'] and document.get('name') == row['name'], 'NATIVE_IDENTITY_DIFFERS')
            source_pkml.saved_settings(document)
            defaults_proof = validate_implicit_defaults(document, original, exported, implicit) if implicit else {}
            models.append({**validate_model(document, changes), **defaults_proof, 'content_id': row['content_id'],
                           'source_blob_sha256': hashlib.sha256(blob).hexdigest()})
    checked(native_ref, root)
    return {'schema_version': 'source-bound-fixed-process-calculation-defaults-proof-v1',
            'equivalence_rule': RULE, 'accepted': True, 'exact_normalized_document_equality': False,
            'numeric_tolerance': {'relative': 0, 'absolute': 0},
            'partition_input': input_ref, 'export_snapshot': export_ref, 'native_source': native_ref,
            'source_original': entry['source_original'], 'raw_comparison': entry['raw_comparison'],
            'original_parameters_unchanged': True, 'changes': changes, 'implicit_protocol_defaults': implicit,
            'exact_remainder_comparison': remainder, 'official_template_defaults': official,
            'native_simulations': models, 'provenance_refs': list({r['path']: r for r in refs}.values()),
            'clinical_outcomes_accessed': False, 'ode_run': False}


def verify(input_ref, export_ref, native_ref, proof_ref, *, project_root=ROOT):
    actual = derive(input_ref, export_ref, native_ref, project_root=project_root)
    expected = json.loads(checked(proof_ref, project_root).read_text())
    require(actual == expected, 'PROOF_CHANGED')
    return actual
