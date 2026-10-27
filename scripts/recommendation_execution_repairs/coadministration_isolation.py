"""Source-bound removal of administration inputs, never of drug mechanisms.

The zero-mass manifold is checked in the original reaction/transport graph.
Only another administered parent's Dose and Start time may be patched.  Its
actual reaction products are followed by entity identity, including explicit
Compound metabolites.  All physiology and shared mechanisms remain present.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import xml.etree.ElementTree as ET

import sympy as sy

from . import event_associations as association, source_pkml, surplus_event_suppression as surplus

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'source-bound-zero-exposure-coadministration-isolation-1'
IMPLEMENTATION_FILES = tuple(dict.fromkeys([
    'scripts/recommendation_execution_repairs/coadministration_isolation.py',
    'scripts/recommendation_execution_repairs/surplus_event_suppression.py',
    *association.IMPLEMENTATION_FILES,
]))


DISCRETE_DISTRIBUTION_EVIDENCE = [{'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/Formulas/DiscreteDistributionFormula.cs', 'sha256': 'c94e3debb237405bfd946ffa1ae556055a091524d62099351113c84122c360f6'}, {'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/Formulas/DistributionFormula.cs', 'sha256': '51600b7bc79ddeb07433571cf9ad0e3bc09b5020165e03981312459b95b21890'}, {'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/DistributedParameter.cs', 'sha256': 'ee040d007e1af89bd8bc5ab7ea656c61f92e2e0574b1e1b2fd40496935165ccb'}]

URINARY_EVENT_EVIDENCE = [{'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/EventAssignment.cs', 'sha256': 'b8ecb2fdb0c8fe129844185283489edadb336fc3609bfe2782f3e339ee409c4d'}, {'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Serialization/SimModel/Services/SimulationExportCreator.cs', 'sha256': '1354329f27b7b42a777f0167607942528e7c1cb3ed3355e32b3d27b450063503'}]

def _require(condition, message):
    if not condition:
        raise ValueError('ISOLATION_' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _model_metadata(value):
    # Database envelope fields are already bound by source_reference; both
    # preparation and execution must hash the same substantive model fields.
    return {k:v for k,v in value.items() if k not in {'mode','content_id','compressed_bytes','source_blob_sha256'}}


def _rewrite(text):
    """Translate the small mathematical grammar without eval/parse_expr."""
    _require(isinstance(text, str) and 0 < len(text) < 20000, 'BOUNDED_FORMULA_REQUIRED')
    # First translate every parenthesized subexpression, then top-level ?:.
    pieces = []; i = 0
    while i < len(text):
        if text[i] != '(':
            pieces.append(text[i]); i += 1; continue
        level = 1; j = i + 1
        while j < len(text) and level:
            level += (text[j] == '(') - (text[j] == ')'); j += 1
        _require(level == 0, 'UNBALANCED_FORMULA')
        pieces.append('(' + _rewrite(text[i + 1:j - 1]) + ')'); i = j
    text = ''.join(pieces)
    depth = 0; question = None; nesting = 0
    for i, char in enumerate(text):
        depth += (char == '(') - (char == ')')
        if depth:
            continue
        if char == '?':
            if question is None: question = i
            nesting += 1
        elif char == ':' and question is not None:
            nesting -= 1
            if nesting == 0:
                return 'where(' + _rewrite(text[:question]) + ',' + _rewrite(text[question + 1:i]) + ',' + _rewrite(text[i + 1:]) + ')'
    _require(question is None, 'UNTERMINATED_CONDITIONAL')
    return re.sub(r'(?<![<>=!])=(?!=)', '==', text.replace('^', '**').replace(';', ',')).replace('NOT ', 'not ').replace(' AND ', ' and ').replace(' OR ', ' or ').replace('&', ' and ').replace('|', ' or ').strip()


class FormulaProof:
    """Conservative symbolic zero proof with explicit static-domain receipts.

Unresolved *static* parameters become symbols; required finite/positive
domains are independently checked against the native model after patching.
Time-dependent or event-written values are never called static.  No clinical
observations, fitted values, or external Python expression evaluation is used.
"""
    def __init__(self, index, zero_paths, zero_doses):
        self.index = index; self.zero_paths = set(zero_paths); self.zero_doses = set(zero_doses)
        self.cache = {}; self.busy = set(); self.static_cache = {}; self.static_busy = set()
        self.dynamic_positive_paths = set(); self.static_symbols = {}; self.symbol_paths = {}; self.positive_paths = set(); self.dynamic_symbols = {}; self.discrete_expansions = {}
        self.written = {index.ids[n.get('changedEntity')] for n in index.model.iter('EventAssignment') if n.get('changedEntity') in index.ids}

    def references(self, formula, path):
        result = {}; times = []
        for item in list(formula.find('Paths')) if formula.find('Paths') is not None else []:
            alias = self.index.strings.get(item.get('as'), item.get('as'))
            if item.tag == 'TimePath':
                times.append(alias or 'Time'); continue
            _require(item.tag == 'Path' and alias and alias not in result, 'UNIQUE_TYPED_FORMULA_PATH_REQUIRED')
            raw = self.index.strings.get(item.get('path'), item.get('path'))
            root_names = {self.index.root_name, self.index.document.get('name')} - {None, ''}
            absolute = [raw[len(name) + 1:] for name in root_names
                        if isinstance(raw, str) and raw.startswith(name + '|')]
            if absolute:
                _require(len(set(absolute)) == 1, 'AMBIGUOUS_ABSOLUTE_ROOT')
                tokens = []
                for token in absolute[0].split('|'):
                    if token == '..':
                        _require(tokens, 'ABSOLUTE_PATH_ESCAPES_MODEL'); tokens.pop()
                    elif token != '.': tokens.append(token)
                resolved = '|'.join(tokens)
                _require(resolved in self.index.paths, 'ABSOLUTE_PATH_NOT_IN_MODEL:' + resolved)
                result[alias] = resolved
                continue
            child = path + '|' + raw if isinstance(raw, str) else None
            # Original reaction/transport formulas also use a bare child name
            # (e.g. Km_app), with the quantity itself as their path context.
            if child in self.index.paths and not raw.startswith(('.', '..')):
                _require(raw not in self.index.paths, 'AMBIGUOUS_BARE_PATH')
                result[alias] = child
            else:
                result[alias] = self.index.resolve(item.get('path'), path)
        return result, times

    def is_static(self, path):
        if path in self.zero_paths or path in self.zero_doses: return False
        if path in self.static_cache: return self.static_cache[path]
        if path in self.static_busy: return False
        node = self.index.paths[path]
        if node.tag == 'MoleculeAmount' or node.get('rhs', node.get('rHSFormula')) or path in self.written:
            return False
        self.static_busy.add(path)
        try:
            formula = self.index.formulas.get(node.get('formula'))
            if formula is None or node.get('isFixedValue') == '1': answer = node.get('value') is not None
            else:
                refs, times = self.references(formula, path)
                answer = not times and all(self.is_static(p) for p in refs.values())
        finally:
            self.static_busy.remove(path)
        self.static_cache[path] = answer
        return answer

    def static_symbol(self, path):
        node = self.index.paths[path]
        _require(node.tag in {'Para', 'Parameter', 'DistPara', 'DistributedParameter'}, 'STATIC_NATIVE_PARAMETER_REQUIRED:' + path)
        if path not in self.static_symbols:
            # Logarithmic dimensions admit negative finite values. Other static
            # coefficients retain the existing conditional nonnegative domain,
            # independently enforced by native parameter readback.
            assumptions={'real':True} if node.get('dim')=='Log Units' and node.get('negativeValuesAllowed')!='0' else {'nonnegative':True}
            symbol = sy.Symbol('p' + str(len(self.static_symbols)), finite=True, **assumptions)
            self.static_symbols[path] = symbol; self.symbol_paths[symbol] = path
        return self.static_symbols[path]

    def value(self, path):
        if path in self.zero_paths or path in self.zero_doses: return sy.Integer(0)
        if path in self.cache: return self.cache[path]
        _require(path not in self.busy, 'CYCLIC_ALGEBRAIC_DEPENDENCY:' + path)
        self.busy.add(path); node = self.index.paths[path]
        try:
            if node.tag == 'MoleculeAmount' or node.get('rhs', node.get('rHSFormula')) or path in self.written:
                nonnegative = node.get('negativeValuesAllowed') == '0'
                positive_volume = node.get('dim') == 'Volume'
                if positive_volume: self.dynamic_positive_paths.add(path)
                value = sy.Symbol('d' + str(len(self.dynamic_symbols)), real=True, finite=True, **({'positive': True} if positive_volume else {'nonnegative': True} if nonnegative else {}))
                self.dynamic_symbols[value] = path
            elif (node.get('dim') and node.tag in {'Para', 'Parameter', 'DistPara', 'DistributedParameter'}
                  and node.get('value') is None and self.is_static(path)
                  and (self.index.formulas[node.get('formula')].tag != 'Formula'
                       or len(self.index.formulas[node.get('formula')].find('Paths') or []) > 0)):
                # Keep a frozen static volume/Km/etc. as one coefficient.
                # Its native value must be finite; positivity is registered only
                # when required to prove a denominator/log/power domain. Dimensionless
                # switch expressions are evaluated, never assumed enabled.
                value = self.static_symbol(path)
            else:
                value = self.initial(node)
        finally:
            self.busy.remove(path)
        self.cache[path] = value
        return value

    def initial(self, node):
        path = self.index.node_paths[id(node)]
        formula = self.index.formulas.get(node.get('formula'))
        if node.get('value') is not None and (formula is None or node.get('isFixedValue') == '1'):
            number = float(node.get('value')); _require(math.isfinite(number), 'FINITE_INITIAL_REQUIRED:' + path)
            return sy.Rational(node.get('value'))
        _require(formula is not None, 'EXPLICIT_INITIAL_OR_FORMULA_REQUIRED:' + path)
        if formula.tag == 'ConstantFormula':
            _require(not len(formula) and math.isfinite(float(formula.get('value'))), 'FINITE_CONSTANT_REQUIRED')
            return sy.Rational(formula.get('value'))
        if formula.tag != 'Formula':
            _require(self.is_static(path), 'UNSUPPORTED_DYNAMIC_FORMULA:' + path + ':' + formula.tag)
            return self.static_symbol(path)
        return self.expression(formula, path)

    def exact_discrete_constant(self, path):
        """Expand only a static dimensionless discrete distribution's explicit
        Mean, using the official formula. Never use default or percentile.
        """
        node=self.index.paths[path]
        formula=self.index.formulas.get(node.get('formula'))
        if (node.tag not in {'DistPara','DistributedParameter'} or formula is None
                or formula.tag!='DiscreteDistributionFormula'
                or node.get('dim') not in (None,'','Fraction','Dimensionless')
                or node.get('value') is not None or node.get('isFixedValue')=='1'
                or not self.is_static(path)):
            return None
        refs,times=self.references(formula,path)
        if times or refs!={'Mean':path+'|Mean'}:return None
        mean=self.index.paths.get(refs['Mean'])
        if (mean is None or mean.tag not in {'Para','Parameter'} or mean.get('value') is None
                or (mean.get('formula') and mean.get('isFixedValue')!='1')
                or not self.is_static(refs['Mean'])
                or mean.get('rhs',mean.get('rHSFormula')) or refs['Mean'] in self.written):
            return None
        _require(math.isfinite(float(mean.get('value'))),'FINITE_DISCRETE_MEAN_REQUIRED')
        exact=sy.Rational(mean.get('value'))
        for ref in DISCRETE_DISTRIBUTION_EVIDENCE:
            _require(source_pkml.sha(ROOT/ref['path'])==ref['sha256'],'OFFICIAL_DISCRETE_FORMULA_EVIDENCE_CHANGED')
        return {'path':path,'id':node.get('id'),'parameter_sha256':source_pkml.element_sha(node),
                'formula_sha256':source_pkml.element_sha(formula),'mean_path':refs['Mean'],
                'mean_id':mean.get('id'),'mean_parameter_sha256':source_pkml.element_sha(mean),
                'exact_rational_value':str(exact),'source_mean_literal':mean.get('value'),
                'default_value_used':False,'percentile_used':False,'event_written':False,
                'native_readback_required':True}

    def denominator(self, value, guards=()):
        self.domain(value,'nonzero','DENOMINATOR_NOT_PROVEN_NONZERO',guards)

    def domain(self,value,attribute,message,guards=()):
        if getattr(value,'is_'+attribute) is True:return
        # A native conditional evaluates only the selected branch. In V>0 ?
        # M/V : 0, V may be zero outside that branch; never impose V>0 globally.
        clauses=[part for guard in guards for part in sy.And.make_args(guard)]
        if attribute=='nonzero' and any(c in (sy.Gt(value,0),sy.Lt(value,0),sy.Ne(value,0)) for c in clauses):return
        guarded={}
        for clause in clauses:
            for symbol in clause.free_symbols:
                if clause in (sy.Gt(symbol,0),sy.Lt(0,symbol)):
                    guarded[symbol]=sy.Symbol(str(symbol)+'_branch',positive=True,finite=True)
        value=value.xreplace(guarded)
        if getattr(value,'is_'+attribute) is True:return
        if value.has(sy.Piecewise):
            folded=sy.piecewise_fold(value)
            _require(isinstance(folded,sy.Piecewise) and len(folded.args)<=64,
                     'BOUNDED_CONDITIONAL_DOMAIN_REQUIRED')
            # Prove the entire denominator on each branch, so a positive Km
            # plus a guarded concentration does not force a zero-volume
            # compartment to become globally positive.
            for branch,condition in folded.args:
                self.domain(branch,attribute,message,guards+(condition,))
            return
        def proven(expression,allow_expand=True):
            if getattr(expression,'is_'+attribute) is True:return True
            if sy.count_ops(expression)>512:return False
            # Exact rational normalization, with no numerical approximation.
            numerator,denominator=sy.fraction(sy.together(expression))
            if allow_expand:numerator=sy.expand(numerator)
            if sy.count_ops(numerator)>4096 or sy.count_ops(denominator)>4096:return False
            same_sign=((numerator.is_positive is True and denominator.is_positive is True)
                       or (numerator.is_negative is True and denominator.is_negative is True))
            return (same_sign if attribute in {'positive','nonnegative'} else
                    numerator.is_nonzero is True and denominator.is_nonzero is True)
        if proven(value,allow_expand=False):return
        static={s:sy.Symbol(str(s)+'_domain',positive=True,finite=True)
                for s in sorted(value.free_symbols,key=str) if s in self.symbol_paths}
        if not proven(value.xreplace(static)):
            # Existing successful symbolic proofs retain their original shape.
            expanded={}
            for symbol in sorted(value.free_symbols,key=str):
                if symbol not in self.symbol_paths:continue
                item=self.exact_discrete_constant(self.symbol_paths[symbol])
                if item is not None:expanded[symbol]=item
            _require(bool(expanded),message+':'+str(value))
            literal=value.xreplace({sym:sy.Rational(item['exact_rational_value']) for sym,item in expanded.items()})
            self.domain(literal,attribute,message,guards)
            self.discrete_expansions.update({item['path']:item for item in expanded.values()})
            return
        # Initial and pruning trials avoid expansion of unconstrained symbolic
        # expressions. Failed cheap trials retain their checked sign requirement.
        # In particular 10**logP is positive for every finite real logP, even <0.
        for symbol in sorted(static,key=lambda s:self.symbol_paths[s]):
            trial={s:v for s,v in static.items() if s!=symbol}
            if getattr(value.xreplace(trial),'is_'+attribute) is True:static=trial
        self.positive_paths.update(self.symbol_paths[s] for s in static)

    def expression(self, formula, path):
        refs, times = self.references(formula, path)
        try: tree = ast.parse(_rewrite(formula.get('formula', '')), mode='eval')
        except SyntaxError as exc: raise ValueError('FORMULA_PARSE:' + path + ':' + formula.get('formula', '') + ':' + _rewrite(formula.get('formula', ''))) from exc
        _require(sum(1 for _ in ast.walk(tree)) <= 4096, 'FORMULA_AST_BOUND_EXCEEDED')
        def truth(value):
            if value in (sy.true, sy.false): return value
            if isinstance(value,(sy.core.relational.Relational,sy.logic.boolalg.BooleanFunction)):
                return value
            return sy.Ne(value, 0)
        def evaluate(node, guards=()):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
                return sy.true if node.value is True else sy.false if node.value is False else sy.Rational(str(node.value))
            if isinstance(node, ast.Name):
                if node.id in refs: return self.value(refs[node.id])
                if node.id in times: return sy.Symbol('Time', nonnegative=True, finite=True)
                if node.id.casefold() in {'true', 'false'}: return sy.true if node.id.casefold() == 'true' else sy.false
                _require(False, 'UNBOUND_FORMULA_ALIAS:' + node.id)
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand,guards)
                if isinstance(node.op, ast.USub): return -value
                if isinstance(node.op, ast.UAdd): return value
                if isinstance(node.op, ast.Not): return sy.Not(truth(value))
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left,guards), evaluate(node.right,guards)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div): self.denominator(right,guards); return left / right
                if isinstance(node.op, ast.Pow):
                    if right.is_negative is True: self.denominator(left,guards)
                    if right.is_integer is not True:self.domain(left,'positive','POWER_DOMAIN_UNRESOLVED',guards)
                    return left ** right
            if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
                left, right = evaluate(node.left,guards), evaluate(node.comparators[0],guards); op = node.ops[0]
                for kind, fn in [(ast.Eq, sy.Eq), (ast.NotEq, sy.Ne), (ast.Lt, sy.Lt), (ast.LtE, sy.Le), (ast.Gt, sy.Gt), (ast.GtE, sy.Ge)]:
                    if isinstance(op, kind): return fn(left, right)
            if isinstance(node, ast.BoolOp):
                return (sy.And if isinstance(node.op, ast.And) else sy.Or)(*(truth(evaluate(v,guards)) for v in node.values))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                name = node.func.id.casefold()
                if name == 'where' and len(node.args) == 3:
                    condition = truth(evaluate(node.args[0],guards))
                    if condition == sy.true: return evaluate(node.args[1],guards)
                    if condition == sy.false: return evaluate(node.args[2],guards)
                    return sy.Piecewise((evaluate(node.args[1],guards+(condition,)), condition), (evaluate(node.args[2],guards+(sy.Not(condition),)), True))
                args = [evaluate(a,guards) for a in node.args]
                if name in {'min', 'max'}: return (sy.Min if name == 'min' else sy.Max)(*args)
                if name in {'ln', 'log', 'log10'} and len(args) == 1:
                    self.domain(args[0],'positive','LOG_DOMAIN_UNRESOLVED',guards)
                    return sy.log(args[0], 10) if name == 'log10' else sy.log(args[0])
                if name in {'exp', 'sqrt', 'abs'} and len(args) == 1:
                    if name == 'sqrt': self.domain(args[0],'nonnegative','SQRT_DOMAIN_UNRESOLVED',guards)
                    return {'exp': sy.exp, 'sqrt': sy.sqrt, 'abs': sy.Abs}[name](args[0])
            _require(False, 'UNSUPPORTED_FORMULA_SYNTAX:' + ast.dump(node, include_attributes=False))
        value = evaluate(tree.body)
        _require(not value.has(sy.nan, sy.zoo, sy.oo, -sy.oo), 'NONFINITE_SYMBOLIC_FORMULA')
        return value



def preserved_urinary_events(document, metadata):
    """Prove original physiology resets without editing or suppressing events.

    Only constant-zero value assignments to physical urine amounts qualify.
    Event timing and triggers remain exactly as authored; no dose is inferred.
    """
    pattern=r'Events\|Urinary bladder emptying\|Urinary bladder emptying_[1-9][0-9]*\|Start time'
    parameters=[r for r in metadata.get('event_parameters',[]) if re.fullmatch(pattern,r['path'])]
    if not parameters:return None
    index=association.ModelIndex(document);assignments=[];groups=[]
    for row in parameters:
        _require(row.get('dimension')=='Time' and row.get('id')==index.paths[row['path']].get('id'),
                 'URINARY_TIME_PARAMETER_IDENTITY_DIFFERS')
        prefix=row['path'].rsplit('|',1)[0];group=index.paths.get(prefix)
        _require(group is not None and group.tag in {'Container','EventGroup'}
                 and group.get('containerType')=='EventGroup','EXACT_URINARY_PHYSIOLOGY_GROUP_REQUIRED')
        _require(not any(a['event_prefix']==prefix for a in metadata.get('administrations',[])),
                 'URINARY_NAME_CANNOT_HIDE_DRUG_ADMINISTRATION')
        _require(not any(n.get('name') in {'Dose','Infusion time','ProtocolSchemaItem'} for n in group.iter()),
                 'URINARY_EVENT_CANNOT_CONTAIN_ADMINISTRATION')
        events=[n for n in group.iter('Event')]
        _require(len(events)==1,'EXACT_ORIGINAL_URINARY_EVENT_REQUIRED')
        event=events[0];trigger=index.formulas.get(event.get('formula'))
        _require(trigger is not None and trigger.tag=='Formula','EXPLICIT_URINARY_TRIGGER_REQUIRED')
        writers=list(group.iter('EventAssignment'));_require(writers,'URINARY_RESET_ASSIGNMENTS_REQUIRED')
        for writer in writers:
            target=index.ids.get(writer.get('changedEntity'));node=index.paths.get(target)
            formula=index.formulas.get(writer.get('formula'))
            _require(target is not None and target.startswith('Organism|Kidney|Urine|')
                     and target.count('|')==3 and node is not None and node.tag=='MoleculeAmount'
                     and node.get('dim')=='Amount','URINARY_RESET_MUST_TARGET_PHYSICAL_URINE_AMOUNT')
            _require(writer.get('useAsValue')=='1' and formula is not None and formula.tag=='ConstantFormula'
                     and not list(formula) and formula.get('value') is not None
                     and sy.Rational(formula.get('value'))==0,'URINARY_RESET_MUST_BE_EXPLICIT_ZERO_VALUE')
            assignments.append({'path':index.node_paths[id(writer)],'changed_entity':target,
                'assignment_sha256':source_pkml.element_sha(writer),'formula_sha256':source_pkml.element_sha(formula),
                'target_id':node.get('id'),'use_as_value':True,'exact_value':'0'})
        groups.append({'path':prefix,'group_sha256':source_pkml.element_sha(group),
                       'trigger_sha256':source_pkml.element_sha(trigger)})
    for ref in URINARY_EVENT_EVIDENCE:
        _require(source_pkml.sha(ROOT/ref['path'])==ref['sha256'],'URINARY_NATIVE_ASSIGNMENT_EVIDENCE_CHANGED')
    return {'verified':True,'simulation_id':document.get('id'),'preserved_event_parameters':deepcopy(parameters),
            'original_event_groups':groups,'zero_reset_assignments':assignments,
            'physiology_events_modified':False,'evidence_refs':deepcopy(URINARY_EVENT_EVIDENCE)}


def species_closure(index, parents):
    """Follow typed original products, never drug-name substrings.

    Official MoleculesAndReactionsCreator.addSpecificBindingProcess creates
    drug-receptor products as Complex. Include those physical states while
    retaining every shared protein. Initial-zero, no-external-production,
    disjoint-target closure, flux and native-zero-monitor checks still apply.
    """
    def partner(name):
        return index.paths[index.ids[name]] if name in index.ids else None
    edges = []
    for path, node in index.paths.items():
        if node.tag != 'Reaction': continue
        educts = [partner(p.get('partner')) for p in node.findall('Educts/ReactionPartner')]
        products = [partner(p.get('partner')) for p in node.findall('Products/ReactionPartner')]
        _require(all(n is not None for n in educts + products), 'REACTION_PARTNER_ID_UNRESOLVED:' + path)
        edges.append((path, educts, products))
    names = set(parents)
    while True:
        extra = {n.get('name') for _, educts, products in edges if any(n.get('name') in names for n in educts)
                 for n in products if n.tag == 'MoleculeAmount' and n.get('quantityType') in {'Drug', 'Metabolite', 'Complex'}} - names
        if not extra: break
        names.update(extra)
    return names, edges


def retained_parent_set(target):
    """Only the existing single parent or the explicitly registered R/S pair."""
    if isinstance(target, str) and target:
        return {target}
    _require(isinstance(target, list) and target == ['Esomeprazole', 'R-omeprazole'],
             'EXACT_REGISTERED_RACEMATE_RETAINED_PARENT_SET_REQUIRED')
    return set(target)


# Cache only the pure symbolic result; every source/metadata/event/lock check
# in prepare_source_isolation and its callers still executes on every call.
_ZERO_PROOF_CACHE = OrderedDict()
_ZERO_PROOF_CACHE_LOCK = threading.RLock()
_ZERO_PROOF_CACHE_MAX_ENTRIES = 128
_ZERO_PROOF_CACHE_MAX_BYTES = 64 * 1024**2
_ZERO_PROOF_CACHE_BYTES = 0


def _proof_implementation_hashes():
    loaded = {
        'scripts/recommendation_execution_repairs/coadministration_isolation.py': getattr(globals().get('__spec__'), 'origin', None) or __file__,
        'scripts/recommendation_execution_repairs/source_pkml.py': getattr(source_pkml.__spec__, 'origin', None) or source_pkml.__file__,
        'scripts/recommendation_execution_repairs/event_associations.py': getattr(association.__spec__, 'origin', None) or association.__file__,
        'scripts/recommendation_execution_repairs/surplus_event_suppression.py': getattr(surplus.__spec__, 'origin', None) or surplus.__file__,
    }
    return tuple((name, source_pkml.sha(Path(loaded.get(name, source_pkml.ROOT / name))))
                 for name in IMPLEMENTATION_FILES)


def _zero_proof_cache_key(document, metadata, target):
    return (hashlib.sha256(ET.tostring(document, encoding='utf-8')).hexdigest(),
            _digest(metadata), _digest({'target': target, 'retained_parents': sorted(retained_parent_set(target))}),
            _proof_implementation_hashes(), sy.__version__,
            tuple((r['path'],source_pkml.sha(ROOT/r['path'])) for r in DISCRETE_DISTRIBUTION_EVIDENCE+URINARY_EVENT_EVIDENCE))


def prove_zero_manifold(document, metadata, target):
    """Content-addressed reuse of successful pure proofs, never of input checks."""
    global _ZERO_PROOF_CACHE_BYTES
    key = _zero_proof_cache_key(document, metadata, target)
    with _ZERO_PROOF_CACHE_LOCK:
        cached = _ZERO_PROOF_CACHE.get(key)
        if cached is not None:
            result = deepcopy(cached[0])
            _ZERO_PROOF_CACHE.move_to_end(key)
        else:
            result = None
    if result is not None and key == _zero_proof_cache_key(document, metadata, target):
        return result
    # Exceptions and unverified results are never cached.
    result = _prove_zero_manifold_uncached(document, metadata, target)
    if result.get('verified') is True and key == _zero_proof_cache_key(document, metadata, target):
        size = len(json.dumps(result, sort_keys=True, ensure_ascii=False).encode())
        if size <= _ZERO_PROOF_CACHE_MAX_BYTES:
            with _ZERO_PROOF_CACHE_LOCK:
                previous = _ZERO_PROOF_CACHE.pop(key, None)
                if previous is not None:
                    _ZERO_PROOF_CACHE_BYTES -= previous[1]
                _ZERO_PROOF_CACHE[key] = (deepcopy(result), size)
                _ZERO_PROOF_CACHE_BYTES += size
                while (len(_ZERO_PROOF_CACHE) > _ZERO_PROOF_CACHE_MAX_ENTRIES
                       or _ZERO_PROOF_CACHE_BYTES > _ZERO_PROOF_CACHE_MAX_BYTES):
                    _, (_, discarded_bytes) = _ZERO_PROOF_CACHE.popitem(last=False)
                    _ZERO_PROOF_CACHE_BYTES -= discarded_bytes
    return result


def _prove_zero_manifold_uncached(document, metadata, target):
    index = association.ModelIndex(document)
    _require(all(a['compound_association_complete'] and len(a['administered_compounds']) == 1 for a in metadata['administrations']), 'ALL_ADMINISTRATION_IDENTITIES_REQUIRED')
    administered = {a['administered_compounds'][0] for a in metadata['administrations']}
    kept_parents = retained_parent_set(target)
    _require(kept_parents < administered, 'TARGET_AND_OTHER_ADMINISTERED_PARENT_REQUIRED')
    other_parents = administered - kept_parents
    removed, edges = species_closure(index, other_parents)
    retained, _ = species_closure(index, kept_parents)
    _require(not removed & retained, 'TARGET_AND_REMOVED_REACTION_PRODUCT_CLOSURES_OVERLAP')
    for path, educts, products in edges:
        if any(n.get('name') in removed for n in products):
            _require(any(n.get('name') in removed for n in educts), 'REMOVED_SPECIES_HAS_EXTERNAL_PRODUCTION:' + path)
    applications = [a for a in metadata['administrations'] if a['administered_compounds'][0] in other_parents]
    dose_paths = {a['event_prefix'] + '|ProtocolSchemaItem|Dose' for a in applications}
    for path in dose_paths:
        _require(path in index.paths and index.paths[path].get('dim') == 'Mass', 'EXACT_OTHER_DRUG_DOSE_REQUIRED')
    states = {path: node for path, node in index.paths.items() if node.tag == 'MoleculeAmount' and node.get('name') in removed}
    # Drug-specific cumulative amounts are additional zero states, never
    # independent external inputs into the removed physical mass system.
    auxiliary = {path: node for path, node in index.paths.items() if node.get('rhs', node.get('rHSFormula'))
                 and node.get('dim') in {'Amount', 'Mass'} and set(path.split('|')) & removed}
    # Event-written zero input states (e.g. IV Application rate) are zero
    # because every original writer is protected by the independently proven
    # future-time barrier, not because their formulas are discarded.
    event_inputs = {}
    for admin in applications:
        prefix = admin['event_prefix']; start = index.paths[prefix + '|ProtocolSchemaItem|Start time']
        surplus.prove_application_barrier(index.model, prefix, start.get('id'))
        for path, node in index.paths.items():
            if not path.startswith(prefix + '|') or node.tag not in {'Para','Parameter'}: continue
            writers = [n for n in index.model.iter('EventAssignment') if n.get('changedEntity') == node.get('id')]
            if not writers: continue
            _require(all(index.node_paths[id(n)].startswith(prefix + '|') for n in writers), 'EXTERNAL_INPUT_STATE_WRITER:' + path)
            _require(node.get('dim') in {'Amount','Mass','Amount per time'} or
                     (not node.get('dim') and node.get('name')=='TabletIsActive'),
                     'UNSUPPORTED_INPUT_STATE_DIMENSION:' + path)
            event_inputs[path] = node
    zeros = {**states, **auxiliary, **event_inputs}; _require(states, 'REMOVED_PHYSICAL_STATES_REQUIRED')
    proof = FormulaProof(index, zeros, dose_paths)
    initial_failures = []
    for path, node in zeros.items():
        try:
            value = proof.initial(node)
            if sy.simplify(value) != 0: initial_failures.append({'path': path, 'expression': str(value)})
        except (ValueError, SyntaxError, TypeError) as exc: initial_failures.append({'path': path, 'reason': str(exc)})
    _require(not initial_failures, 'INITIAL_MASS_NOT_PROVEN_ZERO:' + json.dumps(initial_failures[:4], ensure_ascii=False))
    failures = []; fluxes = []; rhs = []
    for path, node in index.paths.items():
        attached = []
        if node.tag == 'Transport': attached = [node.get('sourceAmount'), node.get('targetAmount')]
        elif node.tag == 'Reaction': attached = [p.get('partner') for p in [*node.findall('Educts/ReactionPartner'), *node.findall('Products/ReactionPartner')]]
        if not any(index.ids.get(identity) in states for identity in attached): continue
        formula = index.formulas.get(node.get('formula'))
        try:
            _require(formula is not None and formula.tag == 'Formula', 'EXPLICIT_FLUX_FORMULA_REQUIRED')
            value = proof.expression(formula, path)
            _require(sy.simplify(value) == 0, 'FLUX_NOT_ZERO:' + str(value))
            fluxes.append({'path': path, 'id': node.get('id'), 'formula_sha256': source_pkml.element_sha(formula)})
        except (ValueError, SyntaxError, TypeError) as exc: failures.append({'path': path, 'reason': str(exc)})
    for path, node in auxiliary.items():
        formula = index.formulas.get(node.get('rhs', node.get('rHSFormula')))
        try:
            _require(formula is not None and formula.tag == 'Formula', 'EXPLICIT_RHS_REQUIRED')
            value = proof.expression(formula, path)
            _require(sy.simplify(value) == 0, 'AUXILIARY_RHS_NOT_ZERO:' + str(value) + ':' + str({str(q):proof.dynamic_symbols.get(q,proof.symbol_paths.get(q)) for q in value.free_symbols}))
            rhs.append({'path': path, 'formula_sha256': source_pkml.element_sha(formula)})
        except (ValueError, SyntaxError, TypeError) as exc: failures.append({'path': path, 'reason': str(exc)})
    _require(not failures, 'ZERO_MANIFOLD_NOT_PROVEN:' + json.dumps(failures[:8], ensure_ascii=False))
    prefixes = [a['event_prefix'] for a in applications]
    physiology=preserved_urinary_events(document,metadata)
    resets={r['path']:r for r in (physiology or {}).get('zero_reset_assignments',[])}
    assignments = []
    for path, node in index.paths.items():
        if node.tag != 'EventAssignment' or index.ids.get(node.get('changedEntity')) not in zeros: continue
        if not any(path.startswith(prefix + '|') for prefix in prefixes):
            reset=resets.get(path)
            _require(reset is not None and reset['changed_entity']==index.ids[node.get('changedEntity')],
                     'REMOVED_MASS_HAS_EXTERNAL_EVENT_WRITER:' + path)
        assignments.append({'path': path, 'changed_entity': index.ids[node.get('changedEntity')]})
    requirements = [{'path': path, 'id': index.paths[path].get('id'),
                     'dimension': index.paths[path].get('dim', index.paths[path].get('dimension', '')) or 'Dimensionless',
                     'finite_required': True, 'nonnegative_required':proof.static_symbols[path].is_nonnegative is True,
                     'strictly_positive_required': path in proof.positive_paths}
                    for path in sorted(proof.static_symbols)]
    result = {'verified': True, 'target': target, 'removed_administered_parents': sorted(other_parents),
            'removed_reaction_product_closure': sorted(removed), 'retained_target_product_closure': sorted(retained),
            'no_target_to_removed_production': True, 'initial_mass_zero_after_only_registered_dose_zero': True,
            'zero_monitor_paths': sorted(zeros), 'zero_monitor_dimensions': {p:n.get('dim') or 'Dimensionless' for p,n in sorted(zeros.items())},
            'positive_domain_monitor_dimensions': {p:index.paths[p].get('dim') for p in sorted(proof.dynamic_positive_paths)}, 'physical_state_count': len(states), 'auxiliary_zero_state_count': len(auxiliary), 'disabled_application_input_state_count': len(event_inputs),
            'zero_flux_count': len(fluxes), 'zero_rhs_count': len(rhs),
            'flux_proof_sha256': _digest(fluxes), 'rhs_proof_sha256': _digest(rhs),
            'source_state_inventory_sha256': _digest([{'path': p, 'id': n.get('id')} for p, n in sorted(zeros.items())]),
            'event_writers': assignments, 'native_parameter_requirements': requirements,
            'proof_domain': 'finite solution of the unchanged original physiology; frozen static parameter domains independently checked after native patches',
            'sympy_version': sy.__version__, 'all_shared_mechanisms_retained': True,
            'scientific_direct_validity_established': False}
    if proof.discrete_expansions:
        result['static_discrete_parameter_expansions']=[proof.discrete_expansions[p] for p in sorted(proof.discrete_expansions)]
        result['evidence_refs']=deepcopy(DISCRETE_DISTRIBUTION_EVIDENCE)
    if physiology:
        result['preserved_urinary_physiology']=physiology
        result['evidence_refs']=result.get('evidence_refs',[])+deepcopy(URINARY_EVENT_EVIDENCE)
    return result


def verify_discrete_parameter_readback(native, proof):
    """Independently verify native values used for exact static expansion."""
    expansions=proof.get('static_discrete_parameter_expansions',[])
    if not expansions:return []
    _require(all(r in proof.get('evidence_refs',[]) for r in DISCRETE_DISTRIBUTION_EVIDENCE),'DISCRETE_PROOF_EVIDENCE_DIFFERS')
    for ref in DISCRETE_DISTRIBUTION_EVIDENCE:
        _require(source_pkml.sha(ROOT/ref['path'])==ref['sha256'],'OFFICIAL_DISCRETE_FORMULA_EVIDENCE_CHANGED')
    rows=native.get('proof_parameter_readback',[])
    result=[]
    for item in expansions:
        matches=[r for r in rows if r.get('path')==item['path']]
        _require(len(matches)==1,'EXACT_DISCRETE_NATIVE_READBACK_MISSING')
        row=matches[0];unit=row.get('unit')
        _require(unit in ('','1','%') and row.get('dimension') in ('Dimensionless','Fraction')
                 and row.get('id')==item['id'] and type(row.get('value')) in (int,float)
                 and math.isfinite(row['value']),'EXACT_DISCRETE_NATIVE_IDENTITY_OR_UNIT_DIFFERS')
        exact=sy.Rational(item['exact_rational_value'])*(100 if unit=='%' else 1)
        _require(row['value']==float(exact),'EXACT_DISCRETE_NATIVE_VALUE_DIFFERS')
        result.append({'path':item['path'],'native_value':row['value'],'native_unit':unit,
                       'exact_rational_base_value':item['exact_rational_value'],'verified':True})
    return result


def target_metadata(metadata, target):
    """Selection view only; the caller must retain the full original metadata."""
    result = deepcopy(metadata)
    kept_parents = retained_parent_set(target)
    target_admins = [a for a in metadata['administrations'] if len(a.get('administered_compounds', [])) == 1
        and a['administered_compounds'][0] in kept_parents]
    _require({a['administered_compounds'][0] for a in target_admins} == kept_parents, 'ALL_RETAINED_PARENT_ADMINISTRATIONS_REQUIRED')
    _require(target_admins, 'NO_EXACT_TARGET_ADMINISTRATIONS')
    prefixes = {a['event_prefix'] + '|ProtocolSchemaItem|' for a in target_admins}
    result['administrations'] = target_admins
    result['event_parameters'] = [row for row in metadata['event_parameters'] if any(row['path'].startswith(p) for p in prefixes)
                                  or '|Standard Meal|' in row['path']]
    # Do not replace compounds, physiology, or formulation metadata by guesses.
    return result


def prepare_source_isolation(source_path, source_sha256, simulation_id, source_blob_sha256,
                             metadata, target, contract, *, project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    source, document = association._load(source_path, source_sha256, simulation_id, source_blob_sha256, root)
    return _document_policy(source,source_sha256,simulation_id,source_blob_sha256,document,metadata,target,contract,root)


def _document_policy(source,source_sha256,simulation_id,source_blob_sha256,document,metadata,target,contract,root,*,original_document=None):
    metadata=_model_metadata(metadata)
    actual, association_proof = association._enrich_document(document,original_document=original_document)
    # Metadata may include the source inventory envelope, but original fields
    # must match independently re-read/enriched source metadata exactly.
    _require(all(metadata.get(k) == value for k, value in actual.items()), 'SOURCE_METADATA_BINDING_DIFFERS')
    proof = prove_zero_manifold(document, actual, target)
    index = association.ModelIndex(document); model = document.find('Model')
    settings=source_pkml.saved_settings(document)
    _require(model is not None, 'MODEL_AND_SETTINGS_REQUIRED')
    end = contract['absolute_sampling_times_min'][-1]
    _require(type(end) in (int,float) and math.isfinite(end) and end >= 0, 'FROZEN_OUTPUT_END_REQUIRED')
    events = []
    for admin in sorted(actual['administrations'], key=lambda a:a['event_prefix']):
        if len(admin['administered_compounds']) == 1 and admin['administered_compounds'][0] in retained_parent_set(target): continue
        prefix = admin['event_prefix']; start_path = prefix + '|ProtocolSchemaItem|Start time'
        dose_path = prefix + '|ProtocolSchemaItem|Dose'
        _require(start_path in index.paths and dose_path in index.paths, 'OTHER_PARENT_DOSE_START_REQUIRED')
        start, dose = index.paths[start_path], index.paths[dose_path]
        _require(dose.get('dim') == 'Mass' and start.get('dim') == 'Time', 'EXACT_ADMINISTRATION_DIMENSIONS_REQUIRED')
        future = end + 60 + len(events)
        _require(math.isfinite(future) and future > end, 'FINITE_FUTURE_START_REQUIRED')
        events.append({'event_prefix': surplus._normalize(prefix), 'source_event_prefix': prefix,
                       'administered_compound': admin['administered_compounds'][0],
                       'dose_parameter_id': dose.get('id'), 'dose_parameter_path': surplus._normalize(dose_path),
                       'start_parameter_id': start.get('id'), 'start_parameter_path': surplus._normalize(start_path),
                       'new_start_time_min': future, 'dose_mg': 0.0,
                       'barrier': surplus.prove_application_barrier(model, prefix, start.get('id'))})
    _require(events, 'OTHER_PARENT_EVENTS_REQUIRED')
    _require(source_pkml.sha(source) == source_sha256, 'SOURCE_CHANGED_DURING_PROOF')
    target_view = target_metadata(actual, target)
    return {'version': VERSION, 'verified': True, 'reference_adaptation_only': True,
            'source_reference': {'path': str(source.relative_to(root)), 'sha256': source_sha256,
                                 'simulation_id': simulation_id, 'source_blob_sha256': source_blob_sha256},
            'source_model_sha256': source_pkml.element_sha(model),
            'source_settings_sha256': source_pkml.element_sha(settings),
            'source_metadata_sha256': _digest(metadata), 'target_selection_metadata_sha256': _digest(target_metadata(metadata,target)),
            'source_event_count': len(actual['administrations']), 'source_target_event_count': len(target_view['administrations']),
            'output_end_min': end, 'contract_sha256': _digest(contract), 'target': target,
            'events': events, 'zero_manifold_proof': proof,
            'implementation_sha256': {p: source_pkml.sha(source_pkml.inside(p, root)) for p in IMPLEMENTATION_FILES},
            'limits': ['Conditional zero invariance on the original model finite positive-volume domain.',
                       'Dynamic volume positivity and zero mass are verified on every registered output point, not every internal solver step.',
                       'All shared kinetics, physiology, initial states and target metabolites remain unchanged.',
                       'No claim of clinical direct applicability or predictive accuracy.'],
            'sealed_observations_accessed': False, 'ode_run': False, 'scientific_validity_established': False}


def verify_source_isolation(policy, metadata, contract, *, project_root=ROOT):
    ref = policy['source_reference']
    current = prepare_source_isolation(ref['path'], ref['sha256'], ref['simulation_id'],
        ref['source_blob_sha256'], metadata, policy['target'], contract, project_root=project_root)
    _require(current == policy, 'FROZEN_SOURCE_POLICY_DIFFERS')
    return current


def prepare_expanded_isolation(policy,receipt_path,metadata,contract,*,project_root=ROOT):
    """Reprove zero invariance on the independently audited expanded model."""
    from . import mixed_application_expansion as mixed
    root=Path(project_root).resolve();ref=policy['source_reference']
    verification=mixed.verify_expansion_receipt(receipt_path,project_root=root)
    source,document=association._load(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],root)
    receipt=json.loads(source_pkml.inside(receipt_path,root).read_text())
    _require(receipt['source_reference']==ref,'EXPANSION_ORIGINAL_SOURCE_DIFFERS')
    original_document=deepcopy(document)
    expanded=mixed.base._read_xml(source_pkml.inside(receipt['pkml']['path'],root))
    document.remove(document.find('Model'));document.append(deepcopy(mixed.base._models(expanded)))
    raw=source_pkml.describe_saved_simulation(document)
    _require(all(metadata.get(k)==v for k,v in raw.items()),'EXPANSION_METADATA_DIFFERS')
    enriched,_=association._enrich_document(document,original_document=original_document)
    actual={**metadata,**enriched}
    result=_document_policy(source,ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
        document,actual,policy['target'],contract,root,original_document=original_document)
    result.update(model_origin='audited_expanded_executable_model',
        original_isolation_policy_sha256=_digest(policy),mixed_expansion_receipt=verification['receipt'])
    return result,actual


def monitor_request(policy):
    if policy is None: return {}
    proof=policy['zero_manifold_proof']
    return {'zero_monitor_paths':proof['zero_monitor_paths'],
            'zero_monitor_dimensions':proof['zero_monitor_dimensions'],
            'proof_parameter_requirements':proof['native_parameter_requirements'],
            'positive_domain_monitor_dimensions':proof['positive_domain_monitor_dimensions']}


def build_isolated_patches(inventory,contract,metadata,policy,*,target_patch_builder,suppression_policy=None):
    from .native_pkml import ADMINISTRATION
    metadata=_model_metadata(metadata)
    _require(policy.get('verified') is True and policy.get('source_metadata_sha256')==_digest(metadata)
             and policy.get('contract_sha256')==_digest(contract),'PATCH_POLICY_METADATA_OR_CONTRACT_DIFFERS')
    target=policy['target']; target_view=target_metadata(metadata,target)
    _require(_digest(target_view)==policy['target_selection_metadata_sha256'],'PATCH_TARGET_SELECTION_VIEW_DIFFERS')
    source_rows={surplus._normalize(r['path']):r for r in metadata['event_parameters'] if ADMINISTRATION.fullmatch(r['path'])}
    rows=[r for r in inventory.get('parameters',[]) if ADMINISTRATION.fullmatch(r.get('path',''))]
    actual={surplus._normalize(r['path']):r for r in rows}
    _require(len(source_rows)==len(rows)==len(actual) and set(source_rows)==set(actual),'FULL_NATIVE_ADMINISTRATION_SET_DIFFERS')
    for path,row in actual.items():
        _require(row.get('id')==source_rows[path]['id'],'FULL_NATIVE_ADMINISTRATION_ID_DIFFERS')
    target_prefixes={surplus._normalize(a['event_prefix']) for a in target_view['administrations']}
    filtered=deepcopy(inventory)
    filtered['parameters']=[r for r in inventory['parameters'] if not ADMINISTRATION.fullmatch(r.get('path',''))
                            or surplus._normalize(r['path'].rsplit('|ProtocolSchemaItem|',1)[0]) in target_prefixes]
    patches,events=target_patch_builder(filtered,contract,target_view,native_compound_name=target,suppression_policy=suppression_policy)
    mass_units={'kg','g','mg','ug','µg'};time_units={'min','h','s','day(s)'}
    other_prefixes={e['event_prefix'] for e in policy['events']}
    all_prefixes={surplus._normalize(a['event_prefix']) for a in metadata['administrations']}
    _require(not target_prefixes & other_prefixes and all_prefixes==target_prefixes|other_prefixes,'UNACCOUNTED_SOURCE_ADMINISTRATION')
    for event in policy['events']:
        dose=actual[event['dose_parameter_path']];start=actual[event['start_parameter_path']]
        _require(dose['id']==event['dose_parameter_id'] and start['id']==event['start_parameter_id'],'OTHER_EVENT_IDS_DIFFER')
        for row,unit_set in [(dose,mass_units),(start,time_units)]:
            _require(row.get('unit') in unit_set and type(row.get('value')) in (int,float)
                     and math.isfinite(row['value']) and row['value']>=0,'OTHER_EVENT_UNITS_OR_VALUE_INVALID')
        for row,value,unit in [(dose,0.0,'mg'),(start,event['new_start_time_min'],'min')]:
            patches.append({'path':row['path'],'expected_before_value':row['value'],
                            'expected_before_unit':row['unit'],'value':value,'unit':unit})
        events.append({'event_prefix':event['event_prefix'],'target_administration_index':None,
                       'disposition':'other_parent_zero_dose_event_moved_beyond_output_window',
                       'administered_compound':event['administered_compound'], 'dose_mg':0.0,
                       'start_time_min':event['new_start_time_min'],'suppression_proof':event['barrier'],
                       'original_source_parameters':{k.rsplit('|',1)[-1]:v for k,v in actual.items()
                           if k.startswith(event['event_prefix']+'|ProtocolSchemaItem|')}})
    return patches,events
