"""Pure-Python scheduler checks; no model, HTTP, production files or GPU."""
from collections import deque
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import queue
import random
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bounded_batch_server', ROOT / 'scripts/deployment/evidence_qwen_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def task(identifier, tokens=512, length=100, **extra):
    return {'body': {'id': identifier, 'max_tokens': tokens, 'length': length,
                     'temperature': 0, 'repetition_penalty': 1.0, **extra}, 'future': Future()}


class FiniteQueue:
    def __init__(self, tasks):
        self.items, self.timeouts = deque(tasks), []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if self.items:
            return self.items.popleft()
        if timeout is None:
            raise StopIteration('finite test workload exhausted')
        raise queue.Empty


class FakeGenerator:
    def __init__(self):
        self.batches = []
        self.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    def render(self, body):
        if body.get('bad_render'):
            raise ValueError('authored rendering failure')
        return body['id'], body['length']

    def generate(self, tasks):
        self.batches.append([t['body']['id'] for t in tasks])
        assert (max(t['length'] for t in tasks) + tasks[0]['body']['max_tokens']) * len(tasks) <= server.TOKEN_BUDGET
        assert len({server.generation_group(t['body']) for t in tasks}) == 1
        if any(t['body'].get('bad_generate') for t in tasks):
            raise ValueError('authored generation failure')
        return [{'choices': [{'finish_reason': 'stop', 'message': {'content': t['body']['id']}}]}
                for t in tasks], {'batch_size': len(tasks)}


def run(tasks, size=2, lookahead=1):
    engine = server.Engine.__new__(server.Engine)
    engine.queue = FiniteQueue(tasks)
    engine.max_batch_size, engine.batch_lookahead = size, lookahead
    engine.state = dict(completed_requests=0, failed_requests=0, responded_requests=0,
                        truncated_requests=0, invalid_requests=0)
    engine.persist = lambda **values: engine.state.update(values)
    generator = FakeGenerator()
    with patch.object(server, 'EvidenceGenerator', return_value=generator):
        with pytest.raises(StopIteration, match='exhausted'):
            engine.work()
    return generator, engine


def assert_mapped(tasks):
    for t in tasks:
        assert t['future'].result()['choices'][0]['message']['content'] == t['body']['id']


def test_default_retains_adjacent_pair_and_does_not_search_past_mismatch():
    tasks = [task('a1'), task('b1', tokens=768), task('a2')]
    gen, _ = run(tasks)
    assert gen.batches == [['a1'], ['b1'], ['a2']]
    assert_mapped(tasks)
    pairs = [task(str(i)) for i in range(4)]
    assert run(pairs)[0].batches == [['0', '1'], ['2', '3']]
    assert_mapped(pairs)


def test_explicit_four_batch_and_identity_mapping():
    tasks = [task(str(i)) for i in range(4)]
    gen, _ = run(tasks, 4, 4)
    assert gen.batches == [['0', '1', '2', '3']]
    assert_mapped(tasks)


def test_lookahead_bypasses_mismatch_once_then_anchors_oldest_deferred():
    tasks = [task('a1'), task('b1', tokens=768), task('a2'), task('c1', tokens=2048), task('b2', tokens=768)]
    gen, _ = run(tasks, 2, 4)
    assert gen.batches == [['a1', 'a2'], ['b1', 'b2'], ['c1']]
    assert_mapped(tasks)


def test_padded_budget_is_recomputed_for_each_addition_and_long_readout_stays_single():
    tasks = [task('a', length=100), task('long', length=8000), task('b', length=3000), task('c', length=3000)]
    gen, _ = run(tasks, 4, 4)
    assert gen.batches == [['a', 'b', 'c'], ['long']]
    assert_mapped(tasks)
    readouts = [task(str(i), tokens=4096, length=5000) for i in range(4)]
    assert run(readouts, 4, 4)[0].batches == [[str(i)] for i in range(4)]
    assert_mapped(readouts)


def test_exact_budget_boundary_does_not_exceed_16384():
    tasks = [task(str(i), tokens=512, length=3584) for i in range(4)]
    assert run(tasks, 4, 4)[0].batches == [['0', '1', '2', '3']]
    longer = [task(str(i), tokens=512, length=3585) for i in range(4)]
    assert [len(b) for b in run(longer, 4, 4)[0].batches] == [3, 1]


def test_sampling_and_penalty_mismatches_never_share_generation():
    tasks = [task('a'), task('sample', temperature=.7), task('penalty', repetition_penalty=1.1), task('a2')]
    gen, _ = run(tasks, 4, 4)
    assert gen.batches == [['a', 'a2'], ['sample'], ['penalty']]
    assert_mapped(tasks)


def test_cancelled_and_bad_render_peers_do_not_drop_valid_requests():
    tasks = [task('a'), task('cancelled'), task('bad', bad_render=True), task('b')]
    tasks[1]['future'].cancel()
    gen, engine = run(tasks, 4, 4)
    assert gen.batches == [['a', 'b']]
    assert tasks[1]['future'].cancelled()
    assert isinstance(tasks[2]['future'].exception(), ValueError)
    assert engine.state['failed_requests'] == 1
    assert_mapped([tasks[0], tasks[3]])


def test_generation_failure_resolves_batch_futures_and_keeps_deferred_work():
    tasks = [task('bad', bad_generate=True), task('different', tokens=768), task('peer')]
    gen, engine = run(tasks, 2, 4)
    assert gen.batches == [['bad', 'peer'], ['different']]
    assert isinstance(tasks[0]['future'].exception(), ValueError)
    assert isinstance(tasks[2]['future'].exception(), ValueError)
    assert engine.state['failed_requests'] == 2
    assert_mapped([tasks[1]])


def test_many_mixed_requests_are_once_only_with_bounded_pending_and_no_starvation():
    rng = random.Random(17)
    tasks = [task(str(i), tokens=rng.choice([512, 768, 2048, 4096]), length=rng.choice([100, 3000, 5000])) for i in range(80)]
    incoming, pending, gen = FiniteQueue(tasks), deque(), FakeGenerator()
    seen = []
    while pending or incoming.items:
        oldest = pending[0] if pending else incoming.items[0]
        first = pending.popleft() if pending else incoming.get()
        batch, failures = server.collect_batch(first, pending, incoming, gen, max_batch_size=4, batch_lookahead=4)
        assert batch[0] is oldest
        assert len(pending) <= 4
        assert failures == 0
        outputs, _ = gen.generate(batch)
        for t, result in zip(batch, outputs, strict=True):
            t['future'].set_result(result)
            seen.append(t['body']['id'])
    assert sorted(seen, key=int) == [str(i) for i in range(80)]
    assert len(seen) == len(set(seen))
    assert_mapped(tasks)
    assert all(t is None or 0 <= t <= .08 for t in incoming.timeouts)


@pytest.mark.parametrize('size,lookahead', [(0,1),(5,1),(True,1),(2,0),(2,5),(2,True)])
def test_options_reject_invalid_caps(size, lookahead):
    with pytest.raises(ValueError):
        server.validate_batch_options(size, lookahead)


def test_cancelled_or_invalid_oldest_request_does_not_block_later_work():
    tasks = [task('cancelled'), task('bad', bad_render=True), task('ok')]
    tasks[0]['future'].cancel()
    gen, engine = run(tasks, 4, 4)
    assert gen.batches == [['ok']]
    assert tasks[0]['future'].cancelled()
    assert isinstance(tasks[1]['future'].exception(), ValueError)
    assert engine.state['failed_requests'] == 1
    assert_mapped(tasks[2:])


@pytest.mark.parametrize('options,expected', [([], (2,1)), (['--max-batch-size','4','--batch-lookahead','4'], (4,4))])
def test_cli_only_forwards_startup_options_without_starting_service(options, expected):
    with patch.object(server.sys, 'argv', ['server.py', 'start', *options]), patch.object(server, 'launch') as launch:
        server.main()
    launch.assert_called_once_with(max_batch_size=expected[0], batch_lookahead=expected[1])
