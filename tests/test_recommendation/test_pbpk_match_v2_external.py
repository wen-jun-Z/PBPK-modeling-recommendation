"""Small, offline contracts for pinned new rerankers; no full model or GPU."""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('pbpk_match_v2_external_tested',
    PROJECT / 'src/pbpk_kb/recommendation/pbpk_match_v2/external.py')
external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(external)


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    pad_token_id = 0
    pad_token = '<pad>'
    eos_token = '<eos>'
    padding_side = 'left'

    def encode(self, text, add_special_tokens=False):
        return [ord(x) + 1 for x in text]

    def __call__(self, text, *, truncation=False, max_length=None,
                 padding=False, return_tensors=None, **kwargs):
        many = isinstance(text, list)
        ids = [self.encode(x) for x in text] if many else [self.encode(text)]
        if truncation:
            ids = [x[:max_length] for x in ids]
        if padding or return_tensors:
            return self.pad([{'input_ids': x} for x in ids], return_tensors=return_tensors)
        return {'input_ids': ids if many else ids[0]}

    def prepare_for_model(self, first, second, max_length, **kwargs):
        return {'input_ids': first + second[:max(0, max_length-len(first))]}

    def pad(self, rows, padding=True, pad_to_multiple_of=None, return_tensors=None, **kwargs):
        maxlen = max(len(x['input_ids']) for x in rows)
        if pad_to_multiple_of:
            maxlen = ((maxlen + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        values, masks = [], []
        for row in rows:
            seq = row['input_ids']; n = maxlen - len(seq)
            values.append([0] * n + seq); masks.append([0] * n + [1] * len(seq))
        return Batch(input_ids=torch.tensor(values), attention_mask=torch.tensor(masks))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=2048)
        self.fail_above = None
        self.fail_always = False
        self.calls = []

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=1):
        self.calls.append(len(input_ids))
        if self.fail_always or (self.fail_above is not None and len(input_ids) > self.fail_above):
            raise torch.OutOfMemoryError('synthetic bounded OOM')
        score = (input_ids * attention_mask).sum(dim=1).float() / 1000
        logits = torch.zeros(len(input_ids), 1, 256)
        logits[:, 0, 0] = score
        logits[:, 0, ord('1') + 1] = score
        logits[:, 0, ord('0') + 1] = 0.25
        return types.SimpleNamespace(logits=logits)


@pytest.fixture
def fixture_model(tmp_path, monkeypatch):
    # Fake artifacts remain under the fixture project; neither hub nor CUDA runs.
    monkeypatch.setattr(external, 'ROOT', tmp_path)
    monkeypatch.setattr(external, 'WEIGHTS', tmp_path/'artifacts/models')
    monkeypatch.setattr(external, 'CACHE', tmp_path/'cache')
    base_module = external.ExternalModel.__init__.__globals__
    monkeypatch.setitem(base_module, 'ROOT', tmp_path)
    import transformers
    calls = []
    model = TinyModel()
    def load_tokenizer(path, **options):
        calls.append(('tokenizer', options)); return Tokenizer()
    def load_model(path, **options):
        calls.append(('model', options)); return model
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', load_model)
    def create(name, batch_size=17, max_length=700):
        path = tmp_path/name;path.mkdir(exist_ok=True)
        records = []
        for filename in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors']:
            f=path/filename;f.write_bytes(b'{}')
            records.append({'name':filename,'bytes':2,'sha256':external._sha(f)})
        receipt={**external.MODEL_SPECS[name], 'completed':True,'files':records}
        (path/'model_receipt.json').write_text(json.dumps(receipt))
        return external.create_adapter(name,path,device='cpu',batch_size=batch_size,
                                       max_length=max_length,instruction='Match the requested drug and route.')
    return create, model, calls


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_all_280_scores_complete_ordered_float32_and_local_only(fixture_model, name):
    create, model, calls=fixture_model
    adapter=create(name)
    docs=[f'Drug candidate {i} with oral input.' for i in range(280)]
    values=adapter.score_query('PBPK prediction', docs)
    assert values.shape==(280,) and values.dtype==np.float32 and np.isfinite(values).all()
    adapter.batch_size=1
    reference=adapter.score_query('PBPK prediction', docs)
    np.testing.assert_array_equal(values,reference)
    assert sum(model.calls[:17])==280
    for _, options in calls:
        assert options['local_files_only'] is True
        assert options['token'] is False and options['trust_remote_code'] is False


def test_prorank_exact_official_template_and_single_binary_tokens(fixture_model):
    create,_,_=fixture_model
    adapter=create('prorank',max_length=1024)
    inputs=adapter._prepare_inputs('PK?', ['drug A'])
    seq=inputs['input_ids'][0][inputs['attention_mask'][0].bool()].tolist()
    expected=(adapter.PREFIX+'instruction: Match the requested drug and route.\nquery: PK?'
              +'\ndocument: drug A\n'+adapter.TASK+adapter.SUFFIX)
    assert seq==adapter.tokenizer.encode(expected)
    assert adapter.yes_id==ord('1')+1 and adapter.no_id==ord('0')+1


def test_contextual_exact_pinned_official_prompt(fixture_model):
    create,_,_=fixture_model
    adapter=create('contextual')
    assert adapter.format_prompt('PK?', 'drug A') == (
        'Check whether a given document contains information helpful to answer the query.\n'
        '<Document> drug A\n<Query> PK? Match the requested drug and route. ??')
    assert adapter.tokenizer.padding_side=='left'


def test_prorank_subtracts_in_native_precision_not_float32(fixture_model):
    create,_,_=fixture_model
    adapter=create('prorank')
    logits=torch.zeros(1,256,dtype=torch.bfloat16)
    logits[0,adapter.yes_id]=1.0; logits[0,adapter.no_id]=0.003
    expected=(logits[:,adapter.yes_id]-logits[:,adapter.no_id]).float()
    wrong=logits[:,adapter.yes_id].float()-logits[:,adapter.no_id].float()
    assert not torch.equal(expected,wrong)
    assert torch.equal(adapter._extract_scores(logits),expected)


def test_contextual_uses_vocab_zero_and_official_bfloat16_cast(fixture_model):
    create,_,_=fixture_model
    adapter=create('contextual')
    logits=torch.tensor([[1.0031,999.0,-300.0]],dtype=torch.float32)
    assert torch.equal(adapter._extract_scores(logits),logits[:,0].bfloat16().float())


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_bounded_oom_retries_restart_without_losing_candidates(fixture_model,name):
    create,model,_=fixture_model
    adapter=create(name,batch_size=16)
    model.fail_above=4
    docs=[f'candidate {i}' for i in range(280)]
    actual=adapter.score_query('query',docs)
    assert adapter.batch_size==4 and adapter.oom_retries==2
    model.fail_above=None;adapter.batch_size=1
    np.testing.assert_array_equal(actual,adapter.score_query('query',docs))


def test_oom_at_one_raises_without_partial_output(fixture_model):
    create,model,_=fixture_model
    adapter=create('prorank',batch_size=4);model.fail_always=True
    with pytest.raises(torch.OutOfMemoryError):adapter.score_query('query',[f'doc {i}' for i in range(280)])
    assert model.calls==[4,2,1] and adapter.oom_retries==2


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_finite_guard_rejects_bad_outputs(fixture_model,name):
    create,model,_=fixture_model;adapter=create(name)
    original=model.forward
    def bad(*a,**kw):
        output=original(*a,**kw);output.logits[:]=float('nan');return output
    model.forward=bad
    with pytest.raises(RuntimeError,match='finite score'):adapter.score_query('query',['doc']*280)


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_truncation_count_and_original_context_bound(fixture_model,name):
    create,_,_=fixture_model;adapter=create(name,max_length=700)
    inp=adapter._prepare_inputs('query',['d'*10000,'short'])
    assert adapter.truncated_input_count==1 and adapter.tokenized_input_count==2
    assert inp['input_ids'].shape[1]<=adapter.model_max_length
    assert inp['attention_mask'][:,-1].all()


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_bad_text_empty_pool_and_unknown_model_rejected(fixture_model,name):
    create,_,_=fixture_model;adapter=create(name)
    with pytest.raises(ValueError):adapter.score_query('', ['doc'])
    with pytest.raises(ValueError):adapter.score_query('query', [])
    with pytest.raises(ValueError):adapter.score_query('query', [''])
    with pytest.raises(ValueError):external.ensure_weights('unregistered-model')


@pytest.mark.parametrize('change',['revision','license','weight','duplicate','outside'])
def test_weight_identity_and_tamper_rejection(fixture_model,change,tmp_path):
    create,_,_=fixture_model;adapter=create('prorank')
    receipt_path=adapter.model_path/'model_receipt.json';receipt=json.loads(receipt_path.read_text())
    if change in {'revision','license'}:receipt[change]='wrong'
    elif change=='weight':(adapter.model_path/'model.safetensors').write_bytes(b'xx')
    elif change=='duplicate':receipt['files'].append(receipt['files'][0])
    else:receipt['files'][0]['name']='../config.json'
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError,FileNotFoundError)):external.verify_weights('prorank',adapter.model_path)


def test_download_disk_guard_precedes_file_transfer(tmp_path,monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(external,'ROOT',tmp_path)
    monkeypatch.setattr(external,'WEIGHTS',tmp_path/'models')
    monkeypatch.setattr(external,'CACHE',tmp_path/'cache')
    spec=external.MODEL_SPECS['prorank']
    item=types.SimpleNamespace(rfilename='model.safetensors',size=100)
    info=types.SimpleNamespace(sha=spec['revision'],card_data={'license':spec['license']},siblings=[item])
    api=types.SimpleNamespace(model_info=lambda *a,**kw: info)
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda token:api if token is False else None)
    monkeypatch.setattr(external.shutil,'disk_usage',lambda _:types.SimpleNamespace(free=50))
    def forbidden(*a,**kw):raise AssertionError('Download must not start')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',forbidden)
    with pytest.raises(RuntimeError,match='filesystem reserve'):external.ensure_weights('prorank')


@pytest.mark.parametrize('name',['prorank','contextual'])
def test_duplicate_input_ties_and_candidate_permutation_are_exact(fixture_model,name):
    create,model,_=fixture_model;adapter=create(name,batch_size=4)
    docs=['candidate A','candidate B','candidate A','candidate C']*70
    values=adapter.score_query('query',docs)
    assert model.calls==[3]
    perm=np.random.default_rng(20261004).permutation(len(docs))
    reverse=adapter.score_query('query',[docs[i] for i in perm])
    np.testing.assert_array_equal(values[perm],reverse)
    assert len(set(values[::4]))==1
