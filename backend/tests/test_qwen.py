"""Exercise real transformer math with a tiny random model; no downloads or GPU."""
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import Qwen3Config, Qwen3ForCausalLM
from backend import qwen


class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        return [3, 4, 5]
    def encode(self, text, **kwargs):
        return [3, 4, 5]
    def decode(self, ids, **kwargs):
        return ''.join(chr(65 + i) for i in ids)
    def convert_ids_to_tokens(self, i):
        return chr(65 + i)


@pytest.fixture()
def client(monkeypatch):
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    config._attn_implementation = 'eager'
    torch.manual_seed(1)
    monkeypatch.setattr(qwen, 'model', Qwen3ForCausalLM(config).eval())
    monkeypatch.setattr(qwen, 'tokenizer', Tokenizer())
    monkeypatch.setattr(qwen, 'DEVICE', 'cpu')
    return TestClient(qwen.app)


def test_measured_attention_and_preserved_ids(client):
    result = client.post('/inspect', json={'token_ids': [9, 7, 8], 'layer': 1, 'head': 1})
    assert result.status_code == 200
    data = result.json()
    assert [t['id'] for t in data['tokens']] == [9, 7, 8]
    for i, row in enumerate(data['attention']):
        assert sum(row) == pytest.approx(1, abs=1e-6)
        assert all(p == 0 for p in row[i+1:])
    assert sum(t['probability'] for t in data['alternatives']) + data['other_probability'] == pytest.approx(1, abs=1e-6)
    assert len(data['embedding']) == 3
    assert data['embedding'] != data['hidden']
    altered = client.post('/inspect', json={'token_ids': [9, 7, 6], 'layer': 1, 'head': 1}).json()
    assert data['hidden'][:2] == altered['hidden'][:2]


@pytest.mark.parametrize('body', [{'token_ids': [-1]}, {'token_ids': [32]}, {'head': 2}, {'layer': 2}, {'token_ids': [3]*97}, {'temperature': 0}])
def test_invalid_input(client, body):
    assert client.post('/inspect', json=body).status_code == 422


def test_busy_and_health(client):
    with qwen.lock:
        assert client.post('/inspect', json={}).status_code == 429
        assert client.post('/reply', json={'text': 'Hi'}).status_code == 429
    assert client.get('/health').json()['loaded']
    assert client.get('/').status_code == 200


def test_weight_is_restored_and_stages_are_measured(client):
    original = qwen.model.model.layers[0].self_attn.q_proj.weight[0, 0].item()
    r = client.post('/inspect', json={'query_delta': 1}).json()
    assert qwen.model.model.layers[0].self_attn.q_proj.weight[0, 0].item() == original
    assert len(r['all_heads']) == 2
    assert set(r['stages']) == {'attention_update', 'after_attention', 'mlp_update'}
    for d in r['temperature_comparison']:
        assert sum(d['counts']) + d['other_count'] == 100
        assert sum(t['probability'] for t in d['alternatives']) + d['other_probability'] == pytest.approx(1, abs=1e-6)


def test_restore_even_when_forward_fails(client, monkeypatch):
    block = qwen.model.model.layers[0]
    original = block.self_attn.q_proj.weight[0, 0].item()
    def fail(*a, **kw):
        raise RuntimeError('test forward failure')
    monkeypatch.setattr(qwen.model, 'forward', fail)
    with pytest.raises(RuntimeError):
        client.post('/inspect', json={'query_delta': 1})
    assert block.self_attn.q_proj.weight[0, 0].item() == original
    assert not block.self_attn._forward_hooks
    assert not block.mlp._forward_hooks
    assert not qwen.lock.locked()


def test_full_lesson_routes(client):
    r = client.post('/lab/run', json={'prompt': 'The cat', 'length': 3, 'mode': 'greedy'})
    assert r.status_code == 200
    d = r.json()
    assert len(d['positions']) >= 1
    assert d['input_ids'] == [3, 4, 5]
    for p in d['positions']:
        assert p['id'] == p['alternatives'][0]['id']
        assert sum(t['probability'] for t in p['alternatives']) + p['other_probability'] == pytest.approx(1, abs=1e-6)
    r = client.post('/next-token', json={'prompt': 'The cat', 'corpus': 'news'})
    assert r.status_code == 200 and r.json()['tokens']
    r = client.post('/chat', json={'messages': [{'role': 'user', 'content': 'Hi'}], 'corpus': 'academic'})
    assert r.status_code == 200 and r.json()['positions']
    assert 'lesson-guide.js' in client.get('/').text
