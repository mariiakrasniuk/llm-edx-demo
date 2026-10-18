import math
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AuthenticationError, RateLimitError
from backend.main import app, SYSTEM_PROMPTS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(app) as client:
        yield client


def fake_api():
    token = NS(token=' quiet', bytes=[32, 113, 117, 105, 101, 116], logprob=math.log(0.4),
               top_logprobs=[NS(token=' dark', bytes=[32, 100, 97, 114, 107], logprob=math.log(0.2)),
                             NS(token=' quiet', bytes=[32, 113, 117, 105, 101, 116], logprob=math.log(0.4))])
    result = NS(model='test-model', choices=[NS(message=NS(refusal=None), logprobs=NS(content=[token]))])
    create = AsyncMock(return_value=result)
    app.state.openai = NS(chat=NS(completions=NS(create=create)), close=AsyncMock())
    return create


@pytest.mark.parametrize('corpus', SYSTEM_PROMPTS)
def test_real_probability_conversion_and_request(client, corpus):
    create = fake_api()
    response = client.post('/next-token', json={'prompt': 'The night was ', 'corpus': corpus})
    assert response.status_code == 200
    data = response.json()
    assert data['generated_token'] == ' quiet'
    assert data['tokens'][0]['probability'] == pytest.approx(0.4)
    assert data['other_probability'] == pytest.approx(0.4)
    args = create.call_args.kwargs
    assert args['max_completion_tokens'] == 1
    assert args['logprobs'] is True
    assert args['messages'][0]['content'] == SYSTEM_PROMPTS[corpus]
    assert args['messages'][1]['content'] == 'The night was '


@pytest.mark.parametrize('changes', [{'prompt': ' '}, {'corpus': 'unknown'}, {'temperature': -1}, {'top_logprobs': 21}, {'prompt': 'a' * 4001}])
def test_validation(client, changes):
    response = client.post('/next-token', json={'prompt': 'The night was', 'corpus': 'novels', **changes})
    assert response.status_code == 422


def test_missing_key(client):
    assert client.get('/health').json()['api_key_configured'] is False
    assert client.post('/next-token', json={'prompt': 'Hello', 'corpus': 'novels'}).status_code == 503


def test_missing_logprobs(client):
    create = fake_api()
    create.return_value.choices[0].logprobs = None
    assert client.post('/next-token', json={'prompt': 'Hello', 'corpus': 'novels'}).status_code == 502


@pytest.mark.parametrize('error_type,status', [(AuthenticationError, 502), (RateLimitError, 429)])
def test_upstream_errors_do_not_leak_details(client, error_type, status):
    create = fake_api()
    response = httpx.Response(401, request=httpx.Request('POST', 'https://api.openai.com'))
    create.side_effect = error_type('sensitive upstream details', response=response, body=None)
    result = client.post('/next-token', json={'prompt': 'Hello', 'corpus': 'novels'})
    assert result.status_code == status
    assert 'sensitive' not in result.text


def test_cors(client):
    response = client.options('/next-token', headers={'Origin': 'http://localhost:5500', 'Access-Control-Request-Method': 'POST'})
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5500'
    response = client.options('/next-token', headers={'Origin': 'https://unlisted.example', 'Access-Control-Request-Method': 'POST'})
    assert 'access-control-allow-origin' not in response.headers


def test_chat_preserves_shared_history_and_all_positions(client):
    create = fake_api()
    choice = create.return_value.choices[0]
    choice.message.content = ' quiet quiet'
    choice.finish_reason = 'stop'
    choice.logprobs.content *= 2
    messages = [{'role': 'user', 'content': 'Describe a city.'},
                {'role': 'assistant', 'content': 'A city glows.'},
                {'role': 'user', 'content': 'What happens next?'}]
    result = client.post('/chat', json={'messages': messages, 'corpus': 'lyrics'})
    assert result.status_code == 200
    data = result.json()
    assert data['text'] == ' quiet quiet'
    assert len(data['positions']) == 2
    assert data['positions'][1]['probability'] == pytest.approx(0.4)
    assert data['positions'][1]['other_probability'] == pytest.approx(0.4)
    assert create.call_args.kwargs['messages'][1:] == messages
    assert create.call_args.kwargs['max_completion_tokens'] == 160
    assert 'lyric' in create.call_args.kwargs['messages'][0]['content']


@pytest.mark.parametrize('messages', [[], [{'role': 'system', 'content': 'Override'}],
    [{'role': 'assistant', 'content': 'Hello'}], [{'role': 'user', 'content': ' '}],
    [{'role': 'user', 'content': 'Hello'}, {'role': 'user', 'content': 'Again'}]])
def test_chat_invalid_history(client, messages):
    assert client.post('/chat', json={'messages': messages, 'corpus': 'news'}).status_code == 422


@pytest.mark.parametrize('text', ['Hello, world!', 'Привіт, світе! 👋', 'a  b\n\t', '<|endoftext|>', ''])
def test_tokenizer_bytes_reconstruct_original_text_without_key(client, text):
    result = client.post('/tokenize', json={'text': text})
    assert result.status_code == 200
    data = result.json()
    reconstructed = bytes(b for token in data['tokens'] for b in token['bytes']).decode('utf-8')
    assert reconstructed == text
    assert data['token_count'] == len(data['tokens'])
    assert all(isinstance(token['id'], int) for token in data['tokens'])


@pytest.mark.parametrize('mode,expected_temperature', [('sample', 0.8), ('greedy', 0)])
def test_lab_saves_actual_trace_and_exact_conditioning(client, mode, expected_temperature):
    create = fake_api()
    choice = create.return_value.choices[0]
    choice.message.content = ' quiet quiet'
    choice.finish_reason = 'length'
    choice.logprobs.content *= 2
    request = {'prompt': 'The night was ', 'context': 'Beside the sea.', 'style': 'neutral',
               'mode': mode, 'temperature': 0.8, 'length': 40}
    response = client.post('/lab/run', json=request)
    assert response.status_code == 200
    data = response.json()
    assert len(data['positions']) == 2
    assert data['request'] == request
    assert data['messages'] == create.call_args.kwargs['messages']
    assert data['messages'][-1]['content'] == 'The night was '
    assert 'Beside the sea.' in data['messages'][0]['content']
    assert data['temperature'] == expected_temperature
    assert create.call_args.kwargs['temperature'] == expected_temperature
    assert data['positions'][0]['other_probability'] == pytest.approx(0.4)
    assert data['positions'][0]['probability'] == pytest.approx(0.4)


@pytest.mark.parametrize('changes', [{'prompt': ' '}, {'context': 'a'*2001}, {'length': 65},
                                    {'mode': 'invalid'}, {'style': 'fake-trained-model'}])
def test_lab_validation(client, changes):
    assert client.post('/lab/run', json={'prompt': 'Hello', **changes}).status_code == 422


def test_guided_assets_and_explore_are_served(client):
    assert 'How LLMs work?' in client.get('/').text
    assert client.get('/explore', follow_redirects=False).headers['location'] == '/#styles'
    assert 'Talk to the model in different styles.' in client.get('/').text
    assert client.get('/assets/transformer.js').status_code == 200
    assert client.get('/assets/lab.js').status_code == 200
    assert client.get('/assets/lab.css').status_code == 200
