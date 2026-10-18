"""Guided lab: actual tokenizer boundaries and replayable generation evidence."""
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import tiktoken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
# Keep downloaded tokenizer data in this project, not in a user's global cache.
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).parent.parent / '.cache' / 'tiktoken'))


@lru_cache(maxsize=4)
def encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        raise HTTPException(503, 'No known tokenizer for the configured model. Configure a supported OpenAI model.') from None
    except (OSError, ValueError):
        raise HTTPException(503, 'Tokenizer vocabulary unavailable. Connect once to download it, then retry.') from None


class TokenizeRequest(BaseModel):
    text: str = Field(default='', max_length=4000)


@router.post('/tokenize')
def tokenize(body: TokenizeRequest):
    from .main import MODEL
    enc = encoding(MODEL)
    ids = enc.encode(body.text, disallowed_special=())
    tokens = []
    for token_id in ids:
        raw = enc.decode_single_token_bytes(token_id)
        try:
            text = raw.decode('utf-8')
            complete = True
        except UnicodeDecodeError:
            text = ''.join(f'\\x{b:02x}' for b in raw)
            complete = False
        tokens.append({'id': token_id, 'text': text, 'bytes': list(raw), 'complete_utf8': complete})
    return {'tokens': tokens, 'token_count': len(ids), 'encoding': enc.name, 'model': MODEL,
            'text': body.text}


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    context: str = Field(default='', max_length=2000)
    style: Literal['neutral', 'novels', 'lyrics', 'news', 'academic'] = 'neutral'
    mode: Literal['sample', 'greedy'] = 'sample'
    temperature: float = Field(default=1, ge=0.1, le=2)
    length: int = Field(default=40, ge=1, le=64)

    @field_validator('prompt')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Enter a text prefix.')
        return value


def messages_for(body):
    from .main import COMMON, SYSTEM_PROMPTS
    system = COMMON if body.style == 'neutral' else SYSTEM_PROMPTS[body.style]
    if body.style == 'neutral':
        system += ('\nReturn the missing suffix, not a rewritten sentence. '
                   'Examples of the required input/output boundary:\n'
                   'Input: The light turned\nOutput: green.\n'
                   'Input: Beyond the window\nOutput: rain blurred the hills.\n'
                   'Preserve any leading space needed to join your continuation to the input.')
    messages = [{'role': 'system', 'content': system}]
    if body.context.strip():
        # A separate message keeps the compared prefix identical in A/B experiments.
        messages[0]['content'] += '\nBackground context for this continuation:\n' + body.context
    messages.append({'role': 'user', 'content': body.prompt})
    return messages


@router.post('/lab/run')
async def run(body: RunRequest):
    from .main import request_completion, alternatives
    messages = messages_for(body)
    temperature = 0 if body.mode == 'greedy' else body.temperature
    result = await request_completion(messages, body.length, temperature, 10)
    choice = result.choices[0]
    positions = []
    for item in choice.logprobs.content:
        options = alternatives(item)
        positions.append({'token': item.token, 'bytes': item.bytes, 'logprob': item.logprob,
            'probability': math.exp(item.logprob),
            'alternatives': [option.model_dump() for option in options],
            'other_probability': max(0, 1 - sum(option.probability for option in options))})
    return {'text': choice.message.content or '', 'positions': positions, 'model': result.model,
            'temperature': temperature, 'mode': body.mode, 'finish_reason': choice.finish_reason,
            'request': body.model_dump(), 'messages': messages,
            'probability_source': 'OpenAI Chat Completions logprobs; exp(logprob), not renormalized.'}
