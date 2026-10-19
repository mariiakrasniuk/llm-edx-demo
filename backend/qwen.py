"""Measured, bounded Qwen inference for the classroom inspector."""
import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv('QWEN_MODEL', 'Qwen/Qwen3-4B')
DEVICE = os.getenv('QWEN_DEVICE', 'cuda:0')
CACHE = str(ROOT / '.cache' / 'huggingface')
lock = threading.Lock()
model = tokenizer = None
app = FastAPI(title='Qwen — inside a trained transformer')
app.mount('/assets', StaticFiles(directory=ROOT / 'frontend' / 'assets'), name='assets')


def load():
    global model, tokenizer
    if model is None:
        tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
        net = AutoModelForCausalLM.from_pretrained(
            MODEL, cache_dir=CACHE, torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else torch.float32,
            attn_implementation='eager').to(DEVICE).eval()
        tokenizer, model = tok, net


@app.get('/')
def index():
    return FileResponse(ROOT / 'frontend' / 'lesson.html', headers={'Cache-Control': 'no-store'})


@app.get('/health')
def health():
    return {'model': MODEL, 'loaded': model is not None, 'device': DEVICE}


class Inspect(BaseModel):
    text: str = Field(default='The cat sat on the mat', min_length=1, max_length=2000)
    token_ids: list[int] | None = Field(default=None, min_length=1, max_length=96)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)
    temperature: float = Field(default=1, ge=0.1, le=2)
    query_delta: float = Field(default=0, ge=-2, le=2)


@app.post('/inspect')
def inspect(body: Inspect):
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'The model is busy. Please try again shortly.')
    try:
        load()
        ids = body.token_ids if body.token_ids is not None else tokenizer.encode(body.text, add_special_tokens=False)
        if not ids or len(ids) > 96:
            raise HTTPException(422, 'Use between 1 and 96 tokens for the attention view.')
        if any(i < 0 or i >= model.config.vocab_size for i in ids):
            raise HTTPException(422, 'Token ID is outside the model vocabulary.')
        if body.layer >= model.config.num_hidden_layers or body.head >= model.config.num_attention_heads:
            raise HTTPException(422, 'Layer or head is outside this model.')
        inputs = torch.tensor([ids], device=DEVICE)
        block = model.model.layers[body.layer]
        stages = {}
        def capture(name):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                stages[name] = value[0, :, :8].detach().float().cpu().tolist()
            return hook
        hooks = [block.self_attn.register_forward_hook(capture('attention_update')),
                 block.mlp.register_forward_hook(capture('mlp_update')),
                 block.post_attention_layernorm.register_forward_pre_hook(
                     lambda module, args: stages.update(after_attention=args[0][0, :, :8].detach().float().cpu().tolist()))]
        weight = block.self_attn.q_proj.weight
        original = weight[0, 0].item()
        try:
            with torch.inference_mode():
                weight[0, 0] = original + body.query_delta
                result = model(inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
        finally:
            with torch.inference_mode():
                weight[0, 0] = original
            for hook in hooks:
                hook.remove()
        with torch.inference_mode():
            probabilities = torch.softmax(result.logits[0, -1].float() / body.temperature, dim=-1)
            values, indices = probabilities.topk(10)
            sampled = torch.multinomial(probabilities, 1).item()
        def token(i):
            return {'id': i, 'text': tokenizer.decode([i]), 'piece': tokenizer.convert_ids_to_tokens(i)}
        return {
            'model': MODEL, 'revision': getattr(model.config, '_commit_hash', None),
            'text': tokenizer.decode(ids), 'tokens': [token(i) for i in ids],
            'layers': model.config.num_hidden_layers, 'heads': model.config.num_attention_heads,
            'width': model.config.hidden_size, 'layer': body.layer, 'head': body.head,
            'query_weight': original, 'query_delta': body.query_delta,
            'stages': stages,
            'block_input': result.hidden_states[body.layer][0, :, :8].float().cpu().tolist(),
            'all_heads': result.attentions[body.layer][0].float().cpu().tolist(),
            'temperature_comparison': [distribution(result.logits[0, -1], t, 100) for t in (0.3, 1.0, 2.0)],
            'attention': result.attentions[body.layer][0, body.head].float().cpu().tolist(),
            'embedding': model.get_input_embeddings()(inputs)[0, :, :8].detach().float().cpu().tolist(),
            'hidden': result.hidden_states[body.layer + 1][0, :, :8].float().cpu().tolist(),
            'hidden_label': 'Final normalized output' if body.layer == model.config.num_hidden_layers - 1 else 'Output of selected decoder block',
            'alternatives': [{**token(i), 'probability': p} for i, p in zip(indices.tolist(), values.tolist())],
            'other_probability': max(0, 1 - values.sum().item()),
            'sampled_id': sampled, 'greedy_id': indices[0].item(),
            'eos_ids': list(set([tokenizer.eos_token_id] + (model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id]))),
            'temperature': body.temperature,
            'source': 'Real forward pass; raw text, no chat template. Full-vocabulary softmax(logits / temperature).',
        }
    finally:
        lock.release()


class Reply(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.post('/reply')
def reply(body: Reply):
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'The model is busy. Please try again shortly.')
    try:
        load()
        inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': body.text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(DEVICE)
        if inputs.shape[1] > 512:
            raise HTTPException(422, 'Please shorten your message to at most 512 tokens including chat formatting.')
        with torch.inference_mode():
            output = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=96,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return {'text': tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True),
                'model': MODEL, 'input_ids': inputs[0].tolist(),
                'note': 'Chat formatting with thinking disabled; greedy generation, maximum 96 new tokens.'}
    finally:
        lock.release()


# These routes keep the complete worksheet on the same local trained model.
from .lab import RunRequest, messages_for
from .main import NextTokenRequest, ChatRequest, SYSTEM_PROMPTS, CHAT_STYLES
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def token_record(token_id):
    piece = tokenizer.convert_ids_to_tokens(token_id)
    try:
        raw = bytes(BYTE_DECODER[c] for c in piece)
    except KeyError:
        raw = tokenizer.decode([token_id]).encode('utf-8')
    return {'id': token_id, 'token': raw.decode('utf-8', errors='replace'), 'bytes': list(raw)}


def distribution(logits, temperature, draws=0):
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    values, indices = probs.topk(10)
    options = [{**token_record(i), 'probability': p} for i, p in zip(indices.tolist(), values.tolist())]
    counts = torch.bincount(torch.multinomial(probs, draws, replacement=True), minlength=len(probs)) if draws else None
    return {'temperature': temperature, 'alternatives': options,
            'other_probability': max(0, 1-values.sum().item()),
            'draws': draws, 'counts': [counts[i].item() for i in indices] if draws else [],
            'other_count': draws-sum(counts[i].item() for i in indices) if draws else 0}


def completion(messages, limit, temperature, greedy=False):
    if not lock.acquire(timeout=60):
        raise HTTPException(429, 'The model is busy. Please try again shortly.')
    try:
        load()
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) > 2048:
            raise HTTPException(422, 'Conversation exceeds the 2,048-token classroom limit. Shorten the input or start a new chat.')
        inputs = torch.tensor([ids], device=DEVICE)
        positions, generated, cache = [], [], None
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        with torch.inference_mode():
            for _ in range(limit):
                output = model(inputs, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[0, -1].float()
                logprobs = torch.log_softmax(logits / (1 if greedy else temperature), dim=-1)
                probs = logprobs.exp()
                chosen = logits.argmax().item() if greedy else torch.multinomial(probs, 1).item()
                values, indices = probs.topk(10)
                alternatives = [{**token_record(i), 'probability': p, 'logprob': logprobs[i].item()}
                                for i, p in zip(indices.tolist(), values.tolist())]
                positions.append({**token_record(chosen), 'probability': probs[chosen].item(),
                    'logprob': logprobs[chosen].item(), 'alternatives': alternatives,
                    'other_probability': max(0, 1-values.sum().item())})
                generated.append(chosen)
                if chosen in eos:
                    break
                inputs = torch.tensor([[chosen]], device=DEVICE)
        return {'text': tokenizer.decode(generated, skip_special_tokens=True), 'positions': positions,
                'model': MODEL, 'messages': messages, 'input_ids': ids,
                'temperature': 1 if greedy else temperature, 'mode': 'greedy' if greedy else 'sample',
                'finish_reason': 'stop' if generated[-1] in eos else 'length',
                'probability_source': 'Qwen full-vocabulary softmax(logits / temperature); greedy displays temperature-1 probabilities.'}
    finally:
        lock.release()


@app.post('/lab/run')
def lesson_run(body: RunRequest):
    return {**completion(messages_for(body), body.length, body.temperature, body.mode == 'greedy'),
            'request': body.model_dump()}


@app.post('/next-token')
def lesson_next(body: NextTokenRequest):
    result = completion([{'role': 'system', 'content': SYSTEM_PROMPTS[body.corpus]},
                         {'role': 'user', 'content': body.prompt}], 1, body.temperature, body.temperature == 0)
    first = result['positions'][0]
    return {'generated_token': first['token'], 'generated_token_bytes': first['bytes'],
            'generated_logprob': first['logprob'], 'generated_probability': first['probability'],
            'tokens': first['alternatives'], 'other_probability': first['other_probability'],
            'corpus': body.corpus, 'model': MODEL, 'temperature': result['temperature']}


@app.post('/chat')
def lesson_chat(body: ChatRequest):
    messages = [{'role': 'system', 'content': 'Answer helpfully in about 50 words. ' + CHAT_STYLES[body.corpus]}]
    messages += [m.model_dump() for m in body.messages]
    return {**completion(messages, 96, body.temperature, body.temperature == 0), 'corpus': body.corpus}
