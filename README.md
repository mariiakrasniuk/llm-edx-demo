# Next Token — interactive language-model lab

A guided Open edX activity about tokenization, next-token probability,
context, and repeated generation. The frontend and backend run together.

## Run

Requires Python 3.10+.

```sh
cd /Users/mariia/llm-edx-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Stop with **Ctrl+C**.

Set `OPENAI_API_KEY` in `backend/.env` for live generation, comparison, and
chat. A starter file is provided; `backend/.env.example` documents its fields.
Restart after changing environment settings. Keys never go to the browser.
`OPENAI_MODEL` defaults to `gpt-4.1-mini` and must support Chat Completions,
logprobs, temperature, and a tokenizer known to tiktoken.

Tokenization needs no API key. On first use, tiktoken downloads the public
vocabulary to `.cache/tiktoken/`; later tokenization works offline. To warm
the cache before a class or an offline test run:

```sh
python -c 'from backend.lab import encoding; from backend.main import MODEL; print(encoding(MODEL).name)'
```

## Single-page learning sequence

The lab is a continuous worksheet. All sections are present on one page;
the links at the top scroll to sections rather than switch screens.

The opening defines LLMs, transformers, training, and generation without assuming a mathematics background. Main activities use plain language; matrices, Q/K/V, token bytes, and full calculations are optional.

1. **Tokens:** whole words, fragments, punctuation, and visible spaces. The
   Unbelievable! preset explicitly names the text it loads.
2. **Vectors:** lists of four numbers, position information, and a block overview.
   Attention, feed-forward processing (MLP), and residual additions are explained
   before the diagram.
3. **Model settings:** move one setting and compare the original result with
   the current result. Matrix arithmetic is available in optional details.
4. **Attention:** context connections and an interactive causal attention heatmap.
   Hover or focus a cell for the exact share; click to inspect its row.
5. **Prediction and repetition:** eight next-token chances; sample, choose,
   append, and undo. Scores and parameter-change deltas remain optional.
   A temperature explorer compares fixed scores at 0.3, 1.0, and 2.0 and counts
   repeated independent draws. A context-window activity removes earlier tokens
   and compares actual teaching-model predictions before and after truncation.
6. **A trained model:** the existing API-backed generation trace and replay.
7. **Context A/B:** controlled API comparison with matching token rows.
8. **Writing styles:** compare instructions using a shared prefix.
9. **Chat:** compare full replies and inspect token probabilities under a shared history.
10. **Reflection:** distinguish changing a weight from changing context.

**Two sources of numbers are kept separate.** Tokenization is real tiktoken
output. Sections 2–5 use a **constructed, untrained teaching transformer**,
not GPT's activations or attention. Its token lookup is reproducibly generated
from token IDs; its positional signal is sinusoidal; all matrices are fixed,
hand-chosen values. It has width 4, one causal attention head, one pre-normalized
decoder block, a 4→8→4 GELU MLP, residual connections, final normalization, and
tied output vectors over eight actual token IDs. Every displayed operation and
probability is computed, not mocked. Its language quality is not meaningful.
It supports up to 32 tokens and preserves generated token IDs rather than
retokenizing them when appending. Tokenization itself can handle longer inputs.

The API sections use actual model output logprobs. The API does not expose
internal vectors, attention weights, or pretrained model parameters. The
teaching displays must never be described as measurements from that API model.

The full trained-model experience in Georgia Tech's Transformer Explainer is
linked as further exploration, including its approximate 600 MB download notice.
No third-party model is downloaded by this lab automatically.

Sections 8 and 9 integrate style comparison and conversational experiments into
the same worksheet, with shared styling and separate request controls. Every
section includes theory and an explicit activity walkthrough. The context
comparison includes an identical-context control. `/explore` is a compatibility
redirect to `/#styles`; there is no separate exercise page.

## What the evidence means

A guided generation run makes **one paid API call**, requesting up to 40 tokens
and logprobs for all returned output positions. The browser then reveals this
real recorded trajectory one position at a time. It does not pretend to make a
new API prediction on each playback click. The model's actual earlier output
is the conditioning context for each later token in that same request.
Playback, undo, and replay do not make further calls. A new run does.

The visible prefix is a user message; generated continuation is assistant
output. Chat boundaries, common continuation instructions, style instructions,
and background context are all part of conditioning. This is not a raw
base-model completion API or an attention visualization. Exact messages and
model identity are included in the downloadable evidence.

Probabilities come from `exp(logprob)` with **no renormalization**. The guided
chart displays eight alternatives; “Other” includes the rest of the vocabulary,
including the two additional returned alternatives that are not in that chart.
The context comparison displays the union of both returned top-10 lists.
Missing entries are unknown, not zero; “Other” is not token-aligned across
conditions. Tiny nonzero probabilities appear as `<0.1%`, and values rounded
near one appear as `>99.9%` rather than falsely implying certainty.

A probability is not factual accuracy or a genre-classification score. Similar
distributions are a valid outcome. Style prompts do not retrain a model. This
version explains how training shapes predictions but does not compare separately
trained models or isolate the causal effect of different training corpora.

## Backend endpoints

- `GET /health`: readiness and model configuration; no credentials returned.
- `POST /tokenize`: `{ "text": "Unbelievable!" }`; returns IDs and byte boundaries.
- `POST /lab/run`: `{ "prompt": "The night was", "context": "", "style": "neutral", "mode": "sample", "temperature": 1, "length": 40 }`; returns actual token trace and conditioning messages.
- `POST /next-token`: original single-token API; `corpus` is novels/lyrics/news/academic.
- `POST /chat`: original chat comparison API, with alternating user/assistant messages.
- `GET /docs`: interactive FastAPI documentation.

```sh
curl --fail-with-body http://localhost:8000/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привіт, світе!"}'

curl --fail-with-body http://localhost:8000/lab/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The night was","mode":"sample","length":8}'
```

Invalid input returns 422; missing credentials 503; provider connection,
authentication, or model failures 502; quota/rate limits 429; timeouts 504.
CORS permits local frontend ports 5500 and 8001 by default. The same-server
frontend needs no CORS configuration. Configure exact Open edX origins using
`CORS_ORIGINS` when integrating. This remains a local prototype: public use
needs learner authentication and server-side usage limits. CORS is not access
control. Open edX packaging/deployment is a later step.

## Checks

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q
node --check frontend/assets/lab.js
node --test frontend/tests/transformer.test.cjs
```

Provider responses are stubbed in automated tests; they incur no API charges.
Tokenizer tests use real vocabularies and require the one-time cache warm-up
above for a fully offline run. Coverage includes Unicode byte reconstruction,
request validation, trace preservation, temperature selection, CORS, provider
errors, and the original next-token/chat endpoints. The numerical transformer tests
check causal isolation, softmax stability, vector transformation arithmetic,
context sensitivity, and changes induced by an edited projection weight.
Browser QA covers live
API generation, context comparison, timeline rewind, and responsive layouts.

## References

- [OpenAI Chat Completions and logprobs](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [OpenAI tokenizer and advanced usage](https://developers.openai.com/api/docs/guides/advanced-usage)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

The design is original; the references inspired the learning flow.
