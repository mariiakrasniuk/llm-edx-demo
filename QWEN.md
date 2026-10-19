# Qwen lab on Cartesian

The original worksheet uses a constructed, untrained width-4 transformer for
internals and OpenAI for generation. This separate working entry point uses
Qwen3-4B for measured attention, embeddings, hidden states, probabilities and chat.
No API key is needed. The original app remains at `backend.main:app`.

## Start

```bash
cd /ssd/mkrasniuk/waterloo/llm-edx-demo
python -m venv .venv
# CUDA 12.4 wheels work with this server's NVIDIA 560 driver:
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r backend/requirements-qwen.txt
bash scripts/run-qwen.sh
```

Open http://localhost:8000 on the server. From your laptop, forward the port:

```bash
ssh -N -L 8000:127.0.0.1:8000 mkrasniuk@cartesian
```

Then open http://localhost:8000 on your laptop. `PORT=8002 bash scripts/run-qwen.sh`
selects a different port; adjust both tunnel port numbers accordingly.

The first inspection downloads about 8 GB to `.cache/huggingface` and loads the
model on GPU 0. `QWEN_DEVICE=cuda:1` selects GPU 1. `QWEN_MODEL` can select another
compatible Qwen3 checkpoint. A single process owns the model; do not add uvicorn
workers. Simultaneous inference requests return 429 for retry rather than queueing.

## What the screen measures

- Raw input text, tokenized by the same Qwen tokenizer. No hidden chat wrapper in
  the inspector. At most 96 tokens, including appended tokens.
- Eager attention from the selected layer and query head. The Head View is an
  original SVG implementation inspired by the two-column BertViz presentation;
  BertViz itself is not installed. Head indices in the UI start at 1; API at 0.
- First 8 coordinates of learned embeddings and selected hidden states. The final
  layer reports final-normalized output. No synthetic vectors or dimensionality
  reduction. Rotary position information is applied in attention.
- Full-vocabulary softmax of logits divided by temperature; top 10 plus remaining
  probability mass. Sampling uses this same full distribution with no top-k/p
  filtering. Appending preserves token IDs, including partial Unicode pieces.
  Individual decoded token text may contain a replacement character; IDs and
  tokenizer pieces are included in downloadable evidence.
- Chat uses the same weights but a different input format: Qwen's non-thinking
  chat template, greedy decoding and a 96-token output cap. It is a single-turn
  demonstration. It is not the inspector's raw continuation.

Attention weights are mixing coefficients, not causal explanations or confidence
scores. This prototype has no authentication or class-wide load management; it
binds to loopback for development. Open edX integration and multi-learner serving
remain separate work.

## Verify

```bash
.venv/bin/python -m pytest backend/tests/test_qwen.py -q
node --check frontend/assets/qwen.js
```

Tests use a tiny random Qwen3 solely to verify numerical and API invariants, never
as displayed teaching data. The app always loads the configured trained checkpoint.

Sources: [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B),
[Transformers Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3).

## Complete lesson (current default)

The Qwen server now serves `frontend/lesson.html` at `/`, with the original
introduction and all ten lesson topics. It preserves the original generation
playback, context A/B comparison, four-style comparison and shared-history chat.
These use local `/lab/run`, `/next-token` and `/chat` routes backed by Qwen.
Comparison requests serialize on one model lock (up to 60 seconds waiting).
No OpenAI requests or keys are used by this entry point.

The inspector now includes all query heads of the selected layer, selectable
colored Head View connections, measured block stages, a temporary Q-projection
weight experiment, temperature comparisons with full-vocabulary sampling, and
an actual truncated-context comparison. Weight edits are restored in `finally`
even on inference failure. The inspector's experimental setting persists in its
controls; the generation and chat lessons always use the original weights.

Transformer Explainer is available as an optional embedded external tool and a
new-tab link, explicitly labeled as a separate GPT-2 demonstration. BertViz's
original tutorial and Tensor2Tensor attribution accompany our Head View.

Browser verification (requires `pip install playwright` and a Chromium install):

```bash
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright .venv/bin/python scripts/smoke-lesson.py
```

The live smoke test exercises all major activities on the real checkpoint and
checks mobile overflow and JavaScript errors. Screenshots are in
`artifacts/full-lesson-*.png`. The earlier `qwen.html` is retained as a prototype
file; it is no longer the default page.

## Short guided route

The default page now offers a roughly 15-minute, four-experiment route for
nontechnical learners: tokens, attention, next-token sampling, and context A/B.
It introduces LLMs, transformers and training in plain language, asks learners
to predict before interacting, shows the growing sentence, and ends with a
concept check with explanatory feedback. Completion is explicitly marked by the
learner, not inferred from scrolling. Progress and reflection are page-local.

All deeper tools remain in closed “Explore further” disclosures, including
vectors, weight edits, repeated sampling, truncation, generation playback,
styles, chat, Transformer Explainer and BertViz. The original ten topics are
available without making all of them part of the required route.

`smoke-guide.py` verifies the default short route, optional controls, concept
feedback, unique element IDs and mobile layout. `smoke-lesson.py` opens the
optional disclosures to verify the complete collection of activities.

The second core experiment now compares Qwen's explanations of “bank” in two
editable sentences, with the same question, style instruction and greedy
selection settings. It includes an identical-sentence control and displays real
responses, not prewritten meanings. Head View is an optional close-up, explicitly
separated from the chat-formatted meaning requests. Its guide explains layers,
heads, unchanged token labels, self-connections and interpretability limits.
Run `scripts/smoke-meaning.py` with the same Playwright environment to check this
activity, the control, optional attention rendering and mobile overflow.
