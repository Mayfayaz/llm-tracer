# llm-tracer

A minimal, dependency-free span tracer for LLM pipelines.

When an LLM call chain gets slow or expensive, you need to know *which step*
is responsible. `llm-tracer` gives you a tiny OpenTelemetry-style tracing
primitive — spans that nest into trees, with per-span latency, token usage,
estimated cost, events and error capture — plus JSON export and a readable
ASCII tree in your terminal. One file (`tracer.py`), zero dependencies.

## What it does

- **Nested spans** via a context manager or decorator; children attach to the
  currently open span using `contextvars` (async- and thread-safe).
- **Token & cost accounting** — record prompt/completion tokens per span;
  costs are estimated from a per-model price table you can override.
- **Error capture** — exceptions inside a span are recorded on it (then
  re-raised, so pipelines fail loudly) and flagged in the tree.
- **Export** — full traces serialize to JSON (`save_json`).
- **Readable output** — `render()` prints an ASCII span tree with latency,
  tokens and cost; `render_summary()` prints a per-step aggregate table.

## Quick start

```python
from tracer import Tracer

tracer = Tracer()

@tracer.trace(model="gpt-4o-mini")
def generate(prompt, __span__=None):
    text, prompt_tok, completion_tok = call_your_llm(prompt)
    __span__.record_usage(prompt_tokens=prompt_tok, completion_tokens=completion_tok)
    return text

@tracer.trace(kind="retrieval")
def retrieve(query):
    return search_index(query)

@tracer.trace(kind="chain")
def answer(query):
    docs = retrieve(query)
    return generate(f"Q: {query}\nContext: {docs}")

answer("What is a span?")
print(tracer.render())
print(tracer.render_summary())
tracer.save_json("trace.json")
```

Output:

```
answer [chain] (   213.4ms) (cost n/a)
├── retrieve [retrieval] (    12.1ms) (cost n/a)
└── generate [llm] (   201.2ms) (  142 tok) ($0.000032)
```

## Run the demo

No API keys needed — the demo uses a deterministic fake LLM so the trace,
cost math and an (intentional) retrieval error are all reproducible:

```
python3 demo.py
```

## Tests

Stdlib only:

```
python3 -m unittest discover tests
```

15 tests cover span nesting, token/cost math, error capture, the decorator,
JSON export and thread safety.

## Design notes

- `Span` is a plain dataclass; `Tracer.span()` is a context manager built on
  `contextlib`, with the active span tracked in a `ContextVar` so nested and
  concurrent traces stay correct.
- Pricing defaults are approximate (per 1M tokens); pass `Tracer(pricing={...})`
  for exact rates, or `model=None` for non-LLM steps where cost is not tracked.
- Token counts are supplied by you (most providers return usage); the
  `estimate_tokens` helper (`~4 chars/token`) covers providers that don't.

## License

MIT — see [LICENSE](LICENSE).
