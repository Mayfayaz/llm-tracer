"""llm-tracer: minimal span-based observability for LLM pipelines.

Zero dependencies. Trace nested LLM calls, tool invocations and retrieval
steps; record latency, token usage and estimated cost; export JSON traces
and render them as ASCII trees in the terminal.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import threading
import time
import uuid
from dataclasses import dataclass, field

__version__ = "0.1.0"

# Approximate prices in USD per 1M tokens: {model: (input, output)}.
# Pass your own `pricing` dict to Tracer for exact numbers.
DEFAULT_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-3-5": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


def estimate_tokens(text: str) -> int:
    """Crude token estimate (~4 chars/token) for when a model doesn't report usage."""
    return max(1, len(text) // 4)


@dataclass
class Span:
    name: str
    kind: str = "llm"  # "llm" | "retrieval" | "tool" | "chain"
    model: str | None = None
    metadata: dict = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    children: list["Span"] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    pricing: dict | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float | None:
        table = self.pricing or DEFAULT_PRICING
        if self.model not in table:
            return None
        p_in, p_out = table[self.model]
        return self.prompt_tokens / 1e6 * p_in + self.completion_tokens / 1e6 * p_out

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> "Span":
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        return self

    def add_event(self, name: str, **attrs) -> "Span":
        self.events.append({"name": name, "at_ms": self.duration_ms, **attrs})
        return self

    def set_error(self, message: str) -> "Span":
        self.error = message
        self.add_event("error", message=message)
        return self

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "metadata": self.metadata,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "error": self.error,
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }


_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "llm_tracer_current_span", default=None
)


class Tracer:
    """Collects spans into trees. Thread-safe; one tracer per pipeline run."""

    def __init__(self, pricing: dict | None = None):
        self.pricing = pricing if pricing is not None else DEFAULT_PRICING
        self._roots: list[Span] = []
        self._lock = threading.Lock()

    # -- span creation ----------------------------------------------------
    @contextlib.contextmanager
    def span(self, name: str, kind: str = "llm", model: str | None = None, **metadata):
        parent = _current_span.get()
        span = Span(
            name=name,
            kind=kind,
            model=model,
            metadata=metadata,
            parent_id=parent.span_id if parent else None,
            pricing=self.pricing,
        )
        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:  # record, then re-raise so pipelines fail loudly
            span.set_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.end_time = time.perf_counter()
            _current_span.reset(token)
            with self._lock:
                if parent is not None:
                    parent.children.append(span)
                else:
                    self._roots.append(span)

    def trace(self, name: str | None = None, **span_kwargs):
        """Decorator version of span(); uses the wrapped function's name by default."""

        def decorator(fn):
            span_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.span(span_name, **span_kwargs) as span:
                    return fn(*args, **kwargs, __span__=span) \
                        if "__span__" in fn.__code__.co_varnames else fn(*args, **kwargs)

            return wrapper

        return decorator

    # -- queries ----------------------------------------------------------
    @property
    def roots(self) -> list[Span]:
        with self._lock:
            return list(self._roots)

    def _walk(self):
        stack = list(self.roots)
        while stack:
            span = stack.pop()
            yield span
            stack.extend(span.children)

    def summary(self) -> list[dict]:
        """Aggregate latency/tokens/cost per span name across all traces."""
        agg: dict[str, dict] = {}
        for span in self._walk():
            row = agg.setdefault(
                span.name,
                {"name": span.name, "calls": 0, "total_ms": 0.0,
                 "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
                 "errors": 0, "cost_unknown": False},
            )
            row["calls"] += 1
            row["total_ms"] += span.duration_ms
            row["prompt_tokens"] += span.prompt_tokens
            row["completion_tokens"] += span.completion_tokens
            row["errors"] += 1 if span.error else 0
            if span.cost_usd is None:
                row["cost_unknown"] = True
            else:
                row["cost_usd"] += span.cost_usd
        rows = sorted(agg.values(), key=lambda r: r["total_ms"], reverse=True)
        for r in rows:
            r["total_ms"] = round(r["total_ms"], 1)
            r["cost_usd"] = round(r["cost_usd"], 6)
        return rows

    # -- export / render --------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": __version__, "traces": [r.to_dict() for r in self.roots]}

    def save_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def render(self) -> str:
        """ASCII tree of every root trace with latency, tokens and cost."""
        lines: list[str] = []

        def fmt(span: Span) -> str:
            bits = [f"{span.duration_ms:8.1f}ms"]
            if span.total_tokens:
                bits.append(f"{span.total_tokens:>5d} tok")
            cost = span.cost_usd
            bits.append(f"${cost:.6f}" if cost is not None else "cost n/a")
            if span.error:
                bits.append("ERROR")
            return f"{span.name} [{span.kind}] " + " ".join(f"({b})" for b in bits)

        def walk(span: Span, prefix: str, last: bool):
            connector = "└── " if last else "├── "
            lines.append(prefix + connector + fmt(span))
            extension = "    " if last else "│   "
            for i, child in enumerate(span.children):
                walk(child, prefix + extension, i == len(span.children) - 1)

        for root in self.roots:
            lines.append(fmt(root))
            for i, child in enumerate(root.children):
                walk(child, "", i == len(root.children) - 1)
            lines.append("")
        return "\n".join(lines).rstrip()

    def render_summary(self) -> str:
        rows = self.summary()
        if not rows:
            return "(no spans recorded)"
        header = f"{'span':28} {'calls':>5} {'total_ms':>9} {'tokens':>7} {'cost_usd':>10} {'err':>3}"
        lines = [header, "-" * len(header)]
        for r in rows:
            cost = "n/a" if r["cost_unknown"] else f"${r['cost_usd']:.6f}"
            lines.append(
                f"{r['name'][:28]:28} {r['calls']:>5} {r['total_ms']:>9.1f} "
                f"{r['prompt_tokens'] + r['completion_tokens']:>7} {cost:>10} {r['errors']:>3}"
            )
        return "\n".join(lines)
