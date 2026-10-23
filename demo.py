"""Demo: trace a small multi-step RAG pipeline with a fake (deterministic) LLM.

No API keys needed — FakeLLM simulates latency, token counts and one
mid-pipeline retrieval error so the trace shows error handling too.
"""

import time

from tracer import Tracer, estimate_tokens

tracer = Tracer()


class FakeLLM:
    """Deterministic stand-in for an LLM API: fixed latency, counted tokens."""

    def __init__(self, model: str, latency_ms: float = 120.0):
        self.model = model
        self.latency_ms = latency_ms

    def generate(self, prompt: str, max_tokens: int = 200) -> tuple[str, int, int]:
        time.sleep(self.latency_ms / 1000.0)
        prompt_tokens = estimate_tokens(prompt)
        # Deterministic canned completion sized by the prompt's topic.
        completion = f"[{self.model}] Draft answer (seeded on {len(prompt)} prompt chars)."
        completion_tokens = min(max_tokens, estimate_tokens(completion))
        return completion, prompt_tokens, completion_tokens


GENERATOR = FakeLLM("gpt-4o-mini", latency_ms=120.0)
JUDGE = FakeLLM("claude-haiku-3-5", latency_ms=90.0)

DOCS = {
    "vector db": "A vector database stores embeddings and retrieves by similarity search.",
    "tracing": "Tracing records spans of work in a tree so you can see latency breakdowns.",
    "rag": "RAG grounds generation in retrieved documents to reduce hallucinations.",
}


@tracer.trace(kind="retrieval")
def retrieve_docs(query: str):
    time.sleep(0.03)
    return [v for k, v in DOCS.items() if k in query.lower()] or list(DOCS.values())[:2]


@tracer.trace(kind="retrieval")
def rerank(docs: list[str], query: str):
    time.sleep(0.02)
    # Toy reranker: prefer the longest doc (stand-in for a cross-encoder score).
    return sorted(docs, key=len, reverse=True)[:2]


@tracer.trace(model="gpt-4o-mini")
def generate_answer(query: str, docs: list[str], __span__=None):
    prompt = f"Question: {query}\nContext:\n" + "\n".join(f"- {d}" for d in docs)
    text, p_tok, c_tok = GENERATOR.generate(prompt)
    __span__.record_usage(prompt_tokens=p_tok, completion_tokens=c_tok)
    return text


@tracer.trace(model="claude-haiku-3-5")
def judge_answer(question: str, answer: str, __span__=None):
    prompt = f"Rate 1-5 whether this answers the question.\nQ: {question}\nA: {answer}"
    verdict, p_tok, c_tok = JUDGE.generate(prompt, max_tokens=20)
    __span__.record_usage(prompt_tokens=p_tok, completion_tokens=c_tok)
    return "5/5 — grounded in retrieved context"


def flaky_retrieval(query: str):
    """A retrieval step that fails, to show error capture in the trace."""
    with tracer.span("flaky_retrieval", kind="retrieval") as span:
        time.sleep(0.01)
        span.set_error("RetrievalError: index shard 2 unavailable (simulated)")
        return []


@tracer.trace(kind="chain")
def answer_question(query: str):
    docs = retrieve_docs(query)
    docs += flaky_retrieval(query)  # errors, keeps going
    top = rerank(docs, query)
    draft = generate_answer(query, top)
    score = judge_answer(query, draft)
    return draft, score


def main():
    with tracer.span("demo_run", kind="chain"):
        draft, score = answer_question("How does tracing help debug a RAG pipeline?")
        answer_question("What is a vector db?")

    print("=== Trace tree ===")
    print(tracer.render())
    print("\n=== Summary ===")
    print(tracer.render_summary())

    path = tracer.save_json("trace_output.json")
    print(f"\nTrace saved to {path} "
          f"({sum(s.total_tokens for s in tracer._walk())} tokens tracked)")


if __name__ == "__main__":
    main()
