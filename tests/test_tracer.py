"""Tests for llm-tracer. Stdlib only: `python -m unittest discover tests`."""

import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer import Span, Tracer, estimate_tokens  # noqa: E402


class TestSpanBasics(unittest.TestCase):
    def test_nested_spans_form_tree(self):
        t = Tracer()
        with t.span("parent"):
            with t.span("child"):
                pass
        self.assertEqual(len(t.roots), 1)
        parent = t.roots[0]
        self.assertEqual(parent.name, "parent")
        self.assertEqual(len(parent.children), 1)
        child = parent.children[0]
        self.assertEqual(child.name, "child")
        self.assertEqual(child.parent_id, parent.span_id)

    def test_sibling_roots(self):
        t = Tracer()
        with t.span("a"):
            pass
        with t.span("b"):
            pass
        self.assertEqual([r.name for r in t.roots], ["a", "b"])

    def test_duration_is_positive(self):
        t = Tracer()
        with t.span("x"):
            pass
        self.assertGreater(t.roots[0].duration_ms, 0)

    def test_error_is_recorded_not_swallowed(self):
        t = Tracer()
        with self.assertRaises(ValueError):
            with t.span("boom"):
                raise ValueError("kaput")
        span = t.roots[0]
        self.assertIn("ValueError", span.error)
        self.assertTrue(any(e["name"] == "error" for e in span.events))


class TestUsageAndCost(unittest.TestCase):
    def test_token_accumulation(self):
        t = Tracer()
        with t.span("llm", model="gpt-4o-mini") as s:
            s.record_usage(prompt_tokens=100, completion_tokens=50)
            s.record_usage(prompt_tokens=10, completion_tokens=5)
        self.assertEqual(t.roots[0].total_tokens, 165)

    def test_cost_math(self):
        pricing = {"m": (2.0, 8.0)}  # per 1M tokens
        t = Tracer(pricing=pricing)
        with t.span("llm", model="m") as s:
            s.record_usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        self.assertAlmostEqual(t.roots[0].cost_usd, 2.0 + 4.0)

    def test_unknown_model_cost_is_none(self):
        t = Tracer()
        with t.span("llm", model="not-a-model") as s:
            s.record_usage(prompt_tokens=100, completion_tokens=100)
        self.assertIsNone(t.roots[0].cost_usd)

    def test_summary_aggregates(self):
        t = Tracer()
        for _ in range(3):
            with t.span("gen", model="gpt-4o-mini") as s:
                s.record_usage(prompt_tokens=100, completion_tokens=10)
        rows = {r["name"]: r for r in t.summary()}
        self.assertEqual(rows["gen"]["calls"], 3)
        self.assertEqual(rows["gen"]["prompt_tokens"], 300)
        self.assertEqual(rows["gen"]["completion_tokens"], 30)


class TestDecorator(unittest.TestCase):
    def test_trace_decorator_uses_function_name(self):
        t = Tracer()

        @t.trace()
        def my_step():
            return 42

        self.assertEqual(my_step(), 42)
        self.assertEqual(t.roots[0].name, "my_step")

    def test_trace_decorator_injects_span(self):
        t = Tracer()

        @t.trace(model="gpt-4o-mini")
        def gen(__span__=None):
            __span__.record_usage(prompt_tokens=7, completion_tokens=3)
            return "ok"

        self.assertEqual(gen(), "ok")
        self.assertEqual(t.roots[0].total_tokens, 10)


class TestExportRender(unittest.TestCase):
    def test_json_round_trip(self):
        t = Tracer()
        with t.span("root", model="gpt-4o") as s:
            s.record_usage(prompt_tokens=10, completion_tokens=5)
            with t.span("child"):
                pass
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.json")
            t.save_json(path)
            data = json.load(open(path))
        self.assertEqual(data["traces"][0]["name"], "root")
        self.assertEqual(data["traces"][0]["children"][0]["name"], "child")
        self.assertEqual(data["traces"][0]["prompt_tokens"], 10)

    def test_render_contains_tree(self):
        t = Tracer()
        with t.span("root"):
            with t.span("child"):
                pass
        out = t.render()
        self.assertIn("root", out)
        self.assertIn("child", out)
        self.assertIn("ms", out)

    def test_render_summary_table(self):
        t = Tracer()
        with t.span("root"):
            pass
        out = t.render_summary()
        self.assertIn("root", out)
        self.assertIn("calls", out)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_spans_all_recorded(self):
        t = Tracer()

        def work(i):
            with t.span(f"worker-{i}"):
                pass

        threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(len(t.roots), 20)


class TestSlowestAndBudget(unittest.TestCase):
    def _tracer_with_known_durations(self):
        """Spans with deterministic durations (set directly, not slept)."""
        t = Tracer(pricing={"m": (2.0, 8.0)})
        with t.span("slow_llm", model="m") as s:
            s.record_usage(prompt_tokens=1_000_000, completion_tokens=500_000)
            with t.span("fast_child"):
                pass
        slow, child = t.roots[0], t.roots[0].children[0]
        slow.end_time = slow.start_time + 0.500   # 500ms
        child.end_time = child.start_time + 0.010  # 10ms
        return t, slow, child

    def test_slowest_orders_by_duration_and_shows_path(self):
        t, slow, child = self._tracer_with_known_durations()
        rows = t.slowest(n=2)
        self.assertEqual(rows[0]["name"], "slow_llm")
        self.assertAlmostEqual(rows[0]["duration_ms"], 500.0, places=0)
        self.assertEqual(rows[1]["name"], "fast_child")
        self.assertEqual(rows[1]["path"], "slow_llm > fast_child")

    def test_totals_and_cost(self):
        t, _, _ = self._tracer_with_known_durations()
        self.assertEqual(t.total_tokens(), 1_500_000)
        self.assertAlmostEqual(t.total_cost_usd(), 6.0)

    def test_total_cost_none_when_no_priced_spans(self):
        t = Tracer()
        with t.span("plain"):
            pass
        self.assertIsNone(t.total_cost_usd())

    def test_check_budget_flags_cost_and_slow_span(self):
        t, _, _ = self._tracer_with_known_durations()
        warnings = t.check_budget(max_cost_usd=5.0, max_span_ms=100.0)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("exceeds budget" in w for w in warnings))
        self.assertTrue(any("slow_llm" in w for w in warnings))

    def test_check_budget_silent_when_within_budget(self):
        t, _, _ = self._tracer_with_known_durations()
        self.assertEqual(t.check_budget(max_cost_usd=10.0, max_span_ms=1000.0), [])

    def test_check_budget_no_cost_warning_without_priced_spans(self):
        t = Tracer()
        with t.span("plain"):
            pass
        self.assertEqual(t.check_budget(max_cost_usd=0.01), [])


class TestHelpers(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertGreater(estimate_tokens("hello world, this is a test"), 0)


if __name__ == "__main__":
    unittest.main()
