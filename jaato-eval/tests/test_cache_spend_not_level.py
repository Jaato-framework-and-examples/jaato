"""Cache figures recorded as spend, not as summed level readings.

`cache_read_tokens` / `cache_creation_tokens` are the LAST RESPONSE's
figures — a level. `spend_cache_read_tokens` /
`spend_cache_creation_tokens` are summed over the turn's responses, the
same shape as `spend_total_tokens`. The SDK documents the distinction as
load-bearing: under `model_tiers` a mid-turn tier switch re-reads the
whole prefix cold at the new model, and the last-response figures hide
exactly that miss.

The runner summed the level pair across turns, which produced neither a
level nor a spend, and recorded the spend pair nowhere — so no consumer
could compute cache economics from a results file. The fingerprint was
visible in the archived corpus: three of four Gemini arms reported
`cache_creation` equal to `cache_read` to the token, which is one level
reading copied into two fields rather than two independent billed sums
(jaato #800).

The comment above `_SUMMED_USAGE` already ruled `total_tokens` out for
precisely this reason; these tests hold the cache pair to the same rule.
"""

import unittest

from jaato_eval.runner import _SUMMED_USAGE, _TurnAccumulator


class _Usage:
    """The usage object carried on a TurnCompletedEvent."""

    def __init__(self, **kw):
        # Everything absent unless the test sets it, as on a real event.
        for key in ("prompt_tokens", "output_tokens", "total_tokens",
                    "spend_total_tokens", "cache_read_tokens",
                    "cache_creation_tokens", "spend_cache_read_tokens",
                    "spend_cache_creation_tokens", "reasoning_tokens",
                    "thinking_tokens", "cost_usd"):
            setattr(self, key, kw.get(key))


class _Turn:
    def __init__(self, usage):
        self.usage = usage


class SummedKeysCase(unittest.TestCase):
    """Which keys the accumulator is allowed to add up."""

    def test_the_spend_cache_pair_is_summed(self):
        self.assertIn("spend_cache_read_tokens", _SUMMED_USAGE)
        self.assertIn("spend_cache_creation_tokens", _SUMMED_USAGE)

    def test_the_level_cache_pair_is_not_summed(self):
        """A level added across turns is neither a level nor a spend."""
        self.assertNotIn("cache_read_tokens", _SUMMED_USAGE)
        self.assertNotIn("cache_creation_tokens", _SUMMED_USAGE)

    def test_total_tokens_stays_excluded(self):
        """The rule the cache pair now follows, stated for the same reason."""
        self.assertNotIn("total_tokens", _SUMMED_USAGE)


class AccumulationCase(unittest.TestCase):

    def _run(self, *turns):
        acc = _TurnAccumulator()
        for usage in turns:
            acc.on_turn(_Turn(usage))
        return acc.snapshot()

    def test_spend_cache_accumulates_across_turns(self):
        snap = self._run(
            _Usage(spend_cache_read_tokens=100, spend_cache_creation_tokens=40),
            _Usage(spend_cache_read_tokens=250, spend_cache_creation_tokens=10),
        )
        self.assertEqual(snap["spend_cache_read_tokens"], 350)
        self.assertEqual(snap["spend_cache_creation_tokens"], 50)

    def test_level_readings_are_not_added_to_the_spend_figures(self):
        """The defect: level readings arriving on the event must not land
        in the recorded totals, however large they are."""
        snap = self._run(
            _Usage(cache_read_tokens=97590, cache_creation_tokens=97590,
                   spend_cache_read_tokens=100, spend_cache_creation_tokens=40),
        )
        self.assertEqual(snap["spend_cache_read_tokens"], 100)
        self.assertEqual(snap["spend_cache_creation_tokens"], 40)
        self.assertNotIn("cache_read_tokens", snap)
        self.assertNotIn("cache_creation_tokens", snap)

    def test_a_provider_reporting_no_cache_records_zero_not_none(self):
        """Absent on the event is 0 spend, which is a real observation —
        distinct from the column being unavailable."""
        snap = self._run(_Usage(spend_total_tokens=500))
        self.assertEqual(snap["spend_cache_read_tokens"], 0)
        self.assertEqual(snap["spend_cache_creation_tokens"], 0)

    def test_cache_share_is_computable_from_a_results_file(self):
        """The consumer-facing point: both figures are spend, same shape,
        so the ratio has a defensible meaning."""
        snap = self._run(
            _Usage(spend_total_tokens=1000, spend_cache_read_tokens=750),
            _Usage(spend_total_tokens=1000, spend_cache_read_tokens=250),
        )
        share = snap["spend_cache_read_tokens"] / snap["spend_total_tokens"]
        self.assertAlmostEqual(share, 0.5)

    def test_spend_total_is_unaffected(self):
        snap = self._run(_Usage(spend_total_tokens=10),
                         _Usage(spend_total_tokens=32))
        self.assertEqual(snap["spend_total_tokens"], 42)
