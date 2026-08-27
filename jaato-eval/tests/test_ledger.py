"""The ledger comes from the SDK; this package only guards daemon version.

jaato #639 put the call identifier on the wire and #640 put the single
pairing rule in ``jaato_sdk.completion_processors.build_ledger``. What is
left here is one question the SDK cannot answer, because it is a property
of the deployment rather than the data: did this daemon emit identifiers
at all?

These tests run against the REAL SDK builder loaded from the checkout —
see ``tests/_real_sdk.py`` for why a stub would be worse than useless.
"""
import unittest

from jaato_eval.ledger import build_ledger_result, history_carries_call_ids

from . import _real_sdk

_SDK = _real_sdk.install()


def _call(name, args=None, call_id=None):
    part = {"type": "function_call", "name": name, "args": args or {}}
    if call_id is not None:
        part["call_id"] = call_id
    return part


def _response(name, call_id, response, is_error=False):
    return {"type": "function_response", "name": name, "call_id": call_id,
            "response": response, "is_error": is_error}


class GuardCase(unittest.TestCase):
    """``history_carries_call_ids`` witnesses the key, not the outcome."""

    def test_current_daemon_carries_ids(self):
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"}, "c1")]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        self.assertTrue(history_carries_call_ids(history))

    def test_pre_639_daemon_does_not(self):
        """The shape the wire had before jaato #639."""
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"})]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        self.assertFalse(history_carries_call_ids(history))

    def test_no_calls_is_not_a_failure_to_pair(self):
        self.assertTrue(history_carries_call_ids(
            [{"role": "model", "parts": [{"type": "text", "text": "hi"}]}]))

    def test_one_missing_id_condemns_the_history(self):
        """Partial identifiers are worse than none: the paired subset would
        look authoritative."""
        history = [{"role": "model", "parts": [_call("a", {}, "c1"), _call("b", {})]}]
        self.assertFalse(history_carries_call_ids(history))


@unittest.skipUnless(_SDK, "real jaato_sdk.completion_processors not in checkout")
class AgainstRealSDKCase(unittest.TestCase):
    """Integration with the shipped pairing rule."""

    def setUp(self):
        # Re-install per test rather than relying on import-time state:
        # tests that stub the SDK clear it from sys.modules in their own
        # cleanup, so depending on collection order here would make this
        # class pass or fail based on the alphabet.
        _real_sdk.install()

    def test_pairs_and_reports_faithful(self):
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"}, "c1")]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        result = build_ledger_result(history)
        self.assertTrue(result.faithful)
        self.assertEqual(result.reason, "")
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0]["call_id"], "c1")
        self.assertTrue(result.entries[0]["success"])

    def test_retry_is_attributed_to_the_right_call(self):
        """The case name-in-order pairing gets wrong, and the reason the
        identifier had to reach the wire: same tool, first call errors,
        second succeeds. The failure must stay on the first."""
        history = [
            {"role": "model", "parts": [_call("render", {"v": 1}, "c1")]},
            {"role": "tool", "parts": [_response("render", "c1", {"error": "bad var"})]},
            {"role": "model", "parts": [_call("render", {"v": 2}, "c2")]},
            {"role": "tool", "parts": [_response("render", "c2", {"path": "out.java"})]},
        ]
        result = build_ledger_result(history)
        self.assertTrue(result.faithful)
        by_id = {e["call_id"]: e for e in result.entries}
        self.assertFalse(by_id["c1"]["success"])
        self.assertTrue(by_id["c2"]["success"])
        self.assertEqual(by_id["c1"]["args"], {"v": 1})

    def test_unpaired_call_counted(self):
        history = [{"role": "model", "parts": [_call("pending", {}, "c9")]}]
        result = build_ledger_result(history)
        self.assertEqual(result.unpaired_calls, 1)
        self.assertFalse(result.entries[0]["success"])

    def test_pre_639_history_is_unfaithful_and_says_why(self):
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"})]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        result = build_ledger_result(history)
        self.assertFalse(result.faithful)
        self.assertIn("#639", result.reason)

    def test_no_tool_calls_is_faithful_and_empty(self):
        result = build_ledger_result(
            [{"role": "model", "parts": [{"type": "text", "text": "hi"}]}])
        self.assertTrue(result.faithful)
        self.assertEqual(result.entries, [])


if __name__ == "__main__":
    unittest.main()
