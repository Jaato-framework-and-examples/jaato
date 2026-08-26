"""Ledger reconstruction must report its own unfaithfulness.

The SDK's serialized ``function_call`` part carries no ``call_id``, so
calls cannot be paired to responses by id.  The whole point of this
module is that it says so rather than producing a plausible pairing.
"""
import unittest

from jaato_eval.ledger import build_ledger


def _call(name, args=None, call_id=None):
    part = {"type": "function_call", "name": name, "args": args or {}}
    if call_id is not None:
        part["call_id"] = call_id
    return part


def _response(name, call_id, response, is_error=False):
    return {"type": "function_response", "name": name, "call_id": call_id,
            "response": response, "is_error": is_error}


class TestLedger(unittest.TestCase):
    def test_no_call_ids_is_unfaithful(self):
        """The shape the SDK actually emits today."""
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"})]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        result = build_ledger(history)
        self.assertFalse(result.faithful)
        self.assertIn("call_id", result.reason)
        self.assertIn("_serialize_part", result.reason)

    def test_with_call_ids_is_faithful(self):
        """The shape it would emit after the one-line serializer fix."""
        history = [
            {"role": "model", "parts": [_call("writeFile", {"path": "a"}, "c1")]},
            {"role": "tool", "parts": [_response("writeFile", "c1", {"ok": True})]},
        ]
        result = build_ledger(history)
        self.assertTrue(result.faithful)
        self.assertEqual(result.reason, "")
        self.assertEqual(len(result.entries), 1)
        self.assertTrue(result.entries[0]["success"])

    def test_is_error_drives_success_not_result_shape(self):
        """``is_error`` is the framework's boundary-computed flag; the
        inner result dict is the plugin's private convention."""
        history = [
            {"role": "model", "parts": [_call("build", {}, "c1")]},
            {"role": "tool", "parts": [_response("build", "c1",
                                                 {"status": "success"}, is_error=True)]},
        ]
        result = build_ledger(history)
        self.assertFalse(result.entries[0]["success"])

    def test_unpaired_call_is_counted_and_marked(self):
        history = [{"role": "model", "parts": [_call("pending", {}, "c9")]}]
        result = build_ledger(history)
        self.assertEqual(result.unpaired_calls, 1)
        self.assertEqual(result.entries[0]["result"], {"error": "no_response"})
        self.assertFalse(result.entries[0]["success"])

    def test_no_tool_calls_is_not_unfaithful(self):
        """A run with no tool calls has nothing to pair; that is not a
        reconstruction failure and must not block graders."""
        result = build_ledger([{"role": "model", "parts": [{"type": "text", "text": "hi"}]}])
        self.assertTrue(result.faithful)
        self.assertEqual(result.entries, [])


if __name__ == "__main__":
    unittest.main()
