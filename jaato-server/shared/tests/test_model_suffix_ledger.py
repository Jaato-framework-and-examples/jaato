"""ToolResult.model_suffix — model-facing steering kept OFF the structured result.

The task-completion spur / mid-turn piggyback / withheld-attachment note used to
be ``str()``-folded into ``ToolResult.result``, turning a structured dict into a
Python-repr string.  That broke every consumer that reads the result
structurally — notably the tool-call ledger / completion-processor provenance
(``result.get('invoice_total')`` → None because the value was double-buried
under the ledger's non-dict wrap of the repr-string).

Fix: the steering rides on ``ToolResult.model_suffix`` and is appended to the
serialized result ONLY at provider-serialization time
(``render_result_for_model`` for text providers; a reserved ``_agent_guidance``
key for dict-response providers).  ``result`` stays the structured source of
truth for the ledger / GC / enrichment.
"""
from types import SimpleNamespace

from jaato_sdk.plugins.model_provider.types import (
    ToolResult, Part, render_result_for_model,
)
from shared.completion_processors import build_tool_call_ledger


# ---- render_result_for_model ------------------------------------------------

def test_render_dict_is_clean_json_plus_suffix():
    out = render_result_for_model({"invoice_total": 250, "status": "issued"},
                                  "<hidden>keep going</hidden>")
    head, _, tail = out.partition("\n\n")
    assert head == '{"invoice_total": 250, "status": "issued"}'   # JSON, not repr
    assert "'" not in head                                        # double quotes
    assert tail == "<hidden>keep going</hidden>"


def test_render_str_result_and_no_suffix():
    assert render_result_for_model("plain", None) == "plain"
    assert render_result_for_model({"a": 1}, None) == '{"a": 1}'


# ---- text-content converters carry the suffix, result stays structured ------

def test_anthropic_converter_appends_suffix_and_keeps_result_structured():
    from shared.plugins.model_provider.anthropic import converters as a
    fr = ToolResult(call_id="c1", name="issue_refund",
                    result={"invoice_total": 250}, model_suffix="<hidden>go</hidden>")
    block = a.part_to_anthropic_content_block(Part(function_response=fr))
    # content carries the JSON result + the suffix
    assert '"invoice_total": 250' in block["content"]
    assert "<hidden>go</hidden>" in block["content"]
    # the ToolResult itself was NOT mutated — result stays the structured dict
    assert fr.result == {"invoice_total": 250}


# ---- END-TO-END: ledger reads the STRUCTURED result (suffix ignored) ---------

def test_ledger_reads_structured_result_despite_model_suffix():
    # This is the shape AFTER the spur under the fix: result stays the flat dict,
    # steering is on model_suffix.  The ledger (which reads history) must see the
    # flat dict so provenance .get('invoice_total') succeeds.
    fc = SimpleNamespace(id="c1", name="issue_refund", args={"invoice_id": "INV-1007"})
    fr = ToolResult(call_id="c1", name="issue_refund",
                    result={"refund_id": "rf_demo_0001", "invoice_total": 250,
                            "status": "issued"},
                    model_suffix="<hidden>VERIFICATION PROTOCOL...</hidden>")
    history = [
        SimpleNamespace(parts=[SimpleNamespace(function_call=fc, function_response=None)]),
        SimpleNamespace(parts=[SimpleNamespace(function_call=None, function_response=fr)]),
    ]
    ledger = build_tool_call_ledger(history)
    entry = next(e for e in ledger if e["name"] == "issue_refund")
    assert entry["result"].get("invoice_total") == 250
    assert entry["result"].get("status") == "issued"
    # the steering suffix must NOT have leaked into the ledgered result
    assert "VERIFICATION PROTOCOL" not in str(entry["result"])
