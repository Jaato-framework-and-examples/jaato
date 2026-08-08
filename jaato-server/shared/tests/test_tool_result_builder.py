"""Unit tests for shared.tool_result_builder.

Locks the behavior of the pure tool-result shape-normalization helpers
extracted from JaatoSession._build_tool_result: executor-result splitting,
multimodal attachment extraction, and result-dict normalization (internal
key stripping, permission-note surfacing, single-key error collapse).
"""

from __future__ import annotations

from shared.tool_result_builder import (
    extract_multimodal_attachments,
    normalize_result_dict,
    split_executor_result,
)


class TestSplitExecutorResult:
    def test_tuple_form(self):
        assert split_executor_result((False, {"error": "x"})) == (False, {"error": "x"})

    def test_bare_dict_is_success(self):
        assert split_executor_result({"result": 1}) == (True, {"result": 1})

    def test_bare_string_is_success(self):
        assert split_executor_result("done") == (True, "done")

    def test_non_two_tuple_is_bare_value(self):
        # A 3-tuple is not the (ok, data) form -> treated as a bare payload.
        val = (1, 2, 3)
        assert split_executor_result(val) == (True, val)


class TestExtractMultimodalAttachments:
    def test_image_attachment(self):
        out = extract_multimodal_attachments({
            "_multimodal": True,
            "_multimodal_type": "image",
            "image_data": b"\x89PNG",
            "mime_type": "image/png",
            "display_name": "shot",
        })
        assert out is not None and len(out) == 1
        assert out[0].mime_type == "image/png"
        assert out[0].data == b"\x89PNG"
        assert out[0].display_name == "shot"

    def test_image_defaults(self):
        out = extract_multimodal_attachments({"image_data": b"x"})
        assert out[0].mime_type == "image/png"
        assert out[0].display_name == "image"

    def test_missing_image_data_returns_none(self):
        assert extract_multimodal_attachments({"_multimodal_type": "image"}) is None

    def test_file_attachment(self):
        # readFile on a .pdf emits _multimodal_type:"file" with file_data;
        # extract pulls it into the generic Attachment for PDF marshalling.
        out = extract_multimodal_attachments({
            "_multimodal": True,
            "_multimodal_type": "file",
            "file_data": b"%PDF-1.4 bytes",
            "mime_type": "application/pdf",
            "display_name": "doc.pdf",
        })
        assert out is not None and len(out) == 1
        assert out[0].mime_type == "application/pdf"
        assert out[0].data == b"%PDF-1.4 bytes"
        assert out[0].display_name == "doc.pdf"

    def test_missing_file_data_returns_none(self):
        assert extract_multimodal_attachments({"_multimodal_type": "file"}) is None

    def test_unknown_type_returns_none(self):
        assert extract_multimodal_attachments({"_multimodal_type": "audio"}) is None


class TestNormalizeResultDict:
    def test_wraps_non_dict(self):
        assert normalize_result_dict(42, ok=True) == {"result": 42}

    def test_strips_internal_keys(self):
        out = normalize_result_dict(
            {"result": "ok", "_permission": {"x": 1}, "_multimodal": True},
            ok=True,
        )
        assert out == {"result": "ok"}

    def test_surfaces_permission_comment(self):
        out = normalize_result_dict(
            {"result": "ok", "_permission": {"comment": "be careful"}},
            ok=True,
        )
        assert out == {"result": "ok", "permission_note": "be careful"}

    def test_single_key_error_collapses_to_string(self):
        out = normalize_result_dict({"error": "boom", "_trace": "x"}, ok=False)
        assert out == "boom"

    def test_multi_key_error_stays_dict(self):
        out = normalize_result_dict({"error": "boom", "code": 1}, ok=False)
        assert out == {"error": "boom", "code": 1}

    def test_error_string_not_collapsed_when_ok(self):
        # ok=True path never collapses, even if an 'error' key is present.
        out = normalize_result_dict({"error": "boom"}, ok=True)
        assert out == {"error": "boom"}
