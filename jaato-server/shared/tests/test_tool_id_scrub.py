"""Scrub provider tool-ids (t_xxxxxxxx / c_xxxxxxxx) out of user-facing model
narration — incl. ids split across streaming chunks (StreamScrubber)."""
from shared.tool_id_map import name_to_id, scrub_tool_ids, StreamScrubber


def test_scrub_replaces_known_id_bare_and_markdown():
    tid = name_to_id("docker_status")                  # populates _reverse
    assert tid.startswith("t_")
    assert scrub_tool_ids(f"I used the {tid} tool") == "I used the docker_status tool"
    assert scrub_tool_ids(f"the `{tid}` tool") == "the `docker_status` tool"


def test_scrub_replaces_markdown_escaped_underscore():
    # Small models emit ids inside **bold** / list items and backslash-escape
    # the underscore (t\_xxxxxxxx) so markdown doesn't italicize it. The scrubber
    # must still rewrite these — otherwise the raw id leaks (observed live).
    tid = name_to_id("browser_navigate")
    escaped = tid.replace("_", "\\_")                   # t\_xxxxxxxx
    assert scrub_tool_ids(f"* **{escaped}**: navigate") == "* **browser_navigate**: navigate"
    assert scrub_tool_ids(f"I used {escaped} to go there") == "I used browser_navigate to go there"


def test_scrub_leaves_unknown_id_unchanged():
    assert scrub_tool_ids("t_deadbeef ran") == "t_deadbeef ran"   # never issued


def test_scrub_category_prefix():
    cid = name_to_id("docker", prefix="c")
    assert cid.startswith("c_")
    assert scrub_tool_ids(f"in the {cid} category") == "in the docker category"


def test_stream_scrubber_id_split_across_chunks():
    tid = name_to_id("generic_run")
    mid = len(tid) // 2
    s = StreamScrubber()
    out = s.feed(f"used the {tid[:mid]}")
    out += s.feed(f"{tid[mid:]} tool")
    out += s.flush()
    assert "generic_run" in out
    assert tid not in out                               # raw id never surfaced


def test_stream_scrubber_complete_id_in_one_feed():
    tid = name_to_id("solo_tool")
    s = StreamScrubber()
    out = s.feed(f"ran {tid}.") + s.flush()
    assert out == "ran solo_tool."


def test_stream_scrubber_lone_trailing_letter_flushes_clean():
    s = StreamScrubber()
    out = s.feed("about") + s.feed(" cats") + s.flush()
    assert out == "about cats"
