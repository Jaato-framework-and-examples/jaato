"""_discover_prompts is mtime-gated (re-attach loop-stall regression guard).

Pre-fix, ``_discover_prompts`` re-walked the filesystem on EVERY call — an
uncached per-item realpath/symlink walk.  On a cold daemon-side instance during
re-attach that ran ~15s ON the asyncio event-loop thread (inside the synchronous
tool-id-registry emit in websocket._register_client_tools), starving the very
register-RPC send it was performing (the self-block).  The fix: re-walk only
when a source dir's mtime changes; explicit "prompts changed" signals
(``_notify_tools_changed``) still force a re-walk so content edits land.
"""
import os
import tempfile
import time
from pathlib import Path

from ..plugin import PromptLibraryPlugin


def _plugin_with_prompt(tmpdir):
    prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "test.md").write_text("# Test")
    plugin = PromptLibraryPlugin()
    plugin.set_workspace_path(tmpdir)
    return plugin, prompts_dir


def _count_walks(plugin):
    """Count ``_load_prompt_info`` calls — non-zero only on an actual walk
    (a cache hit returns before the per-item loop)."""
    n = {"v": 0}
    orig = plugin._load_prompt_info

    def counting(*a, **k):
        n["v"] += 1
        return orig(*a, **k)

    plugin._load_prompt_info = counting
    return n


def test_unchanged_sources_skip_the_walk():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin, _ = _plugin_with_prompt(tmpdir)
        walks = _count_walks(plugin)
        plugin._discover_prompts()                 # cold walk
        first = walks["v"]
        assert first > 0                           # walked
        plugin._discover_prompts()                 # unchanged -> cache hit
        assert walks["v"] == first                 # NO re-walk


def test_source_dir_change_triggers_rewalk():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin, prompts_dir = _plugin_with_prompt(tmpdir)
        walks = _count_walks(plugin)
        plugin._discover_prompts()
        first = walks["v"]
        plugin._discover_prompts()
        assert walks["v"] == first                 # cache hit
        # Add a prompt + bump the dir mtime deterministically (dodge the
        # 1s-resolution same-second flake).
        (prompts_dir / "test2.md").write_text("# Test2")
        future = time.time() + 10
        os.utime(prompts_dir, (future, future))
        plugin._discover_prompts()
        assert walks["v"] > first                  # re-walked


def test_notify_tools_changed_forces_rewalk():
    # A content edit to an EXISTING prompt doesn't bump the dir mtime; the
    # explicit "prompts changed" signal must force a re-walk anyway.
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin, _ = _plugin_with_prompt(tmpdir)
        walks = _count_walks(plugin)
        plugin._discover_prompts()
        first = walks["v"]
        plugin._discover_prompts()
        assert walks["v"] == first                 # cache hit
        plugin._notify_tools_changed([])           # explicit refresh signal
        assert walks["v"] > first                  # forced re-walk
