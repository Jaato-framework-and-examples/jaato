"""Tests for MemoryPlugin.

Covers core CRUD, prompt enrichment, tag validation, and the
knowledge-curation lifecycle fields (maturity, confidence, scope, evidence).
"""

import json
import tempfile
import unittest
from pathlib import Path

from ..models import (
    MATURITY_DISMISSED,
    MATURITY_ESCALATED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    SCOPE_PROJECT,
    SCOPE_UNIVERSAL,
    Memory,
)
from ..plugin import MemoryPlugin
from ..storage import MemoryStorage


class TestMemoryPlugin(unittest.TestCase):
    """Test cases for MemoryPlugin."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = str(Path(self.temp_dir) / "test_memories.jsonl")

        self.global_storage_path = str(Path(self.temp_dir) / "global_memories.jsonl")

        # Initialize plugin with both storage paths in temp dir
        self.plugin = MemoryPlugin()
        self.plugin.initialize({
            "storage_path": self.storage_path,
            "global_storage_path": self.global_storage_path,
        })

    def tearDown(self):
        """Clean up after tests."""
        self.plugin.shutdown()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _promote(self, memory_id: str, maturity: str = "validated"):
        """Simulate the curator promoting a raw memory to the curated store.

        After this call the memory is retrievable via tag/ID search and
        appears in enrichment hints.  Tests that need this end-state
        should call ``_promote(mid)`` immediately after ``store_memory``.
        """
        return self.plugin.get_executors()["update_memory"]({
            "id": memory_id,
            "maturity": maturity,
        })

    def test_plugin_name(self):
        """Test plugin has correct name."""
        self.assertEqual(self.plugin.name, "memory")

    def test_tool_schemas(self):
        """Test plugin provides expected tool schemas."""
        schemas = self.plugin.get_tool_schemas()
        tool_names = [s.name for s in schemas]

        self.assertIn("store_memory", tool_names)
        self.assertIn("retrieve_memories", tool_names)
        self.assertIn("list_memory_tags", tool_names)

    def test_store_memory_schema_has_lifecycle_params(self):
        """Test that store_memory schema includes confidence, scope, evidence."""
        schemas = self.plugin.get_tool_schemas()
        store_schema = next(s for s in schemas if s.name == "store_memory")
        props = store_schema.parameters["properties"]

        self.assertIn("confidence", props)
        self.assertIn("scope", props)
        self.assertIn("evidence", props)

    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving a memory."""
        # Store a memory
        store_result = self.plugin.get_executors()["store_memory"]({
            "content": "The Runtime/Session split allows efficient subagent spawning.",
            "description": "jaato architecture explanation",
            "tags": ["architecture", "runtime", "session"]
        })

        self.assertEqual(store_result["status"], "success")
        self.assertIn("memory_id", store_result)

        # Promote raw → curated so the memory is retrievable.
        self._promote(store_result["memory_id"])

        # Retrieve by tags
        retrieve_result = self.plugin.get_executors()["retrieve_memories"]({
            "tags": ["architecture"],
            "limit": 5
        })

        self.assertEqual(retrieve_result["status"], "success")
        self.assertEqual(retrieve_result["count"], 1)
        self.assertEqual(
            retrieve_result["memories"][0]["description"],
            "jaato architecture explanation"
        )

    def test_store_defaults_to_raw_maturity(self):
        """Test that newly stored memories default to raw maturity."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test content",
            "description": "Test",
            "tags": ["testing"]
        })

        self.assertEqual(result["maturity"], MATURITY_RAW)
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(result["scope"], SCOPE_PROJECT)

    def test_store_with_lifecycle_fields(self):
        """Test storing a memory with explicit lifecycle fields."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "PostgreSQL JSONB indexes don't support partial matching",
            "description": "JSONB indexing limitation",
            "tags": ["postgresql", "jsonb_indexing"],
            "confidence": 0.9,
            "scope": "universal",
            "evidence": "Tested with CREATE INDEX ... USING gin and partial match query returned seq scan"
        })

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["maturity"], MATURITY_RAW)
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["scope"], SCOPE_UNIVERSAL)

    def test_store_clamps_confidence(self):
        """Test that confidence is clamped to 0.0-1.0 range."""
        executors = self.plugin.get_executors()

        # Over 1.0
        result = executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["test_clamp"],
            "confidence": 5.0,
        })
        self.assertEqual(result["confidence"], 1.0)

        # Below 0.0
        result2 = executors["store_memory"]({
            "content": "Test2",
            "description": "Test2",
            "tags": ["test_clamp2"],
            "confidence": -1.0,
        })
        self.assertEqual(result2["confidence"], 0.0)

    def test_store_invalid_scope_defaults_to_project(self):
        """Test that invalid scope values fall back to 'project'."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["test_scope"],
            "scope": "galactic",
        })

        self.assertEqual(result["scope"], SCOPE_PROJECT)

    def test_retrieve_includes_lifecycle_fields(self):
        """Test that retrieved memories include maturity, confidence, scope."""
        executors = self.plugin.get_executors()

        store = executors["store_memory"]({
            "content": "Some insight",
            "description": "Test insight",
            "tags": ["lifecycle_test"],
            "confidence": 0.8,
            "scope": "universal",
        })
        # Promote so it's retrievable.  Maturity stays as set by promotion.
        self._promote(store["memory_id"])

        result = executors["retrieve_memories"]({"tags": ["lifecycle_test"]})
        mem = result["memories"][0]

        self.assertEqual(mem["maturity"], "validated")
        self.assertEqual(mem["confidence"], 0.8)
        self.assertEqual(mem["scope"], SCOPE_UNIVERSAL)

    def test_list_tags(self):
        """Test listing memory tags."""
        # Store some memories AND promote them so they enter the index.
        executors = self.plugin.get_executors()
        m1 = executors["store_memory"]({
            "content": "Auth explanation",
            "description": "Authentication flow",
            "tags": ["auth", "security"]
        })
        m2 = executors["store_memory"]({
            "content": "DB explanation",
            "description": "Database schema",
            "tags": ["database", "schema"]
        })
        self._promote(m1["memory_id"])
        self._promote(m2["memory_id"])

        # List tags
        result = executors["list_memory_tags"]({})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memory_count"], 2)
        self.assertIn("auth", result["tags"])
        self.assertIn("database", result["tags"])

    def test_prompt_enrichment(self):
        """Test prompt enrichment with memory hints.

        Requires at least 2 tag overlaps (min_overlap=2 in the indexer)
        to prevent false-positive matches from large prompts.
        """
        # Store + promote so the memory enters the curated index that
        # enrichment uses.
        store = self.plugin.get_executors()["store_memory"]({
            "content": "Detailed subagent explanation",
            "description": "How to spawn subagents efficiently",
            "tags": ["subagent", "spawn", "efficiently"]
        })
        self._promote(store["memory_id"])

        # Prompt must contain at least 2 keywords matching tags
        result = self.plugin.enrich_prompt(
            "How do I spawn a subagent efficiently?"
        )

        # Should find the memory and add hints
        self.assertIn("💡 **Available Memories**", result.prompt)
        self.assertIn("subagent", result.prompt.lower())
        self.assertEqual(result.metadata["memory_matches"], 1)

    def test_escalated_memories_hidden_from_enrichment(self):
        """Test that escalated memories are NOT surfaced in prompt enrichment.

        Once a memory has been promoted to a reference by the advisor,
        it should no longer appear in memory hints — the reference takes over.
        """
        executors = self.plugin.get_executors()

        # Store and then manually escalate a memory
        result = executors["store_memory"]({
            "content": "Barrel imports prevent circular deps",
            "description": "Always use barrel file for model imports",
            "tags": ["barrel_imports", "circular_deps"]
        })

        # Manually escalate it via storage
        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_ESCALATED
        self.plugin._storage.update(memory)

        # Rebuild index to pick up the change
        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        # Enrichment should NOT find the escalated memory
        enrichment = self.plugin.enrich_prompt(
            "How should I handle barrel imports to avoid circular deps?"
        )
        self.assertEqual(enrichment.metadata.get("memory_matches", 0), 0)

    def test_dismissed_memories_hidden_from_enrichment(self):
        """Test that dismissed memories are NOT surfaced in prompt enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Wrong lesson",
            "description": "Incorrect observation",
            "tags": ["wrong_lesson_test"]
        })

        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_DISMISSED
        self.plugin._storage.update(memory)

        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        enrichment = self.plugin.enrich_prompt(
            "Tell me about wrong_lesson_test"
        )
        self.assertEqual(enrichment.metadata.get("memory_matches", 0), 0)

    def test_validated_memories_visible_in_enrichment(self):
        """Test that validated memories ARE surfaced in prompt enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Validated insight about patterns",
            "description": "Confirmed pattern usage",
            "tags": ["validated", "pattern", "usage"]
        })

        memory = self.plugin._storage.get_by_id(result["memory_id"])
        memory.maturity = MATURITY_VALIDATED
        self.plugin._storage.update(memory)

        self.plugin._indexer.clear()
        self.plugin._indexer.build_index(self.plugin._storage.load_all())

        enrichment = self.plugin.enrich_prompt(
            "How to use validated pattern correctly?"
        )
        self.assertEqual(enrichment.metadata["memory_matches"], 1)

    def test_no_enrichment_when_no_matches(self):
        """Test that prompts without matches are not modified."""
        original_prompt = "What is the weather today?"

        result = self.plugin.enrich_prompt(original_prompt)

        # Prompt should be unchanged
        self.assertEqual(result.prompt, original_prompt)
        self.assertEqual(result.metadata["memory_matches"], 0)

    def test_auto_approved_tools(self):
        """Test that memory tools are auto-approved."""
        auto_approved = self.plugin.get_auto_approved_tools()

        self.assertIn("store_memory", auto_approved)
        self.assertIn("retrieve_memories", auto_approved)
        self.assertIn("list_memory_tags", auto_approved)

    def test_subscribes_to_enrichment(self):
        """Test plugin subscribes to prompt enrichment."""
        self.assertTrue(self.plugin.subscribes_to_prompt_enrichment())

    def test_usage_count_increments(self):
        """Test that usage count increments on retrieval."""
        executors = self.plugin.get_executors()

        # Store memory and promote to curated so it's retrievable.
        store = executors["store_memory"]({
            "content": "Test content",
            "description": "Test memory",
            "tags": ["test"]
        })
        self._promote(store["memory_id"])

        # Retrieve once
        result1 = executors["retrieve_memories"]({"tags": ["test"]})
        self.assertEqual(result1["memories"][0]["usage_count"], 1)

        # Retrieve again
        result2 = executors["retrieve_memories"]({"tags": ["test"]})
        self.assertEqual(result2["memories"][0]["usage_count"], 2)

    def test_single_letter_tags_rejected(self):
        """Test that single-letter tags are rejected."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Some content",
            "description": "Test memory",
            "tags": ["a", "b", "c"]
        })

        self.assertEqual(result["status"], "error")
        self.assertIn("at least 2 characters", result["error"])

    def test_short_tags_filtered_valid_kept(self):
        """Test that short tags are filtered but valid tags are kept."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Authentication details",
            "description": "Auth flow",
            "tags": ["a", "authentication", "x", "oauth"]
        })

        # Should succeed with only the valid tags
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tags"], ["authentication", "oauth"])

    def test_validate_memory_schema_rejects_short_tags(self):
        """Test that _validate_memory_schema rejects short tags."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["a", "b"]
        })

        self.assertIsNotNone(error)
        self.assertIn("at least 2 characters", error)

    def test_validate_memory_schema_accepts_valid_tags(self):
        """Test that _validate_memory_schema accepts valid tags."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["authentication", "oauth"]
        })

        self.assertIsNone(error)

    def test_validate_memory_schema_rejects_invalid_maturity(self):
        """Test that _validate_memory_schema rejects invalid maturity."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "maturity": "legendary",
        })

        self.assertIsNotNone(error)
        self.assertIn("maturity", error)

    def test_validate_memory_schema_accepts_valid_maturity(self):
        """Test that _validate_memory_schema accepts valid maturity values."""
        for maturity in (MATURITY_RAW, MATURITY_VALIDATED, MATURITY_ESCALATED, MATURITY_DISMISSED):
            error = self.plugin._validate_memory_schema({
                "content": "test",
                "description": "test",
                "tags": ["test_tag"],
                "maturity": maturity,
            })
            self.assertIsNone(error, f"maturity={maturity} should be valid")

    def test_validate_memory_schema_rejects_invalid_confidence(self):
        """Test that _validate_memory_schema rejects out-of-range confidence."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "confidence": 2.0,
        })

        self.assertIsNotNone(error)
        self.assertIn("confidence", error)

    def test_validate_memory_schema_rejects_invalid_scope(self):
        """Test that _validate_memory_schema rejects invalid scope."""
        error = self.plugin._validate_memory_schema({
            "content": "test",
            "description": "test",
            "tags": ["test_tag"],
            "scope": "galactic",
        })

        self.assertIsNotNone(error)
        self.assertIn("scope", error)

    def test_store_result_includes_telemetry_dict(self):
        """Test that store_memory result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        result = executors["store_memory"]({
            "content": "Test telemetry emission",
            "description": "Telemetry test",
            "tags": ["telemetry_test"],
            "confidence": 0.8,
            "scope": "universal",
            "evidence": "Observed during testing",
        })

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "store")
        self.assertEqual(telem["jaato.memory.maturity"], "raw")
        self.assertEqual(telem["jaato.memory.confidence"], 0.8)
        self.assertEqual(telem["jaato.memory.scope"], "universal")
        self.assertTrue(telem["jaato.memory.has_evidence"])
        self.assertEqual(telem["jaato.memory.tag_count"], 1)

    def test_retrieve_result_includes_telemetry_dict(self):
        """Test that retrieve_memories result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        m1 = executors["store_memory"]({
            "content": "Insight A",
            "description": "First insight",
            "tags": ["telem_retrieve_test"],
            "confidence": 0.7,
            "scope": "project",
        })
        m2 = executors["store_memory"]({
            "content": "Insight B",
            "description": "Second insight",
            "tags": ["telem_retrieve_test"],
            "confidence": 0.9,
            "scope": "universal",
        })
        # Promote both so they're retrievable.
        self._promote(m1["memory_id"])
        self._promote(m2["memory_id"])

        result = executors["retrieve_memories"]({"tags": ["telem_retrieve_test"]})

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "retrieve")
        self.assertEqual(telem["jaato.memory.count_retrieved"], 2)
        # Maturity is now "validated" after promotion (no longer raw).
        self.assertIn("validated", telem["jaato.memory.maturities_retrieved"])
        self.assertIn("project", telem["jaato.memory.scopes_retrieved"])
        self.assertIn("universal", telem["jaato.memory.scopes_retrieved"])
        self.assertAlmostEqual(telem["jaato.memory.avg_confidence"], 0.8, places=2)

    def test_list_tags_result_includes_telemetry_dict(self):
        """Test that list_memory_tags result includes _telemetry for span enrichment."""
        executors = self.plugin.get_executors()

        store = executors["store_memory"]({
            "content": "Test",
            "description": "Test",
            "tags": ["telem_list_test"],
        })
        # list_memory_tags reads the indexer (curated only) — promote
        # so the memory enters the index.
        self._promote(store["memory_id"])

        result = executors["list_memory_tags"]({})

        self.assertIn("_telemetry", result)
        telem = result["_telemetry"]
        self.assertEqual(telem["jaato.memory.operation"], "list_tags")
        self.assertEqual(telem["jaato.memory.total_count"], 1)
        self.assertGreaterEqual(telem["jaato.memory.tag_count"], 1)
        # After promotion the memory is validated, not raw.
        self.assertEqual(telem["jaato.memory.count_validated"], 1)

    def test_source_agent_captured(self):
        """Test that source_agent is captured from plugin config."""
        plugin = MemoryPlugin()
        plugin.initialize({
            "storage_path": str(Path(self.temp_dir) / "agent_test.jsonl"),
            "agent_name": "analyst-codebase-documentation",
        })

        result = plugin.get_executors()["store_memory"]({
            "content": "Test from named agent",
            "description": "Agent source test",
            "tags": ["agent_test"],
        })

        memory = plugin._storage.get_by_id(result["memory_id"])
        self.assertEqual(memory.source_agent, "analyst-codebase-documentation")
        plugin.shutdown()


class TestSplitStorage(unittest.TestCase):
    """Tests for the raw/curated split storage layout.

    ``MemoryStore.save()`` writes new memories to the raw queue;
    they don't appear in the curated path until promoted via
    ``update()`` with a non-raw, non-dismissed maturity.  Replaces
    the old single-file backward-compat suite (no migration is
    provided per the dev-stage refactor).
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # MemoryStorage accepts a ``.jsonl`` path for compat — the dir
        # is created adjacent (``.../mem.jsonl`` → ``.../mem/``).
        self.storage_path = str(Path(self.temp_dir) / "mem.jsonl")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _store(self, **kwargs) -> Memory:
        defaults = {
            "id": "mem_x",
            "content": "x",
            "description": "x",
            "tags": ["t"],
            "timestamp": "2024-01-01T00:00:00",
            "maturity": MATURITY_RAW,
        }
        defaults.update(kwargs)
        return Memory(**defaults)

    def test_save_routes_to_raw_queue(self):
        store = MemoryStorage(self.storage_path)
        store.save(self._store(id="mem_1"))
        # Curated is empty; raw has the new memory.
        self.assertEqual(store.load_curated(), [])
        raw = store.list_raw()
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].id, "mem_1")

    def test_update_promotes_raw_to_curated(self):
        store = MemoryStorage(self.storage_path)
        m = self._store(id="mem_2", maturity=MATURITY_RAW)
        store.save(m)
        # Promote: same memory with new maturity.
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        # Now curated has it; raw is empty.
        self.assertEqual(len(store.list_raw()), 0)
        curated = store.load_curated()
        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0].id, "mem_2")
        self.assertEqual(curated[0].maturity, MATURITY_VALIDATED)

    def test_update_dismiss_unlinks_from_raw(self):
        store = MemoryStorage(self.storage_path)
        m = self._store(id="mem_3")
        store.save(m)
        m.maturity = "dismissed"
        store.update(m)
        # Gone from both stores.
        self.assertEqual(len(store.list_raw()), 0)
        self.assertEqual(store.load_curated(), [])

    def test_update_modifies_existing_curated(self):
        store = MemoryStorage(self.storage_path)
        # Get a memory into curated first.
        m = self._store(id="mem_4")
        store.save(m)
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        # Now modify its tags.
        m.tags = ["new-tag"]
        store.update(m)
        curated = store.load_curated()
        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0].tags, ["new-tag"])

    def test_search_by_tags_skips_raw(self):
        """Tag search must NOT surface raw memories."""
        store = MemoryStorage(self.storage_path)
        store.save(self._store(id="mem_raw", tags=["secret-tag"]))
        results = store.search_by_tags(["secret-tag"])
        self.assertEqual(results, [])
        # After promotion it surfaces.
        m = store.get_by_id("mem_raw")
        m.maturity = MATURITY_VALIDATED
        store.update(m)
        results = store.search_by_tags(["secret-tag"])
        self.assertEqual(len(results), 1)

    def test_get_by_id_searches_both_stores(self):
        store = MemoryStorage(self.storage_path)
        store.save(self._store(id="mem_in_raw"))
        m2 = self._store(id="mem_in_curated", maturity=MATURITY_VALIDATED)
        store.save(m2)
        store.update(m2)  # promote
        self.assertIsNotNone(store.get_by_id("mem_in_raw"))
        self.assertIsNotNone(store.get_by_id("mem_in_curated"))
        self.assertIsNone(store.get_by_id("mem_does_not_exist"))

    def test_count_by_maturity_combines_both_stores(self):
        store = MemoryStorage(self.storage_path)
        store.save(self._store(id="raw_a"))
        store.save(self._store(id="raw_b"))
        m = self._store(id="curated_v", maturity=MATURITY_VALIDATED)
        store.save(m)
        store.update(m)
        counts = store.count_by_maturity()
        self.assertEqual(counts.get(MATURITY_RAW), 2)
        self.assertEqual(counts.get(MATURITY_VALIDATED), 1)

    def test_workspace_and_global_stores_are_isolated(self):
        """Two MemoryStore instances pointing at sibling .jsonl paths
        must not share storage (each gets its own folder)."""
        ws_path = str(Path(self.temp_dir) / "ws.jsonl")
        global_path = str(Path(self.temp_dir) / "global.jsonl")
        ws = MemoryStorage(ws_path)
        gl = MemoryStorage(global_path)
        ws.save(self._store(id="mem_ws"))
        gl.save(self._store(id="mem_gl"))
        # Each store sees only its own raw memory.
        ws_raw_ids = {m.id for m in ws.list_raw()}
        gl_raw_ids = {m.id for m in gl.list_raw()}
        self.assertEqual(ws_raw_ids, {"mem_ws"})
        self.assertEqual(gl_raw_ids, {"mem_gl"})


class TestSetSessionAutoWiring(unittest.TestCase):
    """Server 0.6.167+ — ``set_session(session)`` auto-wiring + the
    ``source_session`` provenance field on stored memories.

    Pre-0.6.167 the plugin defined ``set_session_context(session_id)``
    but the framework's canonical wiring is
    ``plugin.set_session(session)`` (five call sites in
    ``shared/jaato_session.py``).  Method-name mismatch meant the
    plugin's ``_session_id`` stayed None forever → every memory ever
    written had ``source_session=null``.

    Empirical motivation (peer 7:1, 2026-05-28): jq audit of
    kb-enablement-2.0 store showed 25/25 memories with
    source_session=null despite source_agent populated.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = str(Path(self.temp_dir) / "mem.jsonl")
        self.global_storage_path = str(
            Path(self.temp_dir) / "global_mem.jsonl"
        )
        self.plugin = MemoryPlugin()
        self.plugin.initialize({
            "storage_path": self.storage_path,
            "global_storage_path": self.global_storage_path,
        })
        # Isolate the per-execution session ContextVar — _get_session_id
        # reads the current session from it.  Baseline = no session.
        from shared.session_context import _current_session
        self._sess_token = _current_session.set(None)

    def _set_current_session(self, daemon_session_id):
        """Wire a fake current session carrying ``daemon_session_id``
        (no parent) into the ContextVar that ``_get_session_id`` reads."""
        from shared.session_context import _current_session
        fake = type("FakeSession", (), {
            "_daemon_session_id": daemon_session_id,
            "_parent_session": None,
        })()
        _current_session.set(fake)

    def tearDown(self):
        from shared.session_context import _current_session
        _current_session.reset(self._sess_token)
        self.plugin.shutdown()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _promote(self, memory_id: str, maturity: str = "validated"):
        return self.plugin.get_executors()["update_memory"]({
            "id": memory_id,
            "maturity": maturity,
        })

    def test_set_session_is_noop_no_self_state(self):
        """``set_session`` must NOT store session state on ``self`` —
        plugin instances are shared across sibling subagents, so a
        sibling's call would clobber another's.  Enforced by
        ``tests/test_plugin_session_safety.py``; pinned here too.
        """
        # A fabricated session with _daemon_session_id must NOT leak onto
        # the instance (the old stashing behavior).
        fake_session = type("FakeSession", (), {})()
        fake_session._daemon_session_id = "20260528_223344"
        self.plugin.set_session(fake_session)
        self.assertIsNone(self.plugin._session_id)
        # None / bare sessions are tolerated without raising.
        self.plugin.set_session(None)
        self.plugin.set_session(type("Bare", (), {})())

    def test_store_memory_carries_source_session_via_config_injection(self):
        """End-to-end pin: the source_session field is populated from the
        config-injection path (``initialize`` reads ``config['session_id']``,
        injected runner-side by ``registry._augment_plugin_config``), NOT
        from ``set_session``/``_daemon_session_id`` (which is always None
        runner-side).  Checks the on-disk Memory dataclass directly.
        """
        # Re-initialize with the session_id the framework injects via config.
        self.plugin.initialize({
            "storage_path": self.storage_path,
            "global_storage_path": self.global_storage_path,
            "session_id": "20260528_223344",
        })

        store_result = self.plugin.get_executors()["store_memory"]({
            "content": "session-id propagation pin",
            "description": "verifies source_session populated",
            "tags": ["pin-test"],
        })
        self.assertEqual(store_result["status"], "success")
        mid = store_result["memory_id"]

        persisted = self.plugin._storage.raw.get(mid)
        self.assertIsNotNone(persisted, f"memory {mid} not in raw store")
        self.assertEqual(
            persisted.source_session, "20260528_223344",
            "source_session must be populated via config injection — the "
            "real runner-side path (set_session/_daemon_session_id is None "
            "runner-side; peer 7:1 retry-49: 4/4 source_session=null).",
        )

    def test_store_memory_source_session_is_none_when_set_session_not_called(self):
        """Backwards-compat: if no caller invokes set_session, the
        plugin keeps the pre-0.6.167 behavior (source_session=None
        on persisted memories).  Standalone plugin constructors
        (kb-side scripts, premium extensions) keep working
        unchanged."""
        store_result = self.plugin.get_executors()["store_memory"]({
            "content": "no-wire fallback pin",
            "description": "verifies fallback to None",
            "tags": ["pin-test-no-wire"],
        })
        mid = store_result["memory_id"]
        persisted = self.plugin._storage.raw.get(mid)
        self.assertIsNotNone(persisted)
        self.assertIsNone(persisted.source_session)

    # ------------------------------------------------------------------
    # Server 0.6.168+ — runner-side wiring via config + registry
    # ------------------------------------------------------------------

    def test_initialize_reads_session_id_from_config(self):
        """Server 0.6.168+: the canonical runner-side wiring path.
        ``registry.set_session_id(envelope.session_id)`` is called
        at ``runner/session.py:271``; the registry's
        ``_augment_plugin_config`` injects it into the plugin's
        config dict at ``initialize()`` time.  This is what
        actually fires for cascade sessions — memory is
        ``PLUGIN_TIER="runner"``.

        PR-196's ``set_session(session)`` reads
        ``session._daemon_session_id`` which is set only by
        JaatoClient (daemon-side facade), never on runner-side
        JaatoSession.  Empirical: peer 7:1 retry-49 post-PR-196
        still showed source_session=null on 4/4 memories.  This
        path (config injection) IS reached for runner-side init.
        """
        sub_plugin = MemoryPlugin()
        sub_plugin.initialize({
            "storage_path": str(Path(self.temp_dir) / "sub_mem.jsonl"),
            "global_storage_path": str(
                Path(self.temp_dir) / "sub_global.jsonl"
            ),
            "session_id": "20260530_223344",
        })
        try:
            self.assertEqual(sub_plugin._session_id, "20260530_223344")
        finally:
            sub_plugin.shutdown()

    def test_get_session_id_reads_current_session_not_registry_or_cache(self):
        """``_get_session_id`` reads the CURRENT session's
        ``_daemon_session_id`` (per-execution), NOT the shared
        ``registry._session_id`` and NOT the cached ``self._session_id``.

        The registry is shared across sibling subagents, so reading its
        single ``_session_id`` leaked the last-bootstrapped sibling's id.
        The current session (each sibling has its own) is the correct,
        per-sibling source.  Pin: with a stale cache AND a (now-ignored)
        registry value AND a different current-session id, the
        current-session id wins.
        """
        fake_registry = type("FakeRegistry", (), {})()
        fake_registry._session_id = "registry_sibling_other"  # must be ignored
        self.plugin._session_id = "cached_first_session"       # stale cache
        self.plugin.set_plugin_registry(fake_registry)

        self._set_current_session("current_session_A")
        self.assertEqual(self.plugin._get_session_id(), "current_session_A")
        # Registry advancing (a sibling bootstrapping) must NOT change us.
        fake_registry._session_id = "registry_sibling_yet_another"
        self.assertEqual(self.plugin._get_session_id(), "current_session_A")
        # Switching the executing session DOES (the per-execution source).
        self._set_current_session("current_session_B")
        self.assertEqual(self.plugin._get_session_id(), "current_session_B")

    def test_get_session_id_falls_back_to_cached_when_no_current_session(self):
        """With no session in context (standalone plugin / no real turn),
        ``_get_session_id`` falls back to the cached config-injection
        value — preserves behavior for tests and kb-side scripts that
        construct the plugin directly."""
        self.plugin._session_id = "20260530_standalone"
        # setUp seeded the session ContextVar to None (no current session).
        self.assertEqual(
            self.plugin._get_session_id(), "20260530_standalone"
        )

    def test_store_memory_picks_live_session_id_per_execution(self):
        """End-to-end pin: the same plugin instance reused across
        multiple sessions (cascade-pool HIT slot AND/OR sibling
        subagents) must persist ``source_session`` matching the
        CURRENTLY-EXECUTING session, not the first-session value cached
        at plugin init and not a shared registry value.

        Covers two empirical scenarios at once: (1) peer 7:1 retry-49
        HIT-slot reuse where the cached _session_id went stale, and
        (2) the sibling-subagent leak where the shared
        registry._session_id reflected the last-bootstrapped sibling.
        Both are fixed by reading the per-execution current session.
        """
        # A stale registry value that MUST be ignored throughout.
        fake_registry = type("FakeRegistry", (), {})()
        fake_registry._session_id = "stale_shared_registry"
        self.plugin.set_plugin_registry(fake_registry)

        self._set_current_session("session_1")
        r1 = self.plugin.get_executors()["store_memory"]({
            "content": "first store",
            "description": "session_1 memory",
            "tags": ["pin-hit-1"],
        })
        self._set_current_session("session_2")
        r2 = self.plugin.get_executors()["store_memory"]({
            "content": "second store",
            "description": "session_2 memory",
            "tags": ["pin-hit-2"],
        })
        self._set_current_session("session_3")
        r3 = self.plugin.get_executors()["store_memory"]({
            "content": "third store",
            "description": "session_3 memory",
            "tags": ["pin-hit-3"],
        })
        for rid, expected in [
            (r1["memory_id"], "session_1"),
            (r2["memory_id"], "session_2"),
            (r3["memory_id"], "session_3"),
        ]:
            persisted = self.plugin._storage.raw.get(rid)
            self.assertIsNotNone(persisted)
            self.assertEqual(
                persisted.source_session, expected,
                f"HIT-reused slot: memory {rid} should carry "
                f"source_session={expected!r} (the registry's "
                f"current value at call time), not the slot's "
                f"first-session value.",
            )


class TestMemoryPluginFrameworkWiring(unittest.TestCase):
    """Pin that the framework's set_session(plugin, session) call
    sites actually exercise the memory plugin's set_session.  Source-
    level grep — confirms the wiring isn't broken by a future
    refactor that, e.g., renames set_session or drops one of the
    five jaato_session.py call sites."""

    def test_jaato_session_calls_plugin_set_session(self):
        """At least one ``plugin.set_session(self)`` call exists in
        ``shared/jaato_session.py`` — the canonical wiring memory
        plugin's set_session relies on."""
        from pathlib import Path
        here = Path(__file__).resolve()
        repo_root = here.parents[5]  # tests/ → memory/ → plugins/ → shared/ → jaato-server/ → repo
        js = repo_root / "jaato-server" / "shared" / "jaato_session.py"
        self.assertTrue(js.is_file(), f"jaato_session.py not found at {js}")
        src = js.read_text()
        # ``plugin.set_session(self)`` is the canonical pattern in
        # jaato_session.py — five call sites at time of writing
        # (1927/5405/5678/8295/8358).  At least one must exist or
        # the framework wiring is broken.
        self.assertIn(
            "plugin.set_session(self)", src,
            "jaato_session.py missing 'plugin.set_session(self)' — "
            "the canonical plugin auto-wiring pattern.  Memory "
            "plugin's set_session won't fire without it; "
            "source_session will silently regress to None.",
        )


if __name__ == "__main__":
    unittest.main()
