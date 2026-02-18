"""Tests for InstructionTokenCache."""

import threading
from shared.instruction_token_cache import InstructionTokenCache


class TestInstructionTokenCache:
    """Unit tests for the content-addressed token count cache."""

    def test_get_miss_returns_none(self):
        cache = InstructionTokenCache()
        assert cache.get("anthropic", "hello world") is None

    def test_put_then_get(self):
        cache = InstructionTokenCache()
        cache.put("anthropic", "hello world", 42)
        assert cache.get("anthropic", "hello world") == 42

    def test_provider_isolation(self):
        """Same text under different providers should be independent."""
        cache = InstructionTokenCache()
        cache.put("anthropic", "hello", 10)
        cache.put("google_genai", "hello", 20)
        assert cache.get("anthropic", "hello") == 10
        assert cache.get("google_genai", "hello") == 20

    def test_different_texts_different_keys(self):
        cache = InstructionTokenCache()
        cache.put("p", "text_a", 100)
        cache.put("p", "text_b", 200)
        assert cache.get("p", "text_a") == 100
        assert cache.get("p", "text_b") == 200

    def test_overwrite(self):
        """Putting the same key again should overwrite."""
        cache = InstructionTokenCache()
        cache.put("p", "text", 10)
        cache.put("p", "text", 20)
        assert cache.get("p", "text") == 20

    def test_len(self):
        cache = InstructionTokenCache()
        assert len(cache) == 0
        cache.put("p", "a", 1)
        assert len(cache) == 1
        cache.put("p", "b", 2)
        assert len(cache) == 2
        # Same key doesn't increase length
        cache.put("p", "a", 3)
        assert len(cache) == 2

    def test_concurrent_access(self):
        """Multiple threads writing/reading shouldn't corrupt state."""
        cache = InstructionTokenCache()
        errors = []

        def writer(provider: str, start: int, count: int):
            try:
                for i in range(start, start + count):
                    cache.put(provider, f"text_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(provider: str, start: int, count: int):
            try:
                for i in range(start, start + count):
                    result = cache.get(provider, f"text_{i}")
                    # Result is either None (not yet written) or the correct value
                    if result is not None:
                        assert result == i
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(4):
            threads.append(threading.Thread(target=writer, args=(f"p{t}", 0, 100)))
            threads.append(threading.Thread(target=reader, args=(f"p{t}", 0, 100)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_repr(self):
        cache = InstructionTokenCache()
        assert "entries=0" in repr(cache)
        cache.put("p", "text", 1)
        assert "entries=1" in repr(cache)
