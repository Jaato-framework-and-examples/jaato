# Prompts Architecture

## Files

- **SPEC.md** - LLM-agnostic execution logic (source of truth)
- **claude.txt** - Claude-optimized (XML format)
- **gemini.txt** - Gemini-optimized (Markdown format)

## Module Usage

SPEC.md references module:
```markdown
Use module: ../../modules/mod-code-001-circuit-breaker-java-resilience4j.md
```

Both claude.txt and gemini.txt are generated from SPEC.md.

---

**Last Updated:** 2025-11-21
