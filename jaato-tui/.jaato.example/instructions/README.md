# System Instructions

Markdown files in this directory are loaded as base system instructions for all
agents. They are injected at the start of the conversation as part of the system
prompt.

## Naming Convention

Files are loaded in alphabetical order, so use numeric prefixes to control the
order:

```
instructions/
├── 00-system-instructions.md    # Foundation: role, policies, constraints
├── 10-coding-standards.md       # Project-specific coding conventions
└── 20-domain-knowledge.md       # Domain-specific terminology and rules
```

## Tips

- Keep instructions concise -- they consume context window space on every turn
- Focus on policies, constraints, and conventions (not task-specific instructions)
- Use markdown headers to organize sections
- Test that your instructions fit within the model's context alongside a
  reasonable conversation
