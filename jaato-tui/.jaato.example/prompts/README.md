# Prompt Templates

Reusable prompt templates for common tasks. These can be loaded via the
`/prompt-library` command or referenced programmatically.

## Format

Each prompt is a markdown file with optional YAML frontmatter:

```markdown
---
description: Brief description of what this prompt does
params:
  city:
    required: true
    description: The city to look up
  language:
    required: false
    default: English
    description: Response language
tags: ['weather', 'utility']
---

Your prompt text here. Use {{city}} and {{language}} for parameter substitution.
```

## Parameters

- `description`: Shown when browsing the prompt library
- `params`: Named parameters with `required`, `default`, and `description`
- `tags`: For filtering and organization

Parameters are substituted using `{{param_name}}` syntax in the prompt body.

## Examples

- **weather-forecast-spain.md** -- Get a weather forecast for a Spanish city
- **code-review.md** -- Perform a structured code review on changed files
