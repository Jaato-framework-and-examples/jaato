# Template Plugin

The template plugin provides Jinja2-based template rendering and automatic extraction of embedded templates from documentation.

## Features

### 1. Template Rendering (`renderTemplate`)

Render Jinja2 templates with variable substitution and write results to files.

```python
renderTemplate(
    template="Hello {{ name }}, welcome to {{ project }}!",
    variables={"name": "Alice", "project": "jaato"},
    output_path="greeting.txt"
)
```

Or use a template file:

```python
renderTemplate(
    template_path=".jaato/templates/service.java.tmpl",
    variables={"className": "OrderService", "package": "com.example"},
    output_path="src/main/java/com/example/OrderService.java"
)
```

### 2. Automatic Template Extraction (Prompt Enrichment)

The plugin subscribes to prompt enrichment to automatically detect and extract templates embedded in documentation files (like `MODULE.md`).

**Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent reads MODULE.md containing:                              │
│                                                                 │
│  ## Template 1: Basic with Fallback                             │
│  ```java                                                        │
│  @CircuitBreaker(name = "{{circuitBreakerName}}", ...)         │
│  ```                                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Template plugin's enrich_prompt() is called                    │
│                                                                 │
│  1. Detects code blocks with {{ }} or {% %} syntax             │
│  2. Extracts to .jaato/templates/mod-code-001-basic.java.tmpl  │
│  3. Annotates prompt with extraction info                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent sees annotated prompt:                                   │
│                                                                 │
│  ---                                                            │
│  **Extracted Templates:**                                       │
│  [Template extracted: .jaato/templates/mod-code-001-basic...]   │
│    Variables: circuitBreakerName, fallbackMethodName, ...       │
│    Use: renderTemplate(template_path="...", variables={...})    │
│  ---                                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Simple Template Rendering (`renderTemplateToFile`)

A convenience tool for simple `{{variable}}` substitution that writes directly to a file:

```python
renderTemplateToFile(
    output_path="/src/main/java/com/bank/CustomerService.java",
    template_path="/templates/service.java.tmpl",
    variables={"class_name": "CustomerService", "package": "com.bank.customer"}
)
```

Or with an inline template:

```python
renderTemplateToFile(
    output_path="/src/CustomerService.java",
    template="package {{package}};\n\npublic class {{class_name}} {}",
    variables={"class_name": "CustomerService", "package": "com.bank"}
)
```

**Key differences from `renderTemplate`:**
- Uses simple `{{variable}}` substitution only (no Jinja2 conditionals/loops)
- Has `overwrite` parameter (default: `false`) - errors if file exists
- Returns `bytes_written` instead of `size`/`lines`
- Creates parent directories automatically

**Return format:**
```json
{
  "success": true,
  "output_path": "/src/CustomerService.java",
  "bytes_written": 1234,
  "variables_used": ["class_name", "package"],
  "template_source": "file"
}
```

**Error handling:**
```json
{
  "error": "Undefined variable: author",
  "template_path": "/templates/service.java.tmpl",
  "variables_provided": ["class_name", "package"]
}
```

### 4. List Available Templates (`listAvailableTemplates`)

View all templates that have been extracted in the current session:

```python
listAvailableTemplates()
# Returns:
# {
#   "templates": [
#     {"path": ".jaato/templates/mod-code-001-basic.java.tmpl",
#      "variables": ["circuitBreakerName", "fallbackMethodName", ...]}
#   ],
#   "count": 1
# }
```

## Template Syntax

### For `renderTemplate` (Full Jinja2)

| Syntax | Description | Example |
|--------|-------------|---------|
| `{{ var }}` | Variable substitution | `{{ className }}` |
| `{% if %}` | Conditional | `{% if admin %}...{% endif %}` |
| `{% for %}` | Loop | `{% for item in items %}...{% endfor %}` |
| `{{ x \| filter }}` | Filters | `{{ name \| upper }}` |

### For `renderTemplateToFile` (Simple)

| Syntax | Description | Example |
|--------|-------------|---------|
| `{{var}}` | Variable substitution | `{{className}}` |
| `\{{` | Escape literal `{{` | `\{{not_a_var}}` → `{{not_a_var}}` |

The simple syntax only supports variable substitution—no conditionals, loops, or filters.

## Enrichment Priority

The template plugin runs at **priority 40** in the prompt enrichment pipeline:

| Priority | Plugin | Purpose |
|----------|--------|---------|
| 20 | references | Injects MODULE.md content into prompt |
| **40** | **template** | **Extracts embedded templates** |
| 60 | multimodal | Handles @image references |
| 80 | memory | Adds memory hints |
| 90 | session | Session description hints |

This ordering ensures the template plugin sees content injected by the references plugin.

## Template Naming

Extracted templates are named based on context:

1. **Frontmatter ID**: If the document has `id: mod-code-001`, it's used as prefix
2. **Heading**: Nearby `## Template:` headings provide the template name
3. **Hash fallback**: Content hash if no context available

Example: `mod-code-001-basic-with-fallback.java.tmpl`

## Configuration

```python
registry.expose_tool("template", {
    "base_path": "/path/to/project",  # Optional: override base path
    "agent_name": "main"              # For trace logging
})
```

## Storage

Extracted templates are stored in `.jaato/templates/`:

```
.jaato/
└── templates/
    ├── mod-code-001-basic-with-fallback.java.tmpl
    ├── mod-code-001-config.yaml.tmpl
    └── mod-code-002-retry-pattern.java.tmpl
```

This directory can be gitignored as templates are extracted on-demand.

## Security

**For `renderTemplate` (Jinja2):**
- Uses Jinja2's `SandboxedEnvironment` to prevent arbitrary code execution
- `{% include %}` and `{% import %}` are disabled
- `StrictUndefined` mode catches typos in variable names

**For `renderTemplateToFile` (simple):**
- Uses regex-based substitution (no code execution risk)
- Validates all template variables are provided
- Prevents accidental file overwrite (requires explicit `overwrite=true`)

**Both tools:**
- Require user approval (file writes need permission)
- Create parent directories safely

## Dependencies

Jinja2 is required for `renderTemplate` only:

```bash
pip install Jinja2
```

The plugin gracefully reports if Jinja2 is not installed when `renderTemplate` is called.

**Note:** `renderTemplateToFile` does not require Jinja2—it uses built-in regex substitution.
