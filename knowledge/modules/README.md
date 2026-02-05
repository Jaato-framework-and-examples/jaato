# Modules

> Reusable template packages for code generation

**Version:** 3.0
**Last Updated:** 2026-01-20

## Purpose

Modules are **reusable building blocks** that provide templates and rules for code generation:
- Each module encapsulates templates for a specific pattern
- Modules are referenced by capability features in capability-index.yaml
- Modules include Tier-3 validation scripts
- Modules declare their stack in frontmatter

## Structure

```
modules/
├── README.md
│
└── mod-code-{NNN}-{pattern}-{framework}/
    ├── MODULE.md           # Specification + Template Catalog
    ├── templates/          # Template files (.tpl)
    │   ├── {concern}/      # Organized by architectural concern
    │   │   └── *.tpl
    │   └── config/
    │       └── *.tpl
    └── validation/         # Tier-3 validators
        ├── README.md
        └── {pattern}-check.sh
```

## Naming Convention

```
mod-{domain}-{NNN}-{pattern}-{framework}
```

| Component | Example |
|-----------|---------|
| `domain` | code |
| `NNN` | 001, 015 (3-digit unique ID) |
| `pattern` | circuit-breaker, hexagonal-base |
| `framework` | java-spring, java-resilience4j |

## Current Modules

### By Capability

| Capability | Feature | Module | Stack |
|------------|---------|--------|-------|
| **architecture** | hexagonal-light | mod-code-015-hexagonal-base-java-spring | java-spring |
| **api-architecture** | domain-api | mod-code-019-api-public-exposure-java-spring | java-spring |
| **integration** | api-rest | mod-code-018-api-integration-rest-java-spring | java-spring |
| **persistence** | jpa | mod-code-016-persistence-jpa-spring | java-spring |
| **persistence** | systemapi | mod-code-017-persistence-systemapi | java-spring |
| **resilience** | circuit-breaker | mod-code-001-circuit-breaker-java-resilience4j | java-spring |
| **resilience** | retry | mod-code-002-retry-java-resilience4j | java-spring |
| **resilience** | timeout | mod-code-003-timeout-java-resilience4j | java-spring |
| **resilience** | rate-limiter | mod-code-004-rate-limiter-java-resilience4j | java-spring |
| **distributed-transactions** | saga-compensation | mod-code-020-compensation-java-spring | java-spring |

## Module Frontmatter (v3.0)

Every module declares its capability mapping:

```yaml
---
id: mod-code-001-circuit-breaker-java-resilience4j
version: 2.1
status: Active
derived_from: eri-code-008-circuit-breaker-java-resilience4j
domain: code

implements:
  stack: java-spring          # REQUIRED in v3.0
  pattern: annotation         # Optional (if alternatives exist)
  capability: resilience
  feature: circuit-breaker
---
```

## Template Format

Templates use **Mustache/Handlebars syntax**:

```java
// Template: {{Entity}}Service.java.tpl
package {{basePackage}}.domain.service;

public class {{Entity}}Service {
    {{#fields}}
    private {{type}} {{name}};
    {{/fields}}
}
```

## Validation (Tier-3)

Each module includes validation scripts in `validation/`:
- Scripts verify pattern-specific constraints
- Called after generation during validation phase
- Complement Tier-1 (universal) and Tier-2 (technology) validators

## Relationship to Capability Index

Modules are referenced from capability-index.yaml:

```yaml
# In capability-index.yaml
resilience:
  features:
    circuit-breaker:
      implementations:
        - id: java-spring-resilience4j
          module: mod-code-001-circuit-breaker-java-resilience4j  # ← Reference
          stack: java-spring
          pattern: annotation
```

## Related

- Capability Index: `/runtime/discovery/capability-index.yaml`
- Tier-1/2 validators: `/runtime/validators/`
- Authoring guide: `/model/standards/authoring/MODULE.md`
- Capability documentation: `/model/domains/code/capabilities/`
