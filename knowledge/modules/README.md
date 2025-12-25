# Modules

> Reusable template packages for code generation

## Purpose

Modules are **reusable building blocks** that skills use to generate code:
- Each module encapsulates templates for a specific pattern
- Modules include Tier-3 validation scripts
- Currently used by CODE domain skills
- Other domains may not require modules

## Structure

```
modules/
├── README.md
│
└── mod-code-{NNN}-{pattern}-{framework}-{library}/
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
mod-{domain}-{NNN}-{pattern}-{framework}-{library}
```

| Component | Example |
|-----------|---------|
| `domain` | code (currently only CODE domain) |
| `NNN` | 001, 015 (3-digit unique ID) |
| `pattern` | circuit-breaker, hexagonal-base |
| `framework` | java, spring |
| `library` | resilience4j, jpa (optional) |

## Current Modules

### Resilience (mod-code-001 to 004)
| Module | Pattern |
|--------|---------|
| mod-code-001-circuit-breaker-java-resilience4j | Circuit Breaker |
| mod-code-002-retry-java-resilience4j | Retry |
| mod-code-003-timeout-java-resilience4j | Timeout |
| mod-code-004-rate-limiter-java-resilience4j | Rate Limiter |

### Architecture (mod-code-015)
| Module | Pattern |
|--------|---------|
| mod-code-015-hexagonal-base-java-spring | Hexagonal Light base |

### Persistence (mod-code-016, 017)
| Module | Pattern |
|--------|---------|
| mod-code-016-persistence-jpa-spring | JPA persistence adapter |
| mod-code-017-persistence-systemapi | System API client adapter |

### Integration (mod-code-018)
| Module | Pattern |
|--------|---------|
| mod-code-018-api-integration-rest-java-spring | REST API integration |

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
- Called by skill's validation orchestrator
- Complement Tier-1 (universal) and Tier-2 (technology) validators

## Related

- Skills (consumers): `/skills/`
- Tier-1/2 validators: `/runtime/validators/`
- Authoring guide: `/model/standards/authoring/MODULE.md`
