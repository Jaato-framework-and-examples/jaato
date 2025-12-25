# MOD-002 Validation

## Overview

Tier-3 validation for the Retry pattern implementation using Resilience4j.

## Validation Script

```bash
./retry-check.sh /path/to/project
```

## Checks Performed

### Structural Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| @Retry not in domain | ERROR | @Retry must be in application layer |
| @Retry not on controller | WARNING | @Retry should be on application service |
| Annotation order | ERROR | @CircuitBreaker must come before @Retry |

### Configuration Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Retry config exists | ERROR | resilience4j.retry must be in application.yml |
| retryExceptions defined | WARNING | Should explicitly define retry exceptions |
| ignoreExceptions defined | WARNING | Should define business exceptions to ignore |

### Dependency Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Resilience4j present | ERROR | resilience4j-spring-boot3 required |
| Spring AOP present | ERROR | spring-boot-starter-aop required |

### Testing Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Test coverage | WARNING | Test class should exist for services with @Retry |

## Exit Codes

- `0` - Validation passed (may have warnings)
- `1` - Validation failed (has errors)

## Usage in Skills

```bash
# In skill's validate.sh
source "$KNOWLEDGE_BASE/skills/modules/mod-code-002-retry-java-resilience4j/validation/retry-check.sh" "$TARGET_DIR"
```
