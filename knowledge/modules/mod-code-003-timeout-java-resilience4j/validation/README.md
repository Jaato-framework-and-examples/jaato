# MOD-003 Validation

## Overview

Tier-3 validation for the Timeout pattern (TimeLimiter) implementation using Resilience4j.

## Validation Script

```bash
./timeout-check.sh /path/to/project
```

## Checks Performed

### Structural Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| @TimeLimiter not in domain | ERROR | @TimeLimiter must be in application layer |
| @TimeLimiter not on controller | WARNING | @TimeLimiter should be on application service |
| Returns CompletableFuture | ERROR | Methods MUST return CompletableFuture |
| Annotation order | ERROR | @CircuitBreaker > @TimeLimiter > @Retry |

### Configuration Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| TimeLimiter config exists | ERROR | resilience4j.timelimiter must be in application.yml |
| timeoutDuration defined | WARNING | Should explicitly define timeout duration |

### Dependency Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Resilience4j present | ERROR | resilience4j-spring-boot3 required |
| Spring AOP present | ERROR | spring-boot-starter-aop required |

### Testing Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Test coverage | WARNING | Test class should exist for services with @TimeLimiter |

## Exit Codes

- `0` - Validation passed (may have warnings)
- `1` - Validation failed (has errors)

## Usage in Skills

```bash
# In skill's validate.sh
source "$KNOWLEDGE_BASE/skills/modules/mod-code-003-timeout-java-resilience4j/validation/timeout-check.sh" "$TARGET_DIR"
```
