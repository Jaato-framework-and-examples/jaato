# MOD-004 Validation

## Overview

Tier-3 validation for the Rate Limiter pattern implementation using Resilience4j.

## Validation Script

```bash
./rate-limiter-check.sh /path/to/project
```

## Checks Performed

### Structural Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| @RateLimiter not in domain | ERROR | @RateLimiter must be in application layer |
| @RateLimiter not on controller | WARNING | @RateLimiter should be on application service |
| Annotation order | ERROR | @RateLimiter must come before @Retry |

### Configuration Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| RateLimiter config exists | ERROR | resilience4j.ratelimiter must be in application.yml |
| limitForPeriod defined | ERROR | Must explicitly define rate limit |

### Dependency Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Resilience4j present | ERROR | resilience4j-spring-boot3 required |
| Spring AOP present | ERROR | spring-boot-starter-aop required |

### Exception Handling

| Check | Severity | Description |
|-------|----------|-------------|
| RequestNotPermitted handled | WARNING | Fallback or exception handler should exist |

### Testing Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Test coverage | WARNING | Test class should exist for services with @RateLimiter |

## Exit Codes

- `0` - Validation passed (may have warnings)
- `1` - Validation failed (has errors)

## Usage in Skills

```bash
# In skill's validate.sh
source "$KNOWLEDGE_BASE/skills/modules/mod-code-004-rate-limiter-java-resilience4j/validation/rate-limiter-check.sh" "$TARGET_DIR"
```
