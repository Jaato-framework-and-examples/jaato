# MOD-017 Validation

## Overview

Tier-3 validation for System API persistence implementation.

## Validation Script

```bash
./systemapi-check.sh /path/to/project
```

## Checks Performed

### Structural Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Repository in domain | ERROR | Interface must be in domain/repository/ |
| Adapter location | ERROR | Must be in adapter/systemapi/ |
| Client location | WARNING | Should be in adapter/systemapi/client/ |

### Resilience Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| @CircuitBreaker | ERROR | Must be present on adapter methods |
| @Retry | ERROR | Must be present on adapter methods |
| Resilience4j config | ERROR | Configuration required in application.yml |

### Configuration Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Base URL externalized | WARNING | Should use ${...} placeholder |
| Timeouts configured | WARNING | Connection/read timeouts should be set |

### Headers Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Correlation headers | WARNING | X-Correlation-Id, X-Source-System should be set |

### Dependency Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Resilience4j | ERROR | resilience4j dependency required |

### Testing Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Adapter tests | WARNING | SystemApiAdapter tests should exist |

## Exit Codes

- `0` - Validation passed
- `1` - Validation failed
