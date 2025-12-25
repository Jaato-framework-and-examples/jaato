# MOD-016 Validation

## Overview

Tier-3 validation for JPA persistence implementation following Hexagonal Architecture.

## Validation Script

```bash
./jpa-check.sh /path/to/project
```

## Checks Performed

### Structural Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| JPA not in domain | ERROR | @Entity, @Table, @Id must NOT be in domain/ |
| JPA entities location | WARNING | Should be in adapter/persistence/entity/ |
| Repository in domain | ERROR | Interface must be in domain/repository/ |

### Configuration Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| ddl-auto validate | WARNING | Should be 'validate' in production |
| OSIV disabled | WARNING | open-in-view should be false |

### Dependency Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Spring Data JPA | ERROR | spring-boot-starter-data-jpa required |
| Database driver | WARNING | PostgreSQL/MySQL/H2 driver needed |

### Testing Constraints

| Check | Severity | Description |
|-------|----------|-------------|
| Adapter tests | WARNING | PersistenceAdapter tests should exist |

## Exit Codes

- `0` - Validation passed
- `1` - Validation failed
