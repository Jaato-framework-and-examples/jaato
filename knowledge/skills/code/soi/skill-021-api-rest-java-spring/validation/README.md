# Validation: skill-021-api-rest-java-spring

## Overview

This skill orchestrates validation across three tiers, with layer-specific checks based on the `apiLayer` parameter.

## Validation Tiers

### Tier 1: Universal (Always)

| Validator | Source | Description |
|-----------|--------|-------------|
| traceability-check | tier1/universal | Verifies manifest.json and file headers |
| project-structure-check | tier1/code | Verifies hexagonal directory structure |
| naming-conventions-check | tier1/code | Verifies naming standards |

### Tier 2: Technology-Specific (Always)

| Validator | Source | Description |
|-----------|--------|-------------|
| java-spring-check | tier2/code-projects | Maven build, Spring Boot structure |
| openapi-check | tier2/code-projects | OpenAPI specification validity |

### Tier 3: Module-Specific (Conditional)

| Validator | Module | Condition |
|-----------|--------|-----------|
| pagination-check.sh | mod-019 | Always |
| hateoas-check.sh | mod-019 | apiLayer IN [experience, domain] |
| config-check.sh | mod-019 | Always |
| compensation-interface-check.sh | mod-020 | apiLayer = domain |
| compensation-endpoint-check.sh | mod-020 | apiLayer = domain |
| transaction-log-check.sh | mod-020 | apiLayer = domain |
| circuit-breaker-check.sh | mod-001 | features.resilience.enabled = true |

## Layer-Based Validation Matrix

| Validator | Experience | Composable | Domain | System |
|-----------|------------|------------|--------|--------|
| pagination-check | ✅ | ✅ | ✅ | ✅ |
| hateoas-check | ✅ | ❌ | ✅ | ❌ |
| config-check | ✅ | ✅ | ✅ | ✅ |
| compensation-interface-check | ❌ | ❌ | ✅ | ❌ |
| compensation-endpoint-check | ❌ | ❌ | ✅ | ❌ |
| transaction-log-check | ❌ | ❌ | ✅ | ❌ |

## Usage

```bash
# Run all validations
./validate.sh /path/to/generated/service domain

# Arguments:
#   $1: Path to generated service directory
#   $2: API layer (experience|composable|domain|system)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All validations passed |
| 1-N | Number of failed checks |
