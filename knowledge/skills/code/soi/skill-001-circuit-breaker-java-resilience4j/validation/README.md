# Validation for skill-code-001-add-circuit-breaker-java-resilience4j

## Overview

This validation orchestrator ensures that the Circuit Breaker pattern is correctly implemented across four tiers:

1. **Tier 1 - Infrastructure**: Project structure and naming conventions (ALWAYS)
2. **Tier 2 - Stack**: Maven compilation, Spring Boot configuration (CONDITIONAL)
3. **Tier 3 - Module**: Circuit Breaker feature validation (CONDITIONAL)
4. **Tier 4 - Runtime**: CI/CD tests (FUTURE - not implemented locally)

## Usage

```bash
./validate.sh <service-dir> [base-package-path]
```

**Example:**
```bash
./validate.sh ./customer-service com/company/customer
```

## What Gets Validated

### Tier 1: Infrastructure (ALWAYS)
- ✅ Project structure (src/main/java, src/test/java)
- ✅ Naming conventions (PascalCase classes, lowercase packages)

### Tier 2: Stack (CONDITIONAL)
- ✅ Maven compilation succeeds (if pom.xml exists)
- ✅ Maven tests pass
- ✅ Spring Boot Actuator configured (if Spring Boot detected)
- ✅ application.yml valid
- ✅ Dockerfile valid (if present)

### Tier 3: Module - Circuit Breaker (ALWAYS for this skill)
- ✅ @CircuitBreaker annotation present
- ✅ Fallback methods defined and implemented
- ✅ Resilience4j configuration in application.yml
- ✅ Dependencies in pom.xml
- ✅ Actuator endpoints exposed
- ✅ Annotation in application layer (not domain)

### Tier 4: Runtime (FUTURE)
- ⏳ Integration tests in CI/CD
- ⏳ Contract tests
- ⏳ E2E tests

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

## Related

- **Module:** mod-code-001-circuit-breaker-java-resilience4j
- **Validation Scripts:**
  - Tier 1 Generic: `/knowledge/validators/tier-1-universal/`
  - Tier 2 Artifacts: `/knowledge/validators/tier-2-technology/code-projects/java-spring/`
  - Tier 3 Module: `/knowledge/skills/modules/mod-code-001-.../validation/`
