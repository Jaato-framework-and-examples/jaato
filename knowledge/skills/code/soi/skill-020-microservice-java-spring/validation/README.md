# Validation for skill-code-020-generate-microservice-java-spring

## Overview

This validation orchestrator ensures that generated microservices are correctly implemented. As a **CODE domain skill**, it orchestrates external validators:

1. **Tier 1 Universal**: Traceability (.enablement/manifest.json) - ALWAYS
2. **Tier 1 Code**: Project structure and naming conventions - ALWAYS (CODE domain)
3. **Tier 2**: Maven, Spring Boot, Docker configurations - CONDITIONAL
4. **Tier 3**: Hexagonal architecture (ADR-009), optional Circuit Breaker - CONDITIONAL
5. **Tier 4**: CI/CD tests - FUTURE (not implemented locally)

## Usage

```bash
./validate.sh <service-dir> <base-package-path> [input-json]
```

**Examples:**

Basic validation:
```bash
./validate.sh ./customer-service com/company/customer
```

With feature detection (Circuit Breaker):
```bash
./validate.sh ./customer-service com/company/customer .enablement/inputs/skill-code-020-input.json
```

## What Gets Validated

### Tier 1 Universal: Traceability (ALWAYS - All Domains)
- ✅ `.enablement/` directory exists
- ✅ `manifest.json` is valid JSON
- ✅ Required traceability fields present

### Tier 1 Code: Infrastructure (ALWAYS - CODE Domain)
- ✅ Project structure (src/main/java, src/test/java)
- ✅ Naming conventions (PascalCase classes, lowercase packages)

### Tier 2: Stack (CONDITIONAL)
- ✅ Maven compilation succeeds (if pom.xml exists)
- ✅ Maven tests pass
- ✅ Spring Boot Actuator configured (if Spring Boot detected)
- ✅ application.yml valid and well-formed
- ✅ Dockerfile syntax and best practices (if present)

### Tier 3: Module - Hexagonal (ALWAYS for this skill)

**Hexagonal Architecture (mod-code-015):**
- ✅ Domain layer exists
- ✅ Application layer exists
- ✅ Adapter layer exists
- ✅ **Domain is Spring-free** (ADR-009)
- ✅ **Domain model is JPA-free** (ADR-009)
- ✅ Repository is interface in domain
- ✅ Repository adapter implements interface
- ✅ JPA entities in adapter layer
- ✅ Domain tests don't use Spring context

### Tier 3: Module - Circuit Breaker (CONDITIONAL)

**Circuit Breaker (mod-code-001)** - if `features.circuit_breaker: true` in input JSON:
- ✅ @CircuitBreaker annotation present
- ✅ Fallback methods defined and implemented
- ✅ Resilience4j configuration
- ✅ Dependencies in pom.xml
- ✅ Annotation in application layer (not domain)

### Tier 4: Runtime (FUTURE)
- ⏳ Integration tests in CI/CD
- ⏳ Contract tests
- ⏳ E2E tests

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

## Feature Detection

When `input-json` is provided, the script detects enabled features:

```json
{
  "features": {
    "circuit_breaker": true,
    "saga_pattern": false
  }
}
```

Only validations for **enabled features** will run.

## Validation Order

The script **always** executes validations in this order:

1. **Infrastructure** (always)
2. **Stack** (conditional on technology)
3. **Module** (always hexagonal, conditional for features)
4. **Runtime** (future - in CI/CD only)

This ensures:
- Basic structure is valid before checking technology
- Technology setup is valid before checking patterns
- Patterns are validated last

## Related

- **Modules:**
  - mod-code-015-hexagonal-base-java-spring (always)
  - mod-code-001-circuit-breaker-java-resilience4j (conditional)
- **Validation Scripts:**
  - Tier 1 Universal: `/knowledge/validators/tier-1-universal/traceability/`
  - Tier 1 Code: `/knowledge/validators/tier-1-universal/code-projects/`
  - Tier 2 Artifacts: `/knowledge/validators/tier-2-technology/code-projects/java-spring/`
  - Tier 3 Modules: `/knowledge/skills/modules/mod-*/validation/`

## ADR Compliance

This validation ensures compliance with:
- **ADR-009**: Service Architecture Patterns (Hexagonal Light)
- **ADR-004**: Resilience Patterns (Circuit Breaker, if enabled)

## Example Output

```
========================================
  SKILL-CODE-020 VALIDATION
  Generate Microservice - Java/Spring
  Service: ./customer-service
  Package: com/company/customer
========================================

=== TIER 1: Infrastructure Validations ===

✅ PASS: src/main/java exists
✅ PASS: src/test/java exists
✅ PASS: Java classes follow PascalCase convention


=== TIER 2: Stack Validations ===

--- Java Maven Stack ---
✅ PASS: Maven compilation successful
✅ PASS: All tests passed (4 test file(s))

--- Spring Boot Stack ---
✅ PASS: Spring Boot Actuator configured
✅ PASS: application.yml has valid YAML syntax


=== TIER 3: Module Validations ===

--- Hexagonal Architecture Module ---
✅ PASS: Domain layer exists
✅ PASS: Application layer exists
✅ PASS: Adapter layer exists
✅ PASS: Domain layer is Spring-free (ADR-009 compliant)
✅ PASS: Domain model is JPA-free (ADR-009 compliant)
✅ PASS: Repository is interface in domain layer
✅ PASS: Repository adapter implements domain interface

--- Circuit Breaker Module (Feature Enabled) ---
✅ PASS: @CircuitBreaker annotation found
✅ PASS: Fallback methods implemented
✅ PASS: Resilience4j configuration present


========================================
  VALIDATION SUMMARY
========================================

✅ ALL VALIDATIONS PASSED

Validated:
  - Infrastructure: Project structure, naming conventions
  - Stack: Maven, Spring Boot, Docker (if applicable)
  - Module: Hexagonal architecture (ADR-009)
  - Module: Circuit Breaker (Resilience4j)
```
