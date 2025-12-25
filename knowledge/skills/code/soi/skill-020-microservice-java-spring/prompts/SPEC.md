# SPEC.md - LLM Prompt Specification

## Skill: skill-020-generate-microservice-java-spring

**Version:** 2.0  
**Updated:** 2025-12-01  

---

## Overview

This specification defines how an LLM should generate a complete Java/Spring Boot microservice using the Enablement 2.0 Knowledge Base.

**Key Principle:** The LLM does NOT generate code from scratch. It **applies templates** from MODULEs, substituting variables from the generation request.

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. RECEIVE generation-request.json                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 2. ANALYZE features to determine required modules                    │
│    - integration.enabled → mod-code-018                                   │
│    - persistence.type = system_api → mod-code-017                        │
│    - persistence.type = jpa → mod-code-016                               │
│    - resilience.circuitBreaker.enabled → mod-code-001                    │
│    - resilience.retry.enabled → mod-code-002                             │
│    - resilience.timeout.enabled → mod-code-003                           │
│    - resilience.rateLimiter.enabled → mod-code-004                       │
│    - Always include → mod-code-015 (hexagonal base)                      │
├─────────────────────────────────────────────────────────────────────┤
│ 3. LOAD templates from each required module                          │
│    - Read .tpl files from module's templates/ directory             │
│    - Select variant based on config (e.g., restclient vs feign)     │
├─────────────────────────────────────────────────────────────────────┤
│ 4. BUILD variable context from generation-request.json              │
│    - Extract service.name, basePackage, entities, etc.              │
│    - Derive computed variables (EntityId, entityPlural, etc.)       │
├─────────────────────────────────────────────────────────────────────┤
│ 5. APPLY templates using Mustache/Handlebars syntax                  │
│    - Replace {{variable}} with values                               │
│    - Process {{#section}}...{{/section}} blocks                     │
├─────────────────────────────────────────────────────────────────────┤
│ 6. GENERATE mapping code from mapping.json (if persistence=system_api)│
│    - Field transformations (case, format, enum)                     │
│    - Error code mappings                                            │
├─────────────────────────────────────────────────────────────────────┤
│ 7. VALIDATE output against module validation rules                   │
│    - Hexagonal architecture compliance                              │
│    - Resilience annotations present                                 │
│    - Correlation headers propagated                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 8. OUTPUT generated files                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Resolution Rules

### Always Required

| Module | Purpose |
|--------|---------|
| mod-code-015-hexagonal-base-java-spring | Base project structure, domain/application/adapter layers |

### Conditional Modules

| Condition | Module | Templates Used |
|-----------|--------|----------------|
| `persistence.type = "jpa"` | mod-code-016-persistence-jpa-spring | entity/, repository/, mapper/, adapter/ |
| `persistence.type = "system_api"` | mod-code-017-persistence-systemapi | dto/, mapper/, adapter/ |
| `integration.enabled = true` | mod-code-018-api-integration-rest-java-spring | client/, config/, exception/ |
| `resilience.circuitBreaker.enabled` | mod-code-001-circuit-breaker-java-resilience4j | annotation/, config/ |
| `resilience.retry.enabled` | mod-code-002-retry-java-resilience4j | annotation/, config/ |
| `resilience.timeout.enabled` | mod-code-003-timeout-java-resilience4j | annotation/, config/ |
| `resilience.rateLimiter.enabled` | mod-code-004-rate-limiter-java-resilience4j | annotation/, config/ |

### Module Dependencies

```
mod-code-017 (persistence-systemapi) ──depends──▶ mod-code-018 (integration-rest)
```

When mod-code-017 is selected, mod-code-018 MUST also be included.

---

## Template Processing

### Variable Context

Extract from `generation-request.json`:

```yaml
# Service-level
serviceName: customer-domain-api
basePackage: com.bank.customer
basePackagePath: com/bank/customer
groupId: com.bank
artifactId: customer-domain-api

# Entity-level (for each entity)
Entity: Customer
entity: customer
entityPlural: customers
EntityId: CustomerId

# Integration (if enabled)
ApiName: PartiesApi
apiName: partiesApi
resourcePath: /parties
baseUrlEnv: PARTIES_SYSTEM_API_URL

# Resilience
circuitBreakerName: parties-api
retryName: parties-api
timeoutName: parties-api
timeoutDuration: 5s
```

### Mustache/Handlebars Syntax

| Syntax | Purpose | Example |
|--------|---------|---------|
| `{{variable}}` | Simple substitution | `{{Entity}}` → `Customer` |
| `{{#section}}...{{/section}}` | Loop/conditional | `{{#fields}}...{{/fields}}` |
| `{{^section}}...{{/section}}` | Inverted (if not) | `{{^nullable}}NOT NULL{{/nullable}}` |

### Template Variant Selection

For modules with variants (e.g., mod-code-018):

| Config Value | Template Selected |
|--------------|-------------------|
| `integration.apis[].client = "restclient"` | `client/restclient.java.tpl` |
| `integration.apis[].client = "feign"` | `client/feign.java.tpl` |
| `integration.apis[].client = "resttemplate"` | `client/resttemplate.java.tpl` |

Default: `restclient`

---

## Mapping Generation (System API)

When `persistence.type = "system_api"`, read `mapping.json` to generate mapper code.

### Field Mapping

```java
// Generated from mapping.json fieldMappings
public Customer toDomain(PartyDto dto) {
    return Customer.reconstitute(
        CustomerId.of(insertHyphens(dto.getCUST_ID().toLowerCase())),  // UUID format
        capitalize(dto.getCUST_FNAME()),                               // Case transform
        capitalize(dto.getCUST_LNAME()),
        dto.getCUST_EMAIL_ADDR().toLowerCase(),
        LocalDate.parse(dto.getCUST_DOB()),
        mapStatus(dto.getCUST_STATUS()),                               // Enum mapping
        parseTimestamp(dto.getCUST_CRT_TS()),                          // Timestamp format
        parseTimestamp(dto.getCUST_UPD_TS())
    );
}
```

### Enum Mapping

From `mapping.json`:
```json
"enumMapping": {
  "ACTIVE": "A",
  "INACTIVE": "I",
  "BLOCKED": "B",
  "PENDING_VERIFICATION": "P"
}
```

Generate:
```java
private CustomerStatus mapStatus(String code) {
    return switch (code) {
        case "A" -> CustomerStatus.ACTIVE;
        case "I" -> CustomerStatus.INACTIVE;
        case "B" -> CustomerStatus.BLOCKED;
        case "P" -> CustomerStatus.PENDING_VERIFICATION;
        default -> throw new IllegalArgumentException("Unknown status: " + code);
    };
}
```

### Error Mapping

From `mapping.json`:
```json
"errorMappings": [
  { "systemCode": "04", "httpStatus": 404, "domainCode": "CUSTOMER_NOT_FOUND" }
]
```

Generate error handling in adapter.

---

## Architecture Validation

### Hexagonal Light Rules (from ADR-009)

| Layer | Allowed Annotations | Forbidden |
|-------|--------------------| ----------|
| Domain | None (pure POJOs) | @Service, @Autowired, @Component, @Entity, @Repository |
| Application | @Service, @Transactional | @RestController, @Entity |
| Adapter | @RestController, @Entity, @Repository, @Component | - |
| Infrastructure | @Configuration, @Bean, @ControllerAdvice | - |

### Resilience Validation (from ADR-004)

When resilience features enabled, verify:
- `@CircuitBreaker` annotation present on adapter methods
- `@Retry` annotation with correct order (inner to circuit breaker)
- `@TimeLimiter` annotation with CompletableFuture return type
- Configuration in application.yml

### Integration Validation (from ADR-012)

When integration enabled, verify:
- `X-Correlation-ID` header propagated
- `X-Source-System` header set
- Base URL externalized to environment variable

---

## Output Format

### File Output Structure

```
### FILE: {relative/path/to/File.java}

```java
// Generated from: {module}/{template.tpl}
// Variables: Entity={{Entity}}, basePackage={{basePackage}}

package {{basePackage}}.domain.model;

// ... generated code ...
```

### Generation Order

1. **Project Setup:** pom.xml, .gitignore, README.md
2. **Application Entry:** Application.java
3. **Domain Layer:** entities, value objects, repository interfaces, domain services, exceptions
4. **Application Layer:** application services
5. **Adapter Layer - Inbound:** REST controllers, DTOs, mappers
6. **Adapter Layer - Outbound:** persistence adapters, integration clients
7. **Infrastructure:** configuration, exception handlers
8. **Tests:** domain unit tests, adapter integration tests
9. **Resources:** application.yml, application-{profile}.yml
10. **Docker:** Dockerfile, .dockerignore (if enabled)

---

## Error Handling

### If template not found

```
ERROR: Template not found
Module: mod-code-018-api-integration-rest-java-spring
Template: client/restclient.java.tpl
Action: Verify module structure and template existence
```

### If validation fails

```
VALIDATION ERROR: Hexagonal architecture violation
File: CustomerDomainService.java
Issue: Found @Service annotation in domain layer
Fix: Remove Spring annotations from domain layer
```

---

## Multi-LLM Support

| LLM | Prompt File | Notes |
|-----|-------------|-------|
| Claude | prompts/claude.txt | Full context, best for complex generation |
| Gemini | prompts/gemini.txt | Explicit instructions, structured output |
| GPT-4 | prompts/gpt.txt (future) | Additional emphasis on constraints |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial version |
| 2.0 | 2025-12-01 | Added module references, template processing, mapping generation, integration capability |
