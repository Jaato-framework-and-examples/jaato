---
skill: skill-020-microservice-java-spring
prompt_type: consumer
version: 1.0
---

# SPEC: Generate Microservice (Java/Spring)

## Overview

This specification defines how to generate a complete Java/Spring Boot microservice using the GENERATE flow. The LLM applies templates from MODULEs rather than generating code from scratch.

## Execution Steps

### Step 1: Receive Generation Request

Parse the generation request JSON containing:
- `serviceName`: kebab-case identifier
- `entityName`: PascalCase domain entity name
- `groupId`: Maven group ID
- `basePackage`: Java package root
- `persistence`: "jpa" | "systemapi"
- `features`: Optional feature flags

### Step 2: Analyze Required Modules

Based on the request, determine which modules are needed:

| Condition | Required Module |
|-----------|-----------------|
| Always | mod-code-015-hexagonal-base-java-spring |
| persistence="jpa" | mod-code-016-persistence-jpa-spring |
| persistence="systemapi" | mod-code-017-persistence-systemapi |
| persistence="systemapi" | mod-code-018-api-integration-rest-java-spring |
| features.resilience.circuit_breaker | mod-code-001-circuit-breaker-java-resilience4j |
| features.resilience.retry | mod-code-002-retry-java-resilience4j |
| features.resilience.timeout | mod-code-003-timeout-java-resilience4j |
| features.resilience.rate_limiter | mod-code-004-rate-limiter-java-resilience4j |

### Step 3: Load Template Files

Read MODULE.md from each required module and extract code templates.

### Step 4: Build Variable Context

Map request data to template variables:

```json
{
  "serviceName": "order-service",
  "serviceNamePascal": "OrderService",
  "entityName": "Order",
  "entityNameLower": "order",
  "entityNamePlural": "orders",
  "groupId": "com.example",
  "basePackage": "com.example.orderservice",
  "basePackagePath": "com/example/orderservice"
}
```

### Step 5: Apply Template Substitution

Replace all `{{variable}}` placeholders with context values.

For conditional sections:
- `{{#features.resilience.circuit_breaker}}...{{/features.resilience.circuit_breaker}}`

### Step 6: Generate Mapping Code

For systemapi persistence, generate field transformation code based on mapping.json.

### Step 7: Validate Architecture

Verify output compliance:
- [ ] Domain layer has NO Spring/JPA annotations
- [ ] Application layer uses only @Service, @Transactional
- [ ] Adapter/REST uses @RestController, @RequestMapping
- [ ] Adapter/Persistence uses @Entity, @Repository
- [ ] Dependencies flow inward only

### Step 8: Output Generated Files

Generate files in order:
1. pom.xml
2. Application.java
3. Domain layer (entity, value objects, service, repository port)
4. Application layer (application service, DTOs)
5. Adapter layer (REST controller, persistence adapter)
6. Infrastructure (config, exception handler)
7. application.yml
8. Test files

## Critical Rules

1. **Template Fidelity**: Use EXACT templates from modules - no modifications
2. **Domain Purity**: Zero framework annotations in domain layer
3. **Inward Dependencies**: Adapters → Application → Domain (never reverse)
4. **No Remaining Variables**: Validate no `{{variables}}` remain in output
5. **Feature Flags**: Only include resilience patterns when explicitly enabled

## Example Request

```json
{
  "serviceName": "order-service",
  "entityName": "Order",
  "groupId": "com.example",
  "basePackage": "com.example.orderservice",
  "persistence": "jpa",
  "features": {
    "resilience": {
      "circuit_breaker": true,
      "retry": true
    }
  }
}
```

## Output Validation

Before completing, verify:
1. All template variables substituted
2. No syntax errors in generated code
3. Hexagonal architecture rules enforced
4. Required dependencies included in pom.xml
5. Configuration properties complete
