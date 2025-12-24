# skill-code-001-add-circuit-breaker-java-resilience4j

**Version:** 2.1  
**Type:** TRANSFORMATION  
**Domain:** CODE  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0  
**Status:** Active

---

## Overview

Transforms Java methods to add Resilience4j circuit breaker pattern.

**Implements:** ADR-004 (Resilience Patterns)  
**Uses:** ERI-008 (circuit-breaker-java-resilience4j)  
**Module:** mod-code-001-circuit-breaker-java-resilience4j

---

## Input Schema

```json
{
  "targetClass": "string (required)",
  "targetMethod": "string (required)",
  "circuitBreakerName": "string (optional)",
  "pattern": {
    "type": "basic_fallback | multiple_fallbacks | fail_fast | programmatic",
    "fallback_strategy": "empty_object | failed_object | cached | default_value"
  },
  "config": {
    "failureRateThreshold": 50,
    "waitDurationInOpenState": 60000,
    "slidingWindowSize": 100,
    "minimumNumberOfCalls": 10
  }
}
```

### Example Inputs

**Basic fallback (default):**
```json
{
  "targetClass": "com.example.payment.PaymentService",
  "targetMethod": "processPayment"
}
```

**Fail fast:**
```json
{
  "targetClass": "com.example.analytics.AnalyticsService",
  "targetMethod": "sendEvent",
  "pattern": {"type": "fail_fast"}
}
```

**Multiple fallbacks:**
```json
{
  "targetClass": "com.example.inventory.InventoryService",
  "targetMethod": "checkStock",
  "pattern": {"type": "multiple_fallbacks"}
}
```

---

## Output Schema

```json
{
  "status": "SUCCESS | FAILURE",
  "modifiedFiles": [
    {"path": "...", "type": "JAVA | POM | YAML", "changes": "..."}
  ],
  "output": {
    "modifiedClass": "...",
    "fallbackMethodAdded": "...",
    "pomUpdated": true,
    "configAdded": {...}
  }
}
```

---

## Success Criteria

### Required:
- ✅ `@CircuitBreaker` annotation added
- ✅ Import statements added
- ✅ Configuration in application.yml
- ✅ Dependencies in pom.xml
- ✅ Code compiles

### Pattern-specific:
- **basic_fallback:** Fallback method with Throwable parameter
- **multiple_fallbacks:** Fallback chain generated
- **fail_fast:** No fallback, throws clause added
- **programmatic:** CircuitBreakerRegistry injected

---

## Execution Flow

This skill follows the **ADD** execution flow defined at domain level.

**See:** [`runtime/flows/code/ADD.md`](../../runtime/flows/code/ADD.md)

The ADD flow handles:
1. Input validation
2. Existing code analysis
3. Module resolution
4. Target location identification
5. Variable context building
6. Modification generation
7. Code application
8. Manifest update
9. Validation
10. Output generation

---

## Validation

1. **Static checks:** Syntax validation
2. **Compliance checks:** ADR-004 rules
3. **Runtime tests:** Functional testing

See `validation/` directory.

---

## Dependencies

```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>
```

---

## References

- **ADR-004:** Resilience Patterns
- **ERI-008:** circuit-breaker-java-resilience4j

---

## Modules Used

| Module | Purpose |
|--------|---------|
| `mod-code-001-circuit-breaker-java-resilience4j` | Templates and validation for circuit breaker pattern |

See module's **Template Catalog** for available templates:
- `annotation/basic-circuitbreaker.java.tpl` - Basic @CircuitBreaker
- `annotation/circuitbreaker-with-fallback.java.tpl` - With fallback method
- `config/application-circuitbreaker.yml.tpl` - YAML configuration

---

**Last Updated:** 2025-12-16
