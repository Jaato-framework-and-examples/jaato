---
id: skill-001-circuit-breaker-java-resilience4j
title: "Add Circuit Breaker (Java/Resilience4j)"
complexity: simple
time: 2-3 min
flow: ADD
---

# skill-001-circuit-breaker-java-resilience4j

**Add circuit breaker pattern to existing Java methods calling external services.**

## What It Does

- Adds `@CircuitBreaker` annotation to target method
- Generates fallback method with configurable behavior
- Updates pom.xml with Resilience4j dependency
- Generates circuit breaker configuration in application.yml

## Patterns

| Pattern | Use Case |
|---------|----------|
| basic_fallback | Single recovery method (most common) |
| multiple_fallbacks | Chained fallback options |
| fail_fast | Exception throwing, no recovery |
| programmatic | Non-annotation configuration |

## Prerequisites

- Existing Spring Boot project
- Target class with external service call
- Java 17+

## Related

- **ADR:** adr-004-resilience-patterns
- **ERI:** eri-code-008-circuit-breaker-java-resilience4j
- **Module:** mod-code-001-circuit-breaker-java-resilience4j
