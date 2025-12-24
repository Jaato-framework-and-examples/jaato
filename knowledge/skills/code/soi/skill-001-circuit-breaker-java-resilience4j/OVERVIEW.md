# skill-001-add-circuit-breaker-java-resilience4j

**Type:** TRANSFORMATION  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0  
**Complexity:** Simple  
**Estimated Time:** 2-3 minutes

---

## Purpose

Add circuit breaker pattern to existing Java methods calling external services.

## What it does

- Adds `@CircuitBreaker` annotation
- Generates fallback method (configurable)
- Updates pom.xml with dependencies
- Adds configuration to application.yml

## Patterns Supported

1. **Basic Fallback** - Single fallback (most common)
2. **Multiple Fallbacks** - Fallback chain
3. **Fail Fast** - No fallback, throws exception
4. **Programmatic** - No annotations

## Dependencies

**Implements:** ADR-004 (Resilience Patterns)  
**Uses:** ERI-008 (circuit-breaker-java-resilience4j)  
**Module:** mod-code-001-circuit-breaker-java-resilience4j.md

## Related Skills

- skill-002-add-retry-java
- skill-020-generate-spring-microservice-java
