---
id: mod-code-001-circuit-breaker-java-resilience4j
title: "MOD-001: Circuit Breaker - Java/Resilience4j"
version: 2.1
framework: java
library: resilience4j 2.1.0
used_by:
  - skill-001-circuit-breaker-java-resilience4j
  - skill-020-microservice-java-spring
---

# MOD-001: Circuit Breaker - Java/Resilience4j

## Purpose

Reusable templates for circuit breaker pattern using Resilience4j.

## Template 1: Basic with Fallback

```java
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{fallbackMethodName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}

private {{returnType}} {{fallbackMethodName}}({{methodParameters}}, Throwable throwable) {
    log.warn("Fallback for {}: {}", "{{methodName}}", throwable.getMessage());
    {{fallbackLogic}}
}
```

## Configuration Template

```yaml
resilience4j:
  circuitbreaker:
    instances:
      {{circuitBreakerName}}:
        failure-rate-threshold: {{failureRateThreshold}}
        wait-duration-in-open-state: {{waitDurationInOpenState}}s
        sliding-window-size: {{slidingWindowSize}}
```

## Dependencies

```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>
```
