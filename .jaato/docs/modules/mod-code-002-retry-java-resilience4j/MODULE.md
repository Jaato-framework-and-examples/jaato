---
id: mod-code-002-retry-java-resilience4j
title: "MOD-002: Retry Pattern - Java/Resilience4j"
version: 1.0
framework: java
library: resilience4j 2.1.0
used_by:
  - skill-020-microservice-java-spring
---

# MOD-002: Retry Pattern - Java/Resilience4j

## Purpose

Reusable templates for implementing retry patterns using Resilience4j.

## Template 1: Basic Retry

```java
@Retry(name = "{{retryName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}
```

## Template 2: Retry with Fallback

```java
@Retry(name = "{{retryName}}", fallbackMethod = "{{fallbackMethodName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}

private {{returnType}} {{fallbackMethodName}}({{methodParameters}}, Throwable throwable) {
    log.warn("All retries exhausted for {}: {}", "{{methodName}}", throwable.getMessage());
    {{fallbackLogic}}
}
```

## Template 3: Retry with Circuit Breaker

```java
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{fallbackMethodName}}")
@Retry(name = "{{retryName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}
```

## Configuration Template

```yaml
resilience4j:
  retry:
    instances:
      {{retryName}}:
        max-attempts: {{maxAttempts}}
        wait-duration: {{waitDuration}}ms
        exponential-backoff-multiplier: {{backoffMultiplier}}
        retry-exceptions:
          - java.net.ConnectException
          - java.net.SocketTimeoutException
          - org.springframework.web.client.ResourceAccessException
        ignore-exceptions:
          - {{businessExceptionClass}}
```

## Dependencies

```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
```
