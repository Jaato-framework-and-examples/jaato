---
id: mod-code-003-timeout-java-resilience4j
title: "MOD-003: Timeout Pattern - Java/Resilience4j"
version: 1.0
framework: java
library: resilience4j 2.1.0
used_by:
  - skill-020-microservice-java-spring
---

# MOD-003: Timeout Pattern - Java/Resilience4j

## Purpose

Reusable templates for implementing timeout patterns using Resilience4j TimeLimiter.

**IMPORTANT**: Methods with @TimeLimiter MUST return `CompletableFuture<T>`.

## Template 1: Basic Timeout

```java
@TimeLimiter(name = "{{timeLimiterName}}")
public CompletableFuture<{{returnType}}> {{methodName}}({{methodParameters}}) {
    return CompletableFuture.supplyAsync(() -> {
        {{originalMethodBody}}
    });
}
```

## Template 2: Timeout with Fallback

```java
@TimeLimiter(name = "{{timeLimiterName}}", fallbackMethod = "{{fallbackMethodName}}")
public CompletableFuture<{{returnType}}> {{methodName}}({{methodParameters}}) {
    return CompletableFuture.supplyAsync(() -> {
        {{originalMethodBody}}
    });
}

private CompletableFuture<{{returnType}}> {{fallbackMethodName}}({{methodParameters}}, TimeoutException ex) {
    log.warn("Timeout for {}: {}", "{{methodName}}", ex.getMessage());
    return CompletableFuture.completedFuture({{fallbackValue}});
}
```

## Template 3: Full Resilience Stack

```java
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{fallbackMethodName}}")
@TimeLimiter(name = "{{timeLimiterName}}")
@Retry(name = "{{retryName}}")
public CompletableFuture<{{returnType}}> {{methodName}}({{methodParameters}}) {
    return CompletableFuture.supplyAsync(() -> {
        {{originalMethodBody}}
    });
}
```

## Configuration Template

```yaml
resilience4j:
  timelimiter:
    instances:
      {{timeLimiterName}}:
        timeout-duration: {{timeoutDuration}}s
        cancel-running-future: true
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
