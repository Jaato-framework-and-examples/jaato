---
id: mod-code-004-rate-limiter-java-resilience4j
title: "MOD-004: Rate Limiter Pattern - Java/Resilience4j"
version: 1.0
framework: java
library: resilience4j 2.1.0
used_by:
  - skill-020-microservice-java-spring
---

# MOD-004: Rate Limiter Pattern - Java/Resilience4j

## Purpose

Reusable templates for implementing rate limiting patterns using Resilience4j.

## Template 1: Basic Rate Limiter

```java
@RateLimiter(name = "{{rateLimiterName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}
```

## Template 2: Rate Limiter with Fallback

```java
@RateLimiter(name = "{{rateLimiterName}}", fallbackMethod = "{{fallbackMethodName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}

private {{returnType}} {{fallbackMethodName}}({{methodParameters}}, RequestNotPermitted ex) {
    log.warn("Rate limit exceeded for {}", "{{methodName}}");
    {{fallbackLogic}}
}
```

## Template 3: Full Resilience Stack

```java
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{fallbackMethodName}}")
@RateLimiter(name = "{{rateLimiterName}}")
@Retry(name = "{{retryName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}
```

## Exception Handler Template

```java
@RestControllerAdvice
public class RateLimitExceptionHandler {

    @ExceptionHandler(RequestNotPermitted.class)
    public ResponseEntity<ProblemDetail> handleRateLimitExceeded(RequestNotPermitted ex) {
        ProblemDetail problem = ProblemDetail.forStatus(HttpStatus.TOO_MANY_REQUESTS);
        problem.setTitle("Rate Limit Exceeded");
        problem.setDetail(ex.getMessage());
        return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
            .header("Retry-After", "{{retryAfterSeconds}}")
            .body(problem);
    }
}
```

## Configuration Template

```yaml
resilience4j:
  ratelimiter:
    instances:
      {{rateLimiterName}}:
        limit-for-period: {{limitForPeriod}}
        limit-refresh-period: {{limitRefreshPeriod}}s
        timeout-duration: {{timeoutDuration}}ms
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
