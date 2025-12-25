---
id: mod-code-004-rate-limiter-java-resilience4j
title: "MOD-004: Rate Limiter Pattern - Java/Resilience4j"
version: 1.0
date: 2025-11-28
status: Active
derived_from: eri-code-011-rate-limiter-java-resilience4j
domain: code
tags:
  - java
  - resilience4j
  - rate-limiter
  - throttling
  - fault-tolerance
used_by:
  - skill-code-004-add-rate-limiter-java-resilience4j
  - skill-code-020-generate-microservice-java-spring
---

# MOD-004: Rate Limiter Pattern - Java/Resilience4j

## Overview

Reusable template for implementing the Rate Limiter pattern using Resilience4j in Java/Spring Boot applications.

**Source ERI:** [ERI-CODE-011](../../../ERIs/eri-code-011-rate-limiter-java-resilience4j/ERI.md)

---

## Template Catalog

This catalog defines the templates provided by this module and their output paths.

### Annotation Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `annotation/basic-ratelimiter.java.tpl` | Applied to existing service class | Basic @RateLimiter annotation |
| `annotation/ratelimiter-with-fallback.java.tpl` | Applied to existing service class | @RateLimiter with fallback method |
| `annotation/full-resilience-stack.java.tpl` | Applied to existing service class | Combined with Circuit Breaker, Retry, Timeout |

### Configuration Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `config/application-ratelimiter.yml.tpl` | Merge into `src/main/resources/application.yml` | Rate limiter configuration |
| `config/pom-ratelimiter.xml.tpl` | Merge into `pom.xml` | Maven dependencies |

### Test Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `test/RateLimiterTest.java.tpl` | `src/test/java/{{basePackagePath}}/resilience/RateLimiterTest.java` | Integration tests |

---

## Template: Application Service with Rate Limiter

### Basic Rate Limiter

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    @RateLimiter(name = "{rateLimiterName}")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
}
```

### Rate Limiter with Fallback

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.ratelimiter.RequestNotPermitted;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    private final {QueueService} queueService;
    
    @RateLimiter(name = "{rateLimiterName}", fallbackMethod = "{methodName}Fallback")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
    
    private {ReturnType} {methodName}Fallback({ParamType} {paramName}, RequestNotPermitted ex) {
        log.warn("Rate limit exceeded for {}. Queueing.", {paramName});
        queueService.enqueue({paramName});
        return {queuedResponse};
    }
}
```

### Full Resilience Stack (Rate Limiter + Circuit Breaker + Retry)

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    /**
     * Order: RateLimiter (outer) -> CircuitBreaker -> Retry (inner)
     */
    @RateLimiter(name = "{serviceName}")
    @CircuitBreaker(name = "{serviceName}", fallbackMethod = "{methodName}Fallback")
    @Retry(name = "{serviceName}")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
    
    private {ReturnType} {methodName}Fallback({ParamType} {paramName}, Exception ex) {
        log.warn("Service unavailable for {}. Error: {}", {paramName}, ex.getClass().getSimpleName());
        return {defaultValue};
    }
}
```

---

## Template: Configuration

### application.yml

```yaml
resilience4j:
  ratelimiter:
    configs:
      default:
        limitForPeriod: 50
        limitRefreshPeriod: 1s
        timeoutDuration: 0
        registerHealthIndicator: true
    
    instances:
      {rateLimiterName}:
        baseConfig: default
        limitForPeriod: {limitForPeriod}
        limitRefreshPeriod: {limitRefreshPeriod}
        timeoutDuration: {timeoutDuration}

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,ratelimiters,ratelimiterevents
  health:
    ratelimiters:
      enabled: true
```

---

## Template: Exception Handler

```java
// File: {basePackage}/infrastructure/exception/GlobalExceptionHandler.java

package {basePackage}.infrastructure.exception;

import io.github.resilience4j.ratelimiter.RequestNotPermitted;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import jakarta.servlet.http.HttpServletRequest;

@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {
    
    @ExceptionHandler(RequestNotPermitted.class)
    public ResponseEntity<ErrorResponse> handleRateLimitExceeded(
            RequestNotPermitted ex, HttpServletRequest request) {
        
        log.warn("Rate limit exceeded: {}", request.getRequestURI());
        
        ErrorResponse error = new ErrorResponse(
            HttpStatus.TOO_MANY_REQUESTS.value(),
            "Too Many Requests",
            "Rate limit exceeded. Please try again later.",
            request.getRequestURI()
        );
        
        return ResponseEntity
            .status(HttpStatus.TOO_MANY_REQUESTS)
            .header("Retry-After", "1")
            .body(error);
    }
}
```

---

## Template: Dependencies

### pom.xml

```xml
<!-- Resilience4j Spring Boot Starter -->
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>

<!-- Spring Boot AOP (required for annotations) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>

<!-- Actuator for metrics -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

---

## Template: Unit Test

```java
// File: {basePackage}/application/service/{Service}ApplicationServiceTest.java

package {basePackage}.application.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {Service}ApplicationServiceTest {
    
    @Mock
    private {ExternalClient} client;
    
    @InjectMocks
    private {Service}ApplicationService service;
    
    @Test
    void {methodName}_withinRateLimit_succeeds() {
        // Arrange
        {ReturnType} expected = {expectedValue};
        when(client.{clientMethod}({testParam})).thenReturn(expected);
        
        // Act
        {ReturnType} result = service.{methodName}({testParam});
        
        // Assert
        assertEquals(expected, result);
        verify(client).{clientMethod}({testParam});
    }
}
```

---

## Parameter Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{basePackage}` | Base Java package | `com.company.customer` |
| `{Service}` | Service name (PascalCase) | `Customer` |
| `{ExternalClient}` | Client class name | `SystemApiCustomerClient` |
| `{rateLimiterName}` | Rate limiter instance name | `customerService` |
| `{methodName}` | Method name (camelCase) | `getCustomer` |
| `{ReturnType}` | Return type | `Customer` |
| `{ParamType}` | Parameter type | `String` |
| `{paramName}` | Parameter name | `customerId` |
| `{limitForPeriod}` | Requests per period | `50` |
| `{limitRefreshPeriod}` | Period duration | `1s` |
| `{timeoutDuration}` | Wait time for permit | `0` or `500ms` |

---

## Validation

This module includes Tier-3 validation scripts in `validation/`.

See [validation/README.md](validation/README.md) for details.

---

## Related

- **Source ERI:** [ERI-CODE-011](../../../ERIs/eri-code-011-rate-limiter-java-resilience4j/ERI.md)
- **Used with:** mod-code-001-circuit-breaker, mod-code-002-retry, mod-code-003-timeout
- **Skills:** skill-code-004-add-rate-limiter-java-resilience4j
