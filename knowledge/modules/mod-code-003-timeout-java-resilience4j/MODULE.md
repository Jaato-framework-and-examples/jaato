---
id: mod-code-003-timeout-java-resilience4j
title: "MOD-003: Timeout Pattern - Java/Resilience4j"
version: 1.1
date: 2025-12-22
status: Active
derived_from: eri-code-010-timeout-java-resilience4j
domain: code
tags:
  - java
  - resilience4j
  - timeout
  - time-limiter
  - fault-tolerance
used_by:
  - skill-code-003-add-timeout-java-resilience4j
  - skill-code-020-generate-microservice-java-spring

# Variant Configuration (v1.1)
variants:
  enabled: true
  selection_mode: auto-suggest
  
  default:
    id: client-timeout
    name: "Client-level Timeout (Recommended)"
    description: "Configure timeout in HTTP client (RestClient/WebClient). Simpler, synchronous."
    templates:
      - client/timeout-config.java.tpl
      - config/application-client-timeout.yml.tpl
    
  alternatives:
    - id: annotation-async
      name: "Annotation-based Timeout (@TimeLimiter)"
      description: "Use @TimeLimiter with CompletableFuture. Async required."
      templates:
        - annotation/basic-timeout.java.tpl
        - annotation/timeout-with-fallback.java.tpl
        - config/application-timeout.yml.tpl
      recommend_when:
        - condition: "service.async = true"
          reason: "Service already uses async patterns with CompletableFuture"
        - condition: "resilience.timelimiter.fallback.enabled = true"
          reason: "Specific fallback behavior needed on timeout"
      note: "Requires all timed methods to return CompletableFuture<T>"
---

# MOD-003: Timeout Pattern - Java/Resilience4j

## Overview

Reusable template for implementing the Timeout pattern (TimeLimiter) using Resilience4j in Java/Spring Boot applications.

**Source ERI:** [ERI-CODE-010](../../../ERIs/eri-code-010-timeout-java-resilience4j/ERI.md)

**Important:** Methods with @TimeLimiter MUST return `CompletableFuture<T>`.

---

## Template: Application Service with Timeout

### Basic Timeout

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    @TimeLimiter(name = "{timelimiterName}")
    public CompletableFuture<{ReturnType}> {methodName}({ParamType} {paramName}) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Calling external service: {}", {paramName});
            return client.{clientMethod}({paramName});
        });
    }
}
```

### Timeout with Fallback

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeoutException;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    @TimeLimiter(name = "{timelimiterName}", fallbackMethod = "{methodName}Fallback")
    public CompletableFuture<{ReturnType}> {methodName}({ParamType} {paramName}) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Calling external service: {}", {paramName});
            return client.{clientMethod}({paramName});
        });
    }
    
    private CompletableFuture<{ReturnType}> {methodName}Fallback({ParamType} {paramName}, TimeoutException ex) {
        log.warn("Timeout for {}. Error: {}", {paramName}, ex.getMessage());
        return CompletableFuture.completedFuture({defaultValue});
    }
}
```

### Full Resilience Stack (Circuit Breaker + Timeout + Retry)

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    /**
     * Order: CircuitBreaker (outer) -> TimeLimiter -> Retry (inner) -> Actual call
     */
    @CircuitBreaker(name = "{serviceName}", fallbackMethod = "{methodName}Fallback")
    @TimeLimiter(name = "{serviceName}")
    @Retry(name = "{serviceName}")
    public CompletableFuture<{ReturnType}> {methodName}({ParamType} {paramName}) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Calling external service: {}", {paramName});
            return client.{clientMethod}({paramName});
        });
    }
    
    private CompletableFuture<{ReturnType}> {methodName}Fallback({ParamType} {paramName}, Exception ex) {
        log.warn("Service unavailable for {}. Error: {}", {paramName}, ex.getClass().getSimpleName());
        return CompletableFuture.completedFuture({defaultValue});
    }
}
```

---

## Template: Configuration

### application.yml

```yaml
resilience4j:
  timelimiter:
    configs:
      default:
        timeoutDuration: 5s
        cancelRunningFuture: true
    
    instances:
      {timelimiterName}:
        baseConfig: default
        timeoutDuration: {timeoutDuration}s

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,timelimiters,timelimiterevents
  health:
    timelimiters:
      enabled: true
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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {Service}ApplicationServiceTest {
    
    @Mock
    private {ExternalClient} client;
    
    @InjectMocks
    private {Service}ApplicationService service;
    
    @Test
    void {methodName}_completesWithinTimeout() throws Exception {
        // Arrange
        {ReturnType} expected = {expectedValue};
        when(client.{clientMethod}({testParam})).thenReturn(expected);
        
        // Act
        CompletableFuture<{ReturnType}> future = service.{methodName}({testParam});
        {ReturnType} result = future.get(1, TimeUnit.SECONDS);
        
        // Assert
        assertEquals(expected, result);
    }
    
    @Test
    void {methodName}_timesOut() {
        // Arrange - simulate slow response
        when(client.{clientMethod}({testParam})).thenAnswer(inv -> {
            Thread.sleep(10000);  // Longer than timeout
            return {expectedValue};
        });
        
        // Act
        CompletableFuture<{ReturnType}> future = service.{methodName}({testParam});
        
        // Assert
        assertThrows(TimeoutException.class, 
            () -> future.get(6, TimeUnit.SECONDS));
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
| `{timelimiterName}` | TimeLimiter instance name | `customerService` |
| `{methodName}` | Method name (camelCase) | `getCustomer` |
| `{ReturnType}` | Return type (without Future) | `Customer` |
| `{ParamType}` | Parameter type | `String` |
| `{paramName}` | Parameter name | `customerId` |
| `{timeoutDuration}` | Timeout in seconds | `5` |

---

## Validation

This module includes Tier-3 validation scripts in `validation/`.

See [validation/README.md](validation/README.md) for details.

---

## Related

- **Source ERI:** [ERI-CODE-010](../../../ERIs/eri-code-010-timeout-java-resilience4j/ERI.md)
- **Used with:** mod-code-001-circuit-breaker, mod-code-002-retry
- **Skills:** skill-code-003-add-timeout-java-resilience4j
