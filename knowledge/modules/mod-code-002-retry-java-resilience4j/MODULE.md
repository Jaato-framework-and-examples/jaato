---
id: mod-code-002-retry-java-resilience4j
title: "MOD-002: Retry Pattern - Java/Resilience4j"
version: 1.0
date: 2025-11-28
status: Active
derived_from: eri-code-009-retry-java-resilience4j
domain: code
tags:
  - java
  - resilience4j
  - retry
  - fault-tolerance
used_by:
  - skill-code-002-add-retry-java-resilience4j
  - skill-code-020-generate-microservice-java-spring
---

# MOD-002: Retry Pattern - Java/Resilience4j

## Overview

Reusable template for implementing the Retry pattern using Resilience4j in Java/Spring Boot applications.

**Source ERI:** [ERI-CODE-009](../../../ERIs/eri-code-009-retry-java-resilience4j/ERI.md)

---

## Template: Application Service with Retry

### Basic Retry

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    
    @Retry(name = "{retryName}")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
}
```

### Retry with Fallback

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class {Service}ApplicationService {
    
    private final {ExternalClient} client;
    private final {CacheService} cacheService;
    
    @Retry(name = "{retryName}", fallbackMethod = "{methodName}Fallback")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
    
    private {ReturnType} {methodName}Fallback({ParamType} {paramName}, Exception ex) {
        log.warn("All retries exhausted for {}. Error: {}", {paramName}, ex.getMessage());
        return cacheService.getCached({paramName})
            .orElseThrow(() -> new ServiceUnavailableException("Service unavailable", ex));
    }
}
```

### Combined with Circuit Breaker

```java
// File: {basePackage}/application/service/{Service}ApplicationService.java

package {basePackage}.application.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
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
     * Order: CircuitBreaker (outer) -> Retry (inner) -> Actual call
     */
    @CircuitBreaker(name = "{serviceName}", fallbackMethod = "{methodName}Fallback")
    @Retry(name = "{serviceName}")
    public {ReturnType} {methodName}({ParamType} {paramName}) {
        log.debug("Calling external service: {}", {paramName});
        return client.{clientMethod}({paramName});
    }
    
    private {ReturnType} {methodName}Fallback({ParamType} {paramName}, Exception ex) {
        log.warn("Service unavailable for {}. Error: {}", {paramName}, ex.getMessage());
        return {defaultValue};
    }
}
```

---

## Template: Configuration

### application.yml

```yaml
resilience4j:
  retry:
    configs:
      default:
        maxAttempts: 3
        waitDuration: 500ms
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2
        retryExceptions:
          - java.net.ConnectException
          - java.net.SocketTimeoutException
          - org.springframework.web.client.ResourceAccessException
          - java.io.IOException
        ignoreExceptions:
          - {basePackage}.domain.exception.{BusinessException}
    
    instances:
      {retryName}:
        baseConfig: default
        maxAttempts: {maxAttempts}
        waitDuration: {waitDuration}ms

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,retries,retryevents
  health:
    retries:
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

import java.net.ConnectException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {Service}ApplicationServiceTest {
    
    @Mock
    private {ExternalClient} client;
    
    @InjectMocks
    private {Service}ApplicationService service;
    
    @Test
    void {methodName}_successOnFirstAttempt() {
        // Arrange
        {ReturnType} expected = {expectedValue};
        when(client.{clientMethod}({testParam})).thenReturn(expected);
        
        // Act
        {ReturnType} result = service.{methodName}({testParam});
        
        // Assert
        assertEquals(expected, result);
        verify(client, times(1)).{clientMethod}({testParam});
    }
    
    @Test
    void {methodName}_successAfterRetry() {
        // Arrange
        {ReturnType} expected = {expectedValue};
        when(client.{clientMethod}({testParam}))
            .thenThrow(new ConnectException("Connection refused"))
            .thenReturn(expected);
        
        // Act
        {ReturnType} result = service.{methodName}({testParam});
        
        // Assert
        assertEquals(expected, result);
        verify(client, times(2)).{clientMethod}({testParam});
    }
    
    @Test
    void {methodName}_failsAfterMaxRetries() {
        // Arrange
        when(client.{clientMethod}({testParam}))
            .thenThrow(new ConnectException("Connection refused"));
        
        // Act & Assert
        assertThrows(ConnectException.class, 
            () -> service.{methodName}({testParam}));
        verify(client, times(3)).{clientMethod}({testParam});
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
| `{retryName}` | Retry instance name | `customerService` |
| `{methodName}` | Method name (camelCase) | `getCustomer` |
| `{ReturnType}` | Return type | `Customer` |
| `{ParamType}` | Parameter type | `String` |
| `{paramName}` | Parameter name | `customerId` |
| `{maxAttempts}` | Max retry attempts | `3` |
| `{waitDuration}` | Wait between retries (ms) | `500` |

---

## Validation

This module includes Tier-3 validation scripts in `validation/`.

See [validation/README.md](validation/README.md) for details.

---

## Related

- **Source ERI:** [ERI-CODE-009](../../../ERIs/eri-code-009-retry-java-resilience4j/ERI.md)
- **Used with:** mod-code-001-circuit-breaker-java-resilience4j
- **Skills:** skill-code-002-add-retry-java-resilience4j
