---
id: mod-code-003-timeout-java-resilience4j
title: "MOD-003: Timeout Pattern - Java/Resilience4j"
version: 1.2
date: 2025-12-22
updated: 2026-01-26
status: Active
derived_from: eri-code-010-timeout-java-resilience4j
domain: code
tags:
  - java
  - resilience4j
  - timeout
  - time-limiter
  - fault-tolerance

# Variant Configuration (v1.2)
#
# CHANGE in v1.2: client-timeout is now a TRANSFORMATION, not generation.
# It MODIFIES RestClientConfig.java from mod-018.
#
variants:
  enabled: true
  selection_mode: auto-suggest

  default:
    id: client-timeout
    name: "Client-level Timeout (Recommended)"
    description: "Configure timeout in HTTP client. MODIFIES RestClientConfig from mod-018."
    type: transformation
    targets:
      - pattern: "**/infrastructure/config/RestClientConfig.java"
        generated_by: mod-code-018-api-integration-rest-java-spring
    transformation_descriptor: client/timeout-config-transform.yaml
    yaml_config: config/application-client-timeout.yml.tpl

  alternatives:
    - id: annotation-async
      name: "Annotation-based Timeout (@TimeLimiter)"
      description: "Use @TimeLimiter with CompletableFuture. Async required."
      type: annotation
      templates:
        - annotation/basic-timeout.java.tpl
        - annotation/timeout-with-fallback.java.tpl
      yaml_config: config/application-timeout.yml.tpl
      recommend_when:
        - condition: "service.async = true"
          reason: "Service already uses async patterns with CompletableFuture"
        - condition: "resilience.timelimiter.fallback.enabled = true"
          reason: "Specific fallback behavior needed on timeout"
      note: "Requires all timed methods to return CompletableFuture<T>"

# ═══════════════════════════════════════════════════════════════════
# MODEL v2.0 - Capability Implementation
# ═══════════════════════════════════════════════════════════════════
implements:
  stack: java-spring
  pattern: mixed
  capability: resilience
  feature: timeout

phase_group: cross-cutting
execution_order: 3  # Runs after circuit-breaker and retry

transformation:
  descriptor: transform/timeout-config-transform.yaml  # Default variant
  client-timeout:
    type: modification
    modifies: RestClientConfig.java
    generated_by: mod-code-018-api-integration-rest-java-spring
  annotation-async:
    type: annotation
    targets: adapter_out
---

# MOD-003: Timeout Pattern - Java/Resilience4j

## Overview

Reusable template for implementing the Timeout pattern in Java/Spring Boot applications.

**Source ERI:** [ERI-CODE-010](../../../ERIs/eri-code-010-timeout-java-resilience4j/ERI.md)

---

## ⚠️ CRITICAL: Two Implementation Variants

This module supports **TWO** timeout strategies. **Read this section carefully before generating code.**

| Variant | ID | When to Use | Requires |
|---------|-----|-------------|----------|
| **DEFAULT** | `client-timeout` | Synchronous services (most cases) | HTTP client config only |
| Alternative | `annotation-async` | Async services with fallback needs | `CompletableFuture<T>` on ALL methods |

### Decision Rule

```
IF service uses synchronous patterns (normal case):
    → Use client-timeout (DEFAULT)

IF service ALREADY uses CompletableFuture everywhere AND needs timeout-specific fallbacks:
    → Use annotation-async
```

**When no variant is explicitly specified, ALWAYS use `client-timeout`.**

---

## Variant 1: Client-level Timeout (DEFAULT - RECOMMENDED)

This is the **default and recommended** approach. Timeout is configured at the HTTP client level.

**Use this variant when:**
- Service uses synchronous patterns (most services)
- You want simple, predictable timeout behavior
- You don't need timeout-specific fallback methods

**Benefits:**
- No code changes to service methods
- Works with existing synchronous code
- Timeout applies uniformly to all HTTP calls
- Simpler to understand and maintain

### Template: RestClientConfig.java

```java
// File: {basePackage}/infrastructure/config/RestClientConfig.java
// @module mod-code-003-timeout-java-resilience4j
// @variant client-timeout

package {basePackage}.infrastructure.config;

import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.util.Timeout;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {

    @Value("${integration.timeout.connect:5s}")
    private java.time.Duration connectTimeout;

    @Value("${integration.timeout.read:10s}")
    private java.time.Duration readTimeout;

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder()
            .requestFactory(clientHttpRequestFactory());
    }

    @Bean
    public HttpComponentsClientHttpRequestFactory clientHttpRequestFactory() {
        HttpComponentsClientHttpRequestFactory factory =
            new HttpComponentsClientHttpRequestFactory(httpClient());
        return factory;
    }

    @Bean
    public CloseableHttpClient httpClient() {
        RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(Timeout.of(connectTimeout))
            .setResponseTimeout(Timeout.of(readTimeout))
            .build();

        return HttpClients.custom()
            .setDefaultRequestConfig(requestConfig)
            .build();
    }
}
```

### Template: application.yml (merge)

```yaml
# Variant: client-timeout
integration:
  timeout:
    connect: 5s       # Time to establish connection
    read: 10s         # Time to read response
```

### What NOT to do with client-timeout

When using client-timeout variant, **DO NOT**:

```java
// ❌ WRONG - @TimeLimiter requires CompletableFuture
@TimeLimiter(name = "backend")
public Optional<Customer> findById(CustomerId id) {  // Sync return type!
    return repository.findById(id);
}
```

The service methods remain **synchronous** with no resilience annotations for timeout:

```java
// ✅ CORRECT - client-timeout variant
// Timeout handled at HTTP client level, no annotations needed
public Optional<Customer> findById(CustomerId id) {
    return repository.findById(id);
}
```

---

## Variant 2: Annotation-based Timeout (ALTERNATIVE - @TimeLimiter)

Use this variant **ONLY** when the service already uses async patterns.

**Use this variant when:**
- Service ALREADY uses `CompletableFuture<T>` for all external calls
- You need timeout-specific fallback behavior
- Explicitly requested via `resilience.timeout.variant = annotation-async`

**⚠️ REQUIREMENT:** Methods with `@TimeLimiter` MUST return `CompletableFuture<T>`.

### Template: Application Service with @TimeLimiter

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
    <version>2.2.0</version>
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
