---
id: eri-code-010-timeout-java-resilience4j
title: "ERI-CODE-010: Timeout Pattern - Java/Spring Boot with Resilience4j"
sidebar_label: "Timeout (Java)"
version: 1.1
date: 2025-11-28
updated: 2025-12-22
status: Active
author: "Architecture Team"
domain: code
pattern: timeout
framework: java
library: resilience4j
library_version: 2.1.0
implements:
  - adr-004-resilience-patterns
implements_pattern: timeout
tags:
  - java
  - spring-boot
  - resilience4j
  - timeout
  - time-limiter
  - fault-tolerance
  - microservices
related:
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-code-009-retry-java-resilience4j
  - eri-code-011-rate-limiter-java-resilience4j
automated_by:
  - skill-code-003-add-timeout-java-resilience4j
cross_domain_usage: qa
---

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Timeout pattern (TimeLimiter) in Java/Spring Boot microservices using Resilience4j.

**What this implements:**
- Time-bounded execution of external calls
- Configurable timeout durations
- Timeout cancellation strategies
- Monitoring and metrics integration

**When to use:**
- External API calls with unpredictable latency
- Database queries that might hang
- System API calls to legacy systems
- Any operation where "no response" is worse than "failure"

**When NOT to use:**
- Local computations (use async processing instead)
- Operations that cannot be safely interrupted
- Fire-and-forget operations

---

## Implementation Options

> **NEW in v1.1:** This ERI defines two valid implementation approaches. Modules derived from this ERI MUST only implement these options.

### Recommended Default: Client-level Timeout

**Why Default:** Simpler, works with synchronous code, no CompletableFuture requirement.

### Options Summary

| Option | Status | Recommended When | Module |
|--------|--------|------------------|--------|
| Client-level Timeout | ⭐ DEFAULT | New projects, synchronous code | mod-code-003 (client-timeout variant) |
| @TimeLimiter Annotation | Alternative | Async code, specific fallback needs | mod-code-003 (annotation-async variant) |

### Option A: Client-level Timeout ⭐ DEFAULT

**Description:** Configure timeout at the HTTP client level (RestClient, WebClient, RestTemplate).

**Recommended When:**
- Synchronous code patterns
- New Spring Boot 3.2+ projects
- Simple timeout requirements without specific fallback logic

**Trade-offs:**
- ✅ Simpler - no CompletableFuture required
- ✅ Works with standard synchronous code
- ✅ Centralized configuration
- ⚠️ Less granular control per method

**Reference Implementation:**

```java
@Configuration
public class RestClientConfig {
    
    @Value("${integration.timeout.connect:5s}")
    private Duration connectTimeout;
    
    @Value("${integration.timeout.read:10s}")
    private Duration readTimeout;
    
    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder()
            .requestFactory(clientHttpRequestFactory());
    }
    
    @Bean
    public HttpComponentsClientHttpRequestFactory clientHttpRequestFactory() {
        RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(Timeout.of(connectTimeout))
            .setResponseTimeout(Timeout.of(readTimeout))
            .build();
        
        CloseableHttpClient httpClient = HttpClients.custom()
            .setDefaultRequestConfig(requestConfig)
            .build();
        
        return new HttpComponentsClientHttpRequestFactory(httpClient);
    }
}
```

### Option B: @TimeLimiter Annotation

**Description:** Use Resilience4j @TimeLimiter annotation with CompletableFuture.

**Recommended When:**
- Service already uses async patterns with CompletableFuture
- Specific fallback behavior needed on timeout
- Fine-grained per-method timeout configuration required

**Trade-offs:**
- ✅ Per-method configuration
- ✅ Built-in fallback support
- ✅ Metrics and monitoring integration
- ⚠️ Requires CompletableFuture<T> return type
- ⚠️ More complex code structure

**Reference Implementation:**

```java
@Service
@RequiredArgsConstructor
public class PaymentApplicationService {
    
    private final PaymentClient paymentClient;
    
    @TimeLimiter(name = "paymentService", fallbackMethod = "processPaymentFallback")
    public CompletableFuture<PaymentResult> processPayment(PaymentRequest request) {
        return CompletableFuture.supplyAsync(() -> 
            paymentClient.process(request)
        );
    }
    
    private CompletableFuture<PaymentResult> processPaymentFallback(
            PaymentRequest request, TimeoutException ex) {
        log.warn("Payment timeout: {}", ex.getMessage());
        return CompletableFuture.completedFuture(PaymentResult.pending());
    }
}
```

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Spring Boot** | 3.2.x | Application framework |
| **Resilience4j** | 2.1.0 | TimeLimiter library |
| **Spring AOP** | (included) | Annotation processing |
| **Micrometer** | (included) | Metrics collection |

---

## Dependencies

### Maven (pom.xml)

```xml
<dependencies>
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
    
    <!-- Actuator for metrics (optional but recommended) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
</dependencies>
```

---

## Project Structure

```
src/main/java/com/company/{service}/
├── application/
│   └── service/
│       └── {Service}ApplicationService.java    # @TimeLimiter annotations here
├── adapter/
│   └── client/
│       └── {External}Client.java               # External service client
└── infrastructure/
    └── config/
        └── ResilienceConfig.java               # Programmatic configuration (optional)

src/main/resources/
└── application.yml                             # Timeout configuration

src/test/java/com/company/{service}/
└── application/
    └── service/
        └── {Service}ApplicationServiceTest.java
```

**Key Placement Rules:**
- `@TimeLimiter` annotations go on **Application Service** methods (not domain)
- Methods MUST return `CompletableFuture<T>` for @TimeLimiter to work
- Configuration is externalized in `application.yml`
- Combine with `@CircuitBreaker` and `@Retry` for comprehensive resilience

---

## Configuration

### application.yml

```yaml
resilience4j:
  timelimiter:
    configs:
      # Default configuration for all time limiters
      default:
        timeoutDuration: 5s
        cancelRunningFuture: true
    
    instances:
      # Specific instance for payment service
      paymentService:
        baseConfig: default
        timeoutDuration: 10s
      
      # Specific instance for system API calls (faster timeout)
      systemApiClient:
        baseConfig: default
        timeoutDuration: 3s
      
      # Specific instance for batch operations (longer timeout)
      batchService:
        baseConfig: default
        timeoutDuration: 30s
        cancelRunningFuture: false

# Expose timeout metrics
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,timelimiters,timelimiterevents
  endpoint:
    health:
      show-details: always
```

### Configuration Parameters Explained

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| **timeoutDuration** | 1s | Maximum time to wait | 3-10s for APIs, 30s+ for batch |
| **cancelRunningFuture** | true | Cancel future on timeout | true unless operation must complete |

---

## Implementation Patterns

### Pattern 1: Basic Timeout with CompletableFuture

```java
package com.company.customer.application.service;

import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
@Slf4j
public class CustomerApplicationService {
    
    private final SystemApiCustomerClient systemApiClient;
    
    /**
     * Fetch customer with timeout protection.
     * MUST return CompletableFuture for @TimeLimiter to work.
     */
    @TimeLimiter(name = "systemApiClient")
    public CompletableFuture<Customer> getCustomer(String customerId) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Fetching customer: {}", customerId);
            return systemApiClient.findById(customerId);
        });
    }
}
```

**Key points:**
- ✅ Returns `CompletableFuture<T>` - REQUIRED for @TimeLimiter
- ✅ Uses `supplyAsync` to wrap blocking call
- ✅ Timeout name matches config

---

### Pattern 2: Timeout with Fallback

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class PaymentApplicationService {
    
    private final PaymentSystemApiClient systemApiClient;
    private final PaymentCacheService cacheService;
    
    @TimeLimiter(name = "paymentService", fallbackMethod = "getPaymentStatusFallback")
    public CompletableFuture<PaymentStatus> getPaymentStatus(String transactionId) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Checking payment status: {}", transactionId);
            return systemApiClient.getStatus(transactionId);
        });
    }
    
    /**
     * Fallback when timeout occurs.
     * Signature: same params + Exception at end, returns CompletableFuture.
     */
    private CompletableFuture<PaymentStatus> getPaymentStatusFallback(
            String transactionId, TimeoutException ex) {
        log.warn("Timeout getting payment status: {}. Error: {}", 
                 transactionId, ex.getMessage());
        
        return CompletableFuture.completedFuture(
            cacheService.getCachedStatus(transactionId)
                .orElse(PaymentStatus.UNKNOWN)
        );
    }
}
```

**Key points:**
- ✅ Fallback also returns `CompletableFuture`
- ✅ Uses `CompletableFuture.completedFuture()` for sync fallback
- ✅ Fallback receives `TimeoutException`

---

### Pattern 3: Combined Timeout + Circuit Breaker + Retry

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class InventoryApplicationService {
    
    private final InventorySystemApiClient systemApiClient;
    
    /**
     * Full resilience stack for System API calls.
     * 
     * Order (outer to inner):
     * 1. CircuitBreaker - Prevents calls when service is down
     * 2. TimeLimiter - Prevents hanging calls
     * 3. Retry - Retries transient failures
     * 4. Actual call
     */
    @CircuitBreaker(name = "inventoryService", fallbackMethod = "checkStockFallback")
    @TimeLimiter(name = "inventoryService")
    @Retry(name = "inventoryService")
    public CompletableFuture<StockLevel> checkStock(String productId) {
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Checking stock for product: {}", productId);
            return systemApiClient.getStockLevel(productId);
        });
    }
    
    private CompletableFuture<StockLevel> checkStockFallback(String productId, Exception ex) {
        log.warn("Inventory service unavailable for product: {}. Error: {}", 
                 productId, ex.getClass().getSimpleName());
        return CompletableFuture.completedFuture(StockLevel.UNKNOWN);
    }
}
```

**Annotation Order:**
```
@CircuitBreaker (outer)
  └── @TimeLimiter
        └── @Retry (inner)
              └── Actual method
```

---

### Pattern 4: Using with WebClient (Reactive)

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class CustomerApplicationService {
    
    private final WebClient webClient;
    private final TimeLimiterRegistry timeLimiterRegistry;
    
    /**
     * Programmatic timeout with WebClient.
     * Useful when you need more control.
     */
    public Mono<Customer> getCustomerReactive(String customerId) {
        TimeLimiter timeLimiter = timeLimiterRegistry.timeLimiter("systemApiClient");
        
        return webClient.get()
            .uri("/customers/{id}", customerId)
            .retrieve()
            .bodyToMono(Customer.class)
            .timeout(Duration.ofSeconds(3))  // WebClient native timeout
            .transformDeferred(TimeLimiterOperator.of(timeLimiter));  // Resilience4j timeout
    }
}
```

---

## Synchronous Wrapper Pattern

If you need synchronous API but want timeout protection:

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class CustomerApplicationService {
    
    private final SystemApiCustomerClient systemApiClient;
    
    @TimeLimiter(name = "systemApiClient", fallbackMethod = "getCustomerFallback")
    public CompletableFuture<Customer> getCustomerAsync(String customerId) {
        return CompletableFuture.supplyAsync(() -> 
            systemApiClient.findById(customerId)
        );
    }
    
    /**
     * Synchronous wrapper for callers that need blocking response.
     */
    public Customer getCustomer(String customerId) {
        try {
            return getCustomerAsync(customerId)
                .get(6, TimeUnit.SECONDS);  // Slightly longer than config timeout
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            throw new ServiceUnavailableException("Customer service timeout", e);
        }
    }
    
    private CompletableFuture<Customer> getCustomerFallback(String customerId, Exception ex) {
        return CompletableFuture.failedFuture(
            new ServiceUnavailableException("Customer service unavailable", ex)
        );
    }
}
```

---

## Testing

### Unit Test Example

```java
@ExtendWith(MockitoExtension.class)
class CustomerApplicationServiceTest {
    
    @Mock
    private SystemApiCustomerClient systemApiClient;
    
    @InjectMocks
    private CustomerApplicationService service;
    
    @Test
    void getCustomer_completesWithinTimeout() throws Exception {
        // Arrange
        Customer expected = new Customer("cust-123", "John Doe");
        when(systemApiClient.findById("cust-123")).thenReturn(expected);
        
        // Act
        CompletableFuture<Customer> future = service.getCustomer("cust-123");
        Customer result = future.get(1, TimeUnit.SECONDS);
        
        // Assert
        assertEquals(expected, result);
    }
    
    @Test
    void getCustomer_timesOut() {
        // Arrange - simulate slow response
        when(systemApiClient.findById("cust-123")).thenAnswer(inv -> {
            Thread.sleep(10000);  // Longer than timeout
            return new Customer("cust-123", "John Doe");
        });
        
        // Act
        CompletableFuture<Customer> future = service.getCustomer("cust-123");
        
        // Assert
        assertThrows(TimeoutException.class, 
            () -> future.get(5, TimeUnit.SECONDS));
    }
    
    @Test
    void getCustomer_fallbackOnTimeout() throws Exception {
        // Arrange - simulate slow response
        when(systemApiClient.findById("cust-123")).thenAnswer(inv -> {
            Thread.sleep(10000);
            return null;
        });
        
        // Act
        CompletableFuture<Customer> future = service.getCustomer("cust-123");
        Customer result = future.get(10, TimeUnit.SECONDS);  // Wait for fallback
        
        // Assert - should get fallback value
        assertEquals(Customer.UNKNOWN, result);
    }
}
```

---

## Monitoring and Observability

### Health Endpoint

TimeLimiter state exposed via `/actuator/health`:

```json
{
  "status": "UP",
  "components": {
    "timeLimiters": {
      "status": "UP",
      "details": {
        "systemApiClient": {
          "availablePermissions": 10,
          "numberOfWaitingThreads": 0
        }
      }
    }
  }
}
```

### Metrics Endpoint

```bash
curl http://localhost:8080/actuator/metrics/resilience4j.timelimiter.calls
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `resilience4j_timelimiter_calls{kind="successful"}` | Calls within timeout | - |
| `resilience4j_timelimiter_calls{kind="timeout"}` | Calls that timed out | > 5% of total |
| `resilience4j_timelimiter_calls{kind="failed"}` | Calls that failed | Any occurrence |

---

## Best Practices

### ✅ DO

1. **Always return CompletableFuture** for @TimeLimiter methods
2. **Set reasonable timeouts** based on SLA (typically 3-10s for APIs)
3. **Combine with Circuit Breaker** to stop calling hung services
4. **Log timeout occurrences** for troubleshooting
5. **Provide meaningful fallbacks** for user-facing operations
6. **Use shorter timeouts** for System API calls than for batch operations
7. **Monitor timeout rates** to detect degrading services

### ❌ DON'T

1. **Don't use @TimeLimiter on synchronous methods** - won't work
2. **Don't set very short timeouts** - causes false positives
3. **Don't set very long timeouts** - defeats the purpose
4. **Don't forget cancelRunningFuture** consideration for side effects
5. **Don't use for local computations** - use async processing
6. **Don't ignore TimeoutException** in fallbacks

---

## Common Pitfalls

### Pitfall 1: Not Using CompletableFuture

```java
// ❌ WRONG - @TimeLimiter does nothing on synchronous methods
@TimeLimiter(name = "service")
public Customer getCustomer(String id) {
    return client.findById(id);
}

// ✅ CORRECT - Return CompletableFuture
@TimeLimiter(name = "service")
public CompletableFuture<Customer> getCustomer(String id) {
    return CompletableFuture.supplyAsync(() -> client.findById(id));
}
```

### Pitfall 2: Fallback with Wrong Signature

```java
// ❌ WRONG - Fallback must return CompletableFuture
private Customer getCustomerFallback(String id, Exception ex) {
    return Customer.UNKNOWN;
}

// ✅ CORRECT
private CompletableFuture<Customer> getCustomerFallback(String id, Exception ex) {
    return CompletableFuture.completedFuture(Customer.UNKNOWN);
}
```

### Pitfall 3: Timeout Longer Than Caller's Timeout

```java
// ❌ WRONG - Client times out before server
// Server: timeoutDuration: 30s
// Client RestTemplate: readTimeout: 10s
// Result: Client times out, server keeps processing

// ✅ CORRECT - TimeLimiter shorter than client timeout
// Server: timeoutDuration: 5s
// Client RestTemplate: readTimeout: 10s
```

---

## Related Patterns

| Pattern | Combination | Benefit |
|---------|-------------|---------|
| **Circuit Breaker** | `@CircuitBreaker` + `@TimeLimiter` | Track timeout patterns, open circuit on repeated timeouts |
| **Retry** | `@TimeLimiter` + `@Retry` | Retry after timeout (with shorter timeout) |
| **Bulkhead** | `@Bulkhead` + `@TimeLimiter` | Limit concurrent slow calls |

**Recommended order for System API calls:**
```java
@CircuitBreaker(name = "systemApi")  // Outer
@TimeLimiter(name = "systemApi")
@Retry(name = "systemApi")            // Inner
public CompletableFuture<Result> callSystemApi() { }
```

---

## References

### Documentation
- [Resilience4j TimeLimiter Docs](https://resilience4j.readme.io/docs/timeout)
- [Spring Boot Integration](https://resilience4j.readme.io/docs/getting-started-3)

### Related
- **Implements:** ADR-004 (Resilience Patterns)
- **Module:** mod-code-003-timeout-java-resilience4j
- **Complements:** ERI-008 (Circuit Breaker), ERI-009 (Retry)

---

## Changelog

### v1.0 (2025-11-28)
- Initial version
- Complete TimeLimiter implementation for Java/Spring Boot
- Multiple patterns: basic, with fallback, combined resilience stack
- Unit test examples
- Best practices and common pitfalls

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-010-timeout-constraints
  version: "1.0"
  eri_reference: eri-code-010-timeout-java-resilience4j
  adr_reference: adr-004-resilience-patterns
  
  structural_constraints:
    # Annotation Placement
    - id: annotation-in-application-layer
      rule: "@TimeLimiter annotation MUST be in application layer only (not in domain)"
      validation: "grep -r '@TimeLimiter' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: application
      
    - id: annotation-not-in-controller
      rule: "@TimeLimiter SHOULD NOT be directly on controller methods"
      validation: "grep -r '@TimeLimiter' src/*/java/**/adapter/rest/controller/ returns empty"
      severity: WARNING
      layer: adapter
      
    # Return Type Requirements
    - id: return-completable-future
      rule: "Methods with @TimeLimiter MUST return CompletableFuture<T>"
      validation: "All @TimeLimiter methods have CompletableFuture return type"
      severity: ERROR
      
    # Fallback Requirements
    - id: fallback-returns-future
      rule: "Fallback method MUST return CompletableFuture<T>"
      validation: "Fallback methods return CompletableFuture"
      severity: ERROR
      
    - id: fallback-signature
      rule: "Fallback method MUST have Exception as last parameter"
      validation: "Fallback methods signature includes Exception parameter"
      severity: ERROR
      
    # Naming
    - id: timelimiter-name-matches-config
      rule: "TimeLimiter name MUST match a configured instance in application.yml"
      validation: "Name parameter value exists in resilience4j.timelimiter.instances"
      severity: ERROR
      
    # Combination Order
    - id: circuitbreaker-before-timelimiter
      rule: "When combined, @CircuitBreaker MUST be declared before @TimeLimiter"
      validation: "@CircuitBreaker annotation appears before @TimeLimiter"
      severity: ERROR
      
    - id: timelimiter-before-retry
      rule: "When combined, @TimeLimiter MUST be declared before @Retry"
      validation: "@TimeLimiter annotation appears before @Retry"
      severity: ERROR

  configuration_constraints:
    - id: resilience4j-timelimiter-config-exists
      rule: "resilience4j.timelimiter section MUST exist in application.yml"
      validation: "application.yml contains resilience4j.timelimiter key"
      severity: ERROR
      
    - id: timeout-duration-reasonable
      rule: "timeoutDuration SHOULD be between 1s and 30s for API calls"
      validation: "timeoutDuration >= 1s AND timeoutDuration <= 30s"
      severity: WARNING
      
    - id: timeout-shorter-than-client
      rule: "Server timeout SHOULD be shorter than client timeout"
      validation: "timeoutDuration < client connection timeout"
      severity: WARNING

  dependency_constraints:
    required:
      - groupId: io.github.resilience4j
        artifactId: resilience4j-spring-boot3
        minVersion: "2.0.0"
        reason: "TimeLimiter implementation for Spring Boot 3.x"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-aop
        reason: "Required for @TimeLimiter annotation processing"
        
    optional:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-actuator
        reason: "Required for timelimiter health and metrics endpoints"

  testing_constraints:
    - id: timeout-scenario-tested
      rule: "Timeout scenario MUST be tested"
      validation: "Test exists that verifies timeout behavior"
      severity: ERROR
      
    - id: success-within-timeout-tested
      rule: "Success within timeout scenario MUST be tested"
      validation: "Test verifies successful completion within timeout"
      severity: ERROR
      
    - id: fallback-on-timeout-tested
      rule: "Fallback on timeout scenario SHOULD be tested"
      validation: "Test verifies fallback is called on timeout"
      severity: WARNING
```

---

## Alternative Strategy: Client-Level Timeout

While `@TimeLimiter` provides powerful annotation-based timeout control, it requires methods to return `CompletableFuture<T>`, which adds complexity in synchronous codebases. For simpler cases, **client-level timeouts** offer an alternative.

### When to Use Client-Level Timeout

| Criteria | @TimeLimiter (Resilience4j) | Client-Level Timeout |
|----------|----------------------------|---------------------|
| **Code style** | Async (CompletableFuture) | Synchronous |
| **Granularity** | Per-method | Per-client |
| **Configuration** | YAML + annotations | YAML only |
| **Fallback support** | Built-in | Manual (try-catch) |
| **Metrics** | Automatic via Actuator | Manual or none |
| **Complexity** | Higher (async wrapper) | Lower |

### Default Recommendation

For **new Domain API projects** consuming System APIs:
- **Default:** Client-level timeout (simpler, synchronous)
- **Use @TimeLimiter when:** Need per-method control, fallback methods, or existing async patterns

### Client-Level Implementation (RestClient)

```java
// RestClientConfig.java
@Configuration
public class RestClientConfig {
    
    @Value("${integration.system-api.connect-timeout:5s}")
    private Duration connectTimeout;
    
    @Value("${integration.system-api.read-timeout:10s}")
    private Duration readTimeout;
    
    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder()
            .requestFactory(clientHttpRequestFactory());
    }
    
    @Bean
    public ClientHttpRequestFactory clientHttpRequestFactory() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(connectTimeout);
        factory.setReadTimeout(readTimeout);
        return factory;
    }
}
```

```yaml
# application.yml
integration:
  system-api:
    connect-timeout: 5s
    read-timeout: 10s
```

### Client-Level Implementation (RestTemplate)

```java
@Bean
public RestTemplate restTemplate(RestTemplateBuilder builder) {
    return builder
        .setConnectTimeout(Duration.ofSeconds(5))
        .setReadTimeout(Duration.ofSeconds(10))
        .build();
}
```

### Client-Level Implementation (WebClient)

```java
@Bean
public WebClient webClient() {
    HttpClient httpClient = HttpClient.create()
        .responseTimeout(Duration.ofSeconds(10))
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 5000);
    
    return WebClient.builder()
        .clientConnector(new ReactorClientHttpConnector(httpClient))
        .build();
}
```

### Strategy Selection in generation-request.json

```json
{
  "features": {
    "resilience": {
      "timeout": {
        "enabled": true,
        "strategy": "client_level",  // or "timelimiter"
        "duration": "10s",
        "connectTimeout": "5s"
      }
    }
  }
}
```

| Strategy Value | Module | Template |
|----------------|--------|----------|
| `timelimiter` | mod-code-003-timeout-java-resilience4j | @TimeLimiter annotation |
| `client_level` | mod-code-018-api-integration-rest-java-spring | RestClient/RestTemplate config |

### Module Reference

- **@TimeLimiter strategy:** [mod-code-003-timeout-java-resilience4j](../../skills/modules/mod-code-003-timeout-java-resilience4j/)
- **Client-level strategy:** [mod-code-018-api-integration-rest-java-spring](../../skills/modules/mod-code-018-api-integration-rest-java-spring/)

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills based on this ERI.

```yaml
eri_constraints:
  id: eri-code-010-timeout-constraints
  version: "1.1"
  eri_reference: eri-code-010-timeout-java-resilience4j
  adr_reference: adr-004-resilience-patterns
  
  implementation_options:
    default: client-timeout
    options:
      - id: client-timeout
        name: "Client-level Timeout"
        status: default
        recommended_when:
          - "Synchronous code patterns"
          - "New Spring Boot 3.2+ projects"
          - "Simple timeout without specific fallback"
          
      - id: annotation-async
        name: "@TimeLimiter Annotation"
        status: alternative
        recommended_when:
          - "Service uses async patterns with CompletableFuture"
          - "Specific fallback behavior needed on timeout"
          - "Per-method timeout configuration required"
  
  structural_constraints:
    - id: timelimiter-completable-future
      rule: "Methods with @TimeLimiter MUST return CompletableFuture<T>"
      validation: "Methods annotated with @TimeLimiter have CompletableFuture return type"
      severity: ERROR
      applies_to: [annotation-async]
      
    - id: timelimiter-in-application-layer
      rule: "@TimeLimiter annotation MUST be in application layer"
      validation: "@TimeLimiter found only in application/service/ classes"
      severity: ERROR
      applies_to: [annotation-async]
      
    - id: client-timeout-in-config
      rule: "Client timeout MUST be configured in infrastructure layer"
      validation: "Timeout configuration in infrastructure/config/ classes"
      severity: ERROR
      applies_to: [client-timeout]
      
  configuration_constraints:
    - id: timeout-externalized
      rule: "Timeout values MUST be externalized in application.yml"
      validation: "Timeout values configurable via properties"
      severity: ERROR
```

---

**Status:** ✅ Production-Ready  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0
