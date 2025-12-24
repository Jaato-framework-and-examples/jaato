---
id: eri-code-009-retry-java-resilience4j
title: "ERI-CODE-009: Retry Pattern - Java/Spring Boot with Resilience4j"
sidebar_label: "Retry (Java)"
version: 1.0
date: 2025-11-28
updated: 2025-11-28
status: Active
author: "Architecture Team"
domain: code
pattern: retry
framework: java
library: resilience4j
library_version: 2.1.0
implements:
  - adr-004-resilience-patterns
implements_pattern: retry
tags:
  - java
  - spring-boot
  - resilience4j
  - retry
  - fault-tolerance
  - microservices
related:
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-code-010-timeout-java-resilience4j
  - eri-code-011-rate-limiter-java-resilience4j
automated_by:
  - skill-code-002-add-retry-java-resilience4j
cross_domain_usage: qa
---

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Retry pattern in Java/Spring Boot microservices using Resilience4j.

**What this implements:**
- Automatic retry for transient failures
- Configurable retry attempts and intervals
- Exponential backoff strategies
- Exception-based retry decisions
- Monitoring and metrics integration

**When to use:**
- Transient network failures
- Temporary service unavailability
- Database connection timeouts
- External API rate limiting (with backoff)

**When NOT to use:**
- Business logic failures (validation errors)
- Authentication/authorization failures
- Non-idempotent operations without careful consideration

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Spring Boot** | 3.2.x | Application framework |
| **Resilience4j** | 2.1.0 | Retry library |
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
│       └── {Service}ApplicationService.java    # @Retry annotations here
├── adapter/
│   └── client/
│       └── {External}Client.java               # External service client
└── infrastructure/
    └── config/
        └── ResilienceConfig.java               # Programmatic configuration (optional)

src/main/resources/
└── application.yml                             # Retry configuration

src/test/java/com/company/{service}/
└── application/
    └── service/
        └── {Service}ApplicationServiceTest.java  # Unit tests with retry scenarios
```

**Key Placement Rules:**
- `@Retry` annotations go on **Application Service** methods (not domain)
- Configuration is externalized in `application.yml`
- Combine with `@CircuitBreaker` for comprehensive resilience

---

## Configuration

### application.yml

```yaml
resilience4j:
  retry:
    configs:
      # Default configuration for all retries
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
          - java.lang.IllegalArgumentException
          - com.company.service.exception.BusinessException
    
    instances:
      # Specific instance for payment service
      paymentService:
        baseConfig: default
        maxAttempts: 5
        waitDuration: 1000ms
      
      # Specific instance for system API calls
      systemApiClient:
        baseConfig: default
        maxAttempts: 3
        waitDuration: 200ms
        exponentialBackoffMultiplier: 1.5

# Expose retry metrics
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,retries,retryevents
  endpoint:
    health:
      show-details: always
```

### Configuration Parameters Explained

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| **maxAttempts** | 3 | Maximum retry attempts (including initial) | 3-5 for external APIs |
| **waitDuration** | 500ms | Wait time between retries | 200-1000ms typical |
| **enableExponentialBackoff** | false | Use exponential backoff | true for external APIs |
| **exponentialBackoffMultiplier** | 2 | Multiplier for backoff | 1.5-2.0 recommended |
| **retryExceptions** | empty | Exceptions that trigger retry | Network/IO exceptions |
| **ignoreExceptions** | empty | Exceptions that skip retry | Business exceptions |

---

## Implementation Patterns

### Pattern 1: Basic Retry

```java
package com.company.customer.application.service;

import io.github.resilience4j.retry.annotation.Retry;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class CustomerApplicationService {
    
    private final SystemApiCustomerClient systemApiClient;
    
    public CustomerApplicationService(SystemApiCustomerClient systemApiClient) {
        this.systemApiClient = systemApiClient;
    }
    
    /**
     * Fetch customer with automatic retry on transient failures.
     */
    @Retry(name = "systemApiClient")
    public Customer getCustomer(String customerId) {
        log.debug("Fetching customer: {}", customerId);
        return systemApiClient.findById(customerId);
    }
}
```

**Key points:**
- ✅ `@Retry` annotation with name matching config
- ✅ Method is idempotent (safe to retry)
- ✅ Logging for observability

---

### Pattern 2: Retry with Fallback

```java
@Service
@Slf4j
public class PaymentApplicationService {
    
    private final PaymentSystemApiClient systemApiClient;
    private final PaymentCacheService cacheService;
    
    @Retry(name = "paymentService", fallbackMethod = "getPaymentStatusFallback")
    public PaymentStatus getPaymentStatus(String transactionId) {
        log.debug("Checking payment status: {}", transactionId);
        return systemApiClient.getStatus(transactionId);
    }
    
    /**
     * Fallback after all retries exhausted.
     * Signature: same params + Exception at end.
     */
    private PaymentStatus getPaymentStatusFallback(String transactionId, Exception ex) {
        log.warn("All retries exhausted for transaction: {}. Error: {}", 
                 transactionId, ex.getMessage());
        
        // Try cache as fallback
        return cacheService.getCachedStatus(transactionId)
            .orElse(PaymentStatus.UNKNOWN);
    }
}
```

**Key points:**
- ✅ Fallback method for graceful degradation
- ✅ Fallback signature: same params + `Exception`
- ✅ Cache as fallback strategy

---

### Pattern 3: Retry Combined with Circuit Breaker

```java
@Service
@Slf4j
public class InventoryApplicationService {
    
    private final InventorySystemApiClient systemApiClient;
    
    /**
     * Combined resilience: Retry first, then Circuit Breaker.
     * Order matters! Retry is inner, CircuitBreaker is outer.
     * 
     * Flow: Request → CircuitBreaker → Retry → Actual Call
     * If circuit is OPEN, no retries attempted.
     */
    @CircuitBreaker(name = "inventoryService", fallbackMethod = "checkStockFallback")
    @Retry(name = "inventoryService")
    public StockLevel checkStock(String productId) {
        log.debug("Checking stock for product: {}", productId);
        return systemApiClient.getStockLevel(productId);
    }
    
    private StockLevel checkStockFallback(String productId, Exception ex) {
        log.warn("Inventory service unavailable for product: {}", productId);
        return StockLevel.UNKNOWN;
    }
}
```

**Annotation Order:**
- `@CircuitBreaker` → `@Retry` → Actual method
- Retries happen INSIDE the circuit breaker
- If circuit opens, retries stop immediately

---

### Pattern 4: Retry with Exponential Backoff (Programmatic)

```java
@Configuration
public class ResilienceConfig {
    
    @Bean
    public RetryConfig customRetryConfig() {
        return RetryConfig.custom()
            .maxAttempts(5)
            .waitDuration(Duration.ofMillis(500))
            .intervalFunction(IntervalFunction.ofExponentialBackoff(
                Duration.ofMillis(500),  // Initial interval
                2.0,                      // Multiplier
                Duration.ofSeconds(10)    // Max interval
            ))
            .retryOnException(e -> e instanceof ConnectException 
                                || e instanceof SocketTimeoutException)
            .ignoreExceptions(BusinessException.class)
            .build();
    }
    
    @Bean
    public RetryRegistry retryRegistry(RetryConfig customRetryConfig) {
        return RetryRegistry.of(customRetryConfig);
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
    void getCustomer_successOnFirstAttempt() {
        // Arrange
        Customer expected = new Customer("cust-123", "John Doe");
        when(systemApiClient.findById("cust-123")).thenReturn(expected);
        
        // Act
        Customer result = service.getCustomer("cust-123");
        
        // Assert
        assertEquals(expected, result);
        verify(systemApiClient, times(1)).findById("cust-123");
    }
    
    @Test
    void getCustomer_successAfterRetry() {
        // Arrange
        Customer expected = new Customer("cust-123", "John Doe");
        when(systemApiClient.findById("cust-123"))
            .thenThrow(new ConnectException("Connection refused"))
            .thenThrow(new ConnectException("Connection refused"))
            .thenReturn(expected);
        
        // Act
        Customer result = service.getCustomer("cust-123");
        
        // Assert
        assertEquals(expected, result);
        verify(systemApiClient, times(3)).findById("cust-123");
    }
    
    @Test
    void getCustomer_failsAfterMaxRetries() {
        // Arrange
        when(systemApiClient.findById("cust-123"))
            .thenThrow(new ConnectException("Connection refused"));
        
        // Act & Assert
        assertThrows(ConnectException.class, 
            () -> service.getCustomer("cust-123"));
        verify(systemApiClient, times(3)).findById("cust-123"); // maxAttempts = 3
    }
    
    @Test
    void getCustomer_noRetryOnBusinessException() {
        // Arrange
        when(systemApiClient.findById("invalid"))
            .thenThrow(new BusinessException("Customer not found"));
        
        // Act & Assert
        assertThrows(BusinessException.class, 
            () -> service.getCustomer("invalid"));
        verify(systemApiClient, times(1)).findById("invalid"); // No retry
    }
}
```

---

## Monitoring and Observability

### Health Endpoint

Retry state exposed via `/actuator/health`:

```json
{
  "status": "UP",
  "components": {
    "retries": {
      "status": "UP",
      "details": {
        "systemApiClient": {
          "numberOfSuccessfulCallsWithoutRetryAttempt": 150,
          "numberOfSuccessfulCallsWithRetryAttempt": 12,
          "numberOfFailedCallsWithoutRetryAttempt": 0,
          "numberOfFailedCallsWithRetryAttempt": 2
        }
      }
    }
  }
}
```

### Metrics Endpoint

Retry metrics via `/actuator/metrics`:

```bash
# Query metrics
curl http://localhost:8080/actuator/metrics/resilience4j.retry.calls

# Sample response
{
  "name": "resilience4j.retry.calls",
  "measurements": [
    {"statistic": "COUNT", "value": 164.0}
  ],
  "availableTags": [
    {"tag": "name", "values": ["systemApiClient", "paymentService"]},
    {"tag": "kind", "values": ["successful_without_retry", "successful_with_retry", "failed_without_retry", "failed_with_retry"]}
  ]
}
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `resilience4j_retry_calls{kind="successful_with_retry"}` | Calls that needed retry | > 10% of total |
| `resilience4j_retry_calls{kind="failed_with_retry"}` | Failed after all retries | Any occurrence |

---

## Best Practices

### ✅ DO

1. **Only retry idempotent operations** - GET, DELETE (with care), safe POSTs
2. **Use exponential backoff** for external APIs to avoid thundering herd
3. **Set reasonable maxAttempts** - 3-5 is usually sufficient
4. **Combine with Circuit Breaker** for comprehensive resilience
5. **Log retry attempts** for troubleshooting
6. **Configure retryExceptions explicitly** - don't retry everything
7. **Monitor retry metrics** to detect degrading services

### ❌ DON'T

1. **Don't retry non-idempotent operations** without idempotency keys
2. **Don't retry business exceptions** (validation, not found, etc.)
3. **Don't set very high maxAttempts** - delays user response
4. **Don't use fixed intervals** for external APIs - causes spikes
5. **Don't ignore the exception** in fallback method
6. **Don't retry authentication failures** - they won't magically succeed

---

## Common Pitfalls

### Pitfall 1: Retrying Non-Idempotent Operations

```java
// ❌ DANGEROUS - Payment might be processed multiple times
@Retry(name = "paymentService")
public PaymentResult processPayment(PaymentRequest request) {
    return paymentClient.charge(request);
}

// ✅ SAFE - Use idempotency key
@Retry(name = "paymentService")
public PaymentResult processPayment(PaymentRequest request) {
    return paymentClient.charge(request.getIdempotencyKey(), request);
}
```

### Pitfall 2: Wrong Annotation Order

```java
// ❌ WRONG - Retry outside Circuit Breaker
// Each retry attempt is treated as separate call for circuit breaker
@Retry(name = "service")
@CircuitBreaker(name = "service")
public Result doSomething() { }

// ✅ CORRECT - CircuitBreaker outside, Retry inside
@CircuitBreaker(name = "service")
@Retry(name = "service")
public Result doSomething() { }
```

### Pitfall 3: Retrying All Exceptions

```java
// ❌ WRONG - Will retry even business exceptions
resilience4j:
  retry:
    instances:
      myService:
        maxAttempts: 3
        # No retryExceptions configured = retries everything!

// ✅ CORRECT - Explicit exception configuration
resilience4j:
  retry:
    instances:
      myService:
        maxAttempts: 3
        retryExceptions:
          - java.net.ConnectException
          - java.io.IOException
        ignoreExceptions:
          - com.company.exception.BusinessException
```

---

## Related Patterns

| Pattern | Combination | Benefit |
|---------|-------------|---------|
| **Circuit Breaker** | `@CircuitBreaker` + `@Retry` | Stop retrying when service is down |
| **Timeout** | `@TimeLimiter` + `@Retry` | Don't wait forever between retries |
| **Bulkhead** | `@Bulkhead` + `@Retry` | Limit concurrent retry storms |

**Recommended combination for System API calls:**
```java
@CircuitBreaker(name = "systemApi")
@TimeLimiter(name = "systemApi")
@Retry(name = "systemApi")
public Result callSystemApi() { }
```

---

## References

### Documentation
- [Resilience4j Retry Docs](https://resilience4j.readme.io/docs/retry)
- [Spring Boot Integration](https://resilience4j.readme.io/docs/getting-started-3)

### Related
- **Implements:** ADR-004 (Resilience Patterns)
- **Module:** mod-code-002-retry-java-resilience4j
- **Complements:** ERI-008 (Circuit Breaker), ERI-010 (Timeout)

---

## Changelog

### v1.0 (2025-11-28)
- Initial version
- Complete Retry implementation for Java/Spring Boot
- Multiple patterns: basic, with fallback, combined with circuit breaker
- Unit test examples
- Best practices and common pitfalls

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-009-retry-constraints
  version: "1.0"
  eri_reference: eri-code-009-retry-java-resilience4j
  adr_reference: adr-004-resilience-patterns
  
  structural_constraints:
    # Annotation Placement
    - id: annotation-in-application-layer
      rule: "@Retry annotation MUST be in application layer only (not in domain)"
      validation: "grep -r '@Retry' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: application
      
    - id: annotation-not-in-controller
      rule: "@Retry SHOULD NOT be directly on controller methods"
      validation: "grep -r '@Retry' src/*/java/**/adapter/rest/controller/ returns empty"
      severity: WARNING
      layer: adapter
      
    # Fallback Requirements
    - id: fallback-signature
      rule: "Fallback method (if used) MUST have Exception as last parameter"
      validation: "Fallback methods signature includes Exception parameter at the end"
      severity: ERROR
      
    - id: fallback-same-return-type
      rule: "Fallback method MUST have same return type as protected method"
      validation: "Fallback method return type matches annotated method return type"
      severity: ERROR
      
    # Naming
    - id: retry-name-matches-config
      rule: "Retry name MUST match a configured instance in application.yml"
      validation: "Name parameter value exists in resilience4j.retry.instances"
      severity: ERROR
      
    # Combination Order
    - id: circuitbreaker-before-retry
      rule: "When combined, @CircuitBreaker MUST be declared before @Retry"
      validation: "@CircuitBreaker annotation appears before @Retry on same method"
      severity: ERROR

  configuration_constraints:
    - id: resilience4j-retry-config-exists
      rule: "resilience4j.retry section MUST exist in application.yml"
      validation: "application.yml contains resilience4j.retry key"
      severity: ERROR
      
    - id: retry-exceptions-configured
      rule: "retryExceptions SHOULD be explicitly configured"
      validation: "resilience4j.retry.instances.*.retryExceptions is not empty"
      severity: WARNING
      
    - id: ignore-exceptions-configured
      rule: "ignoreExceptions SHOULD include business exceptions"
      validation: "resilience4j.retry.instances.*.ignoreExceptions is configured"
      severity: WARNING
      
    - id: exponential-backoff-for-external
      rule: "External API calls SHOULD use exponential backoff"
      validation: "enableExponentialBackoff is true for external API retry configs"
      severity: WARNING
      
    - id: max-attempts-reasonable
      rule: "maxAttempts SHOULD be between 2 and 5"
      validation: "maxAttempts >= 2 AND maxAttempts <= 5"
      severity: WARNING

  dependency_constraints:
    required:
      - groupId: io.github.resilience4j
        artifactId: resilience4j-spring-boot3
        minVersion: "2.0.0"
        reason: "Retry implementation for Spring Boot 3.x"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-aop
        reason: "Required for @Retry annotation processing"
        
    optional:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-actuator
        reason: "Required for retry health and metrics endpoints"

  testing_constraints:
    - id: retry-success-tested
      rule: "Success after retry scenario MUST be tested"
      validation: "Test exists that verifies success after initial failure"
      severity: ERROR
      
    - id: max-retries-tested
      rule: "Max retries exhausted scenario MUST be tested"
      validation: "Test verifies behavior when all retries fail"
      severity: ERROR
      
    - id: no-retry-on-business-exception
      rule: "Business exception no-retry behavior SHOULD be tested"
      validation: "Test verifies business exceptions are not retried"
      severity: WARNING
      
    - id: mock-transient-failures
      rule: "Tests MUST mock transient failures to verify retry behavior"
      validation: "Test uses when().thenThrow().thenReturn() pattern"
      severity: ERROR
```

---

**Status:** ✅ Production-Ready  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0
