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
tags:
  - java
  - spring-boot
  - resilience4j
  - retry
  - fault-tolerance
automated_by:
  - skill-code-002-add-retry-java-resilience4j
---

# ERI-CODE-009: Retry Pattern - Java/Spring Boot with Resilience4j

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Retry pattern in Java/Spring Boot microservices using Resilience4j.

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

## Dependencies

### Maven (pom.xml)

```xml
<dependencies>
    <dependency>
        <groupId>io.github.resilience4j</groupId>
        <artifactId>resilience4j-spring-boot3</artifactId>
        <version>2.1.0</version>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-aop</artifactId>
    </dependency>
</dependencies>
```

---

## Configuration

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
          - java.io.IOException
        ignoreExceptions:
          - java.lang.IllegalArgumentException
          - com.company.exception.BusinessException

    instances:
      systemApiClient:
        baseConfig: default
        maxAttempts: 3
        waitDuration: 200ms
        exponentialBackoffMultiplier: 1.5
```

---

## Implementation Patterns

### Pattern 1: Basic Retry

```java
@Service
@Slf4j
public class CustomerApplicationService {

    private final SystemApiCustomerClient systemApiClient;

    @Retry(name = "systemApiClient")
    public Customer getCustomer(String customerId) {
        log.debug("Fetching customer: {}", customerId);
        return systemApiClient.findById(customerId);
    }
}
```

### Pattern 2: Retry with Fallback

```java
@Service
@Slf4j
public class PaymentApplicationService {

    @Retry(name = "paymentService", fallbackMethod = "getPaymentStatusFallback")
    public PaymentStatus getPaymentStatus(String transactionId) {
        return systemApiClient.getStatus(transactionId);
    }

    private PaymentStatus getPaymentStatusFallback(String transactionId, Exception ex) {
        log.warn("All retries exhausted for: {}. Error: {}",
                 transactionId, ex.getMessage());
        return cacheService.getCachedStatus(transactionId)
            .orElse(PaymentStatus.UNKNOWN);
    }
}
```

### Pattern 3: Combined with Circuit Breaker

```java
@Service
public class InventoryApplicationService {

    // Order matters! CircuitBreaker is outer, Retry is inner
    @CircuitBreaker(name = "inventoryService", fallbackMethod = "checkStockFallback")
    @Retry(name = "inventoryService")
    public StockLevel checkStock(String productId) {
        return systemApiClient.getStockLevel(productId);
    }
}
```

---

## Best Practices

### ✅ DO

1. Only retry idempotent operations
2. Use exponential backoff for external APIs
3. Set reasonable maxAttempts (3-5)
4. Combine with Circuit Breaker
5. Configure retryExceptions explicitly

### ❌ DON'T

1. Don't retry non-idempotent operations without idempotency keys
2. Don't retry business exceptions
3. Don't set very high maxAttempts
4. Don't retry authentication failures

---

## Common Pitfalls

### Wrong Annotation Order

```java
// ❌ WRONG - Retry outside Circuit Breaker
@Retry(name = "service")
@CircuitBreaker(name = "service")
public Result doSomething() { }

// ✅ CORRECT - CircuitBreaker outside, Retry inside
@CircuitBreaker(name = "service")
@Retry(name = "service")
public Result doSomething() { }
```

---

**Status:** ✅ Production-Ready
**Framework:** Java/Spring Boot
**Library:** Resilience4j 2.1.0
