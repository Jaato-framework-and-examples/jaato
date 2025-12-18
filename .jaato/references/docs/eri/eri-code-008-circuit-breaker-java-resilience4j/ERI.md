---
id: eri-code-008-circuit-breaker-java-resilience4j
title: "ERI-CODE-008: Circuit Breaker Pattern - Java/Spring Boot with Resilience4j"
sidebar_label: "Circuit Breaker (Java)"
version: 2.1
date: 2024-11-20
updated: 2025-11-27
status: Active
author: "Architecture Team"
domain: code
pattern: circuit-breaker
framework: java
library: resilience4j
library_version: 2.1.0
implements:
  - adr-004-resilience-patterns
tags:
  - java
  - spring-boot
  - resilience4j
  - circuit-breaker
  - fault-tolerance
automated_by:
  - skill-code-001-add-circuit-breaker-java-resilience4j
---

# ERI-CODE-008: Circuit Breaker Pattern - Java/Spring Boot with Resilience4j

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Circuit Breaker pattern in Java/Spring Boot microservices using Resilience4j.

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
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
</dependencies>
```

---

## Configuration

### application.yml

```yaml
resilience4j:
  circuitbreaker:
    configs:
      default:
        slidingWindowType: COUNT_BASED
        slidingWindowSize: 100
        minimumNumberOfCalls: 10
        failureRateThreshold: 50
        waitDurationInOpenState: 60000
        permittedNumberOfCallsInHalfOpenState: 3
        automaticTransitionFromOpenToHalfOpenEnabled: true
        recordExceptions:
          - java.net.ConnectException
          - java.net.SocketTimeoutException
          - org.springframework.web.client.ResourceAccessException
        ignoreExceptions:
          - java.lang.IllegalArgumentException

    instances:
      paymentService:
        baseConfig: default
        failureRateThreshold: 40
        waitDurationInOpenState: 30000

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,circuitbreakers
  health:
    circuitbreakers:
      enabled: true
```

---

## Implementation Pattern

### Basic Circuit Breaker with Fallback

```java
@Service
@Slf4j
public class PaymentService {

    private final PaymentApiClient paymentApiClient;
    private final PaymentCacheService cacheService;

    @CircuitBreaker(name = "paymentService", fallbackMethod = "processPaymentFallback")
    public PaymentResponse processPayment(PaymentRequest request) {
        log.debug("Processing payment for order: {}", request.getOrderId());
        return paymentApiClient.charge(request);
    }

    private PaymentResponse processPaymentFallback(PaymentRequest request,
                                                   Throwable throwable) {
        log.warn("Circuit breaker fallback for order: {}. Reason: {}",
                 request.getOrderId(), throwable.getMessage());

        return cacheService.getLastSuccessfulPayment(request.getOrderId())
            .orElseThrow(() -> new PaymentUnavailableException(
                "Payment service temporarily unavailable", throwable));
    }
}
```

**Key points:**
- ✅ `@CircuitBreaker` annotation with name matching config
- ✅ `fallbackMethod` specified
- ✅ Fallback signature: same params + `Throwable` at end
- ✅ Graceful degradation (cache or meaningful error)

---

## Best Practices

### ✅ DO

1. Always provide meaningful fallbacks for user-facing operations
2. Log fallback triggers with context for troubleshooting
3. Use cached data when acceptable as fallback strategy
4. Configure per service based on SLA and criticality
5. Monitor circuit breaker states in production

### ❌ DON'T

1. Don't ignore the Throwable in fallback method
2. Don't put business logic in fallback methods
3. Don't use circuit breaker for business rule failures
4. Don't set thresholds too low (causes false positives)

---

## Related Patterns

| Pattern | Combination | Benefit |
|---------|-------------|---------|
| **Retry** | Circuit Breaker + Retry | Retry transient, circuit for persistent |
| **Timeout** | Circuit Breaker + Timeout | Timeout prevents hanging |
| **Bulkhead** | Bulkhead + Circuit Breaker | Isolate failing services |

---

**Status:** ✅ Production-Ready
**Framework:** Java/Spring Boot
**Library:** Resilience4j 2.1.0
