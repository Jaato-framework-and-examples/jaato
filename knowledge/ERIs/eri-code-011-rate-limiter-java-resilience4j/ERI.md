---
id: eri-code-011-rate-limiter-java-resilience4j
title: "ERI-CODE-011: Rate Limiter Pattern - Java/Spring Boot with Resilience4j"
sidebar_label: "Rate Limiter (Java)"
version: 1.0
date: 2025-11-28
updated: 2025-11-28
status: Active
author: "Architecture Team"
domain: code
pattern: rate-limiter
framework: java
library: resilience4j
library_version: 2.1.0
implements:
  - adr-004-resilience-patterns
implements_pattern: rate-limiter
tags:
  - java
  - spring-boot
  - resilience4j
  - rate-limiter
  - throttling
  - fault-tolerance
  - microservices
related:
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-code-009-retry-java-resilience4j
  - eri-code-010-timeout-java-resilience4j
automated_by:
  - skill-code-004-add-rate-limiter-java-resilience4j
cross_domain_usage: qa
---

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Rate Limiter pattern in Java/Spring Boot microservices using Resilience4j.

**What this implements:**
- Request rate limiting to protect downstream services
- Configurable rate limits (requests per time period)
- Overflow handling strategies
- Monitoring and metrics integration

**When to use:**
- Protect System APIs with limited capacity
- Prevent abuse of public APIs
- Respect third-party API rate limits
- Control outbound traffic to legacy systems
- Implement fair resource sharing

**When NOT to use:**
- Internal service-to-service calls (usually)
- Operations with highly variable execution times (use Bulkhead)
- Single-request protection (use Circuit Breaker)

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Spring Boot** | 3.2.x | Application framework |
| **Resilience4j** | 2.1.0 | Rate limiter library |
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
│       └── {Service}ApplicationService.java    # @RateLimiter annotations here
├── adapter/
│   └── client/
│       └── {External}Client.java               # External service client
└── infrastructure/
    └── config/
        └── ResilienceConfig.java               # Programmatic configuration (optional)

src/main/resources/
└── application.yml                             # Rate limiter configuration

src/test/java/com/company/{service}/
└── application/
    └── service/
        └── {Service}ApplicationServiceTest.java
```

**Key Placement Rules:**
- `@RateLimiter` annotations go on **Application Service** methods (not domain)
- Configuration is externalized in `application.yml`
- Consider combining with Circuit Breaker for comprehensive protection

---

## Configuration

### application.yml

```yaml
resilience4j:
  ratelimiter:
    configs:
      # Default configuration for all rate limiters
      default:
        limitForPeriod: 10
        limitRefreshPeriod: 1s
        timeoutDuration: 500ms
        registerHealthIndicator: true
        eventConsumerBufferSize: 100
    
    instances:
      # Rate limit for System API calls (conservative)
      systemApiClient:
        baseConfig: default
        limitForPeriod: 50
        limitRefreshPeriod: 1s
        timeoutDuration: 0  # Fail immediately if limit reached
      
      # Rate limit for external payment API
      paymentApi:
        baseConfig: default
        limitForPeriod: 100
        limitRefreshPeriod: 1s
        timeoutDuration: 2s  # Wait up to 2s for permit
      
      # Rate limit for batch operations (lower rate)
      batchService:
        baseConfig: default
        limitForPeriod: 5
        limitRefreshPeriod: 1s
        timeoutDuration: 0

# Expose rate limiter metrics
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,ratelimiters,ratelimiterevents
  endpoint:
    health:
      show-details: always
  health:
    ratelimiters:
      enabled: true
```

### Configuration Parameters Explained

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| **limitForPeriod** | 50 | Max requests per refresh period | Based on downstream capacity |
| **limitRefreshPeriod** | 500ns | Period for limit refresh | 1s for TPS, 1m for TPM |
| **timeoutDuration** | 5s | Max wait time for permit | 0 for fail-fast, 1-5s for wait |
| **registerHealthIndicator** | false | Expose in health endpoint | true for production |

---

## Implementation Patterns

### Pattern 1: Basic Rate Limiter

```java
package com.company.customer.application.service;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class CustomerApplicationService {
    
    private final SystemApiCustomerClient systemApiClient;
    
    /**
     * Rate-limited call to System API.
     * Fails fast if rate limit exceeded.
     */
    @RateLimiter(name = "systemApiClient")
    public Customer getCustomer(String customerId) {
        log.debug("Fetching customer: {}", customerId);
        return systemApiClient.findById(customerId);
    }
}
```

**Key points:**
- ✅ `@RateLimiter` with name matching config
- ✅ Protects downstream System API from overload
- ✅ Will throw `RequestNotPermitted` if limit exceeded

---

### Pattern 2: Rate Limiter with Fallback

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class PaymentApplicationService {
    
    private final PaymentApiClient paymentApiClient;
    private final PaymentQueueService queueService;
    
    @RateLimiter(name = "paymentApi", fallbackMethod = "processPaymentFallback")
    public PaymentResult processPayment(PaymentRequest request) {
        log.debug("Processing payment: {}", request.getOrderId());
        return paymentApiClient.charge(request);
    }
    
    /**
     * Fallback when rate limit exceeded.
     * Queue for later processing instead of failing.
     */
    private PaymentResult processPaymentFallback(PaymentRequest request, 
                                                  RequestNotPermitted ex) {
        log.warn("Rate limit exceeded for payment: {}. Queueing for later.", 
                 request.getOrderId());
        
        // Queue for async processing
        queueService.enqueue(request);
        
        return PaymentResult.builder()
            .status(PaymentStatus.QUEUED)
            .message("Payment queued for processing")
            .build();
    }
}
```

**Key points:**
- ✅ Fallback receives `RequestNotPermitted` exception
- ✅ Graceful degradation: queue instead of fail
- ✅ User gets immediate feedback

---

### Pattern 3: Rate Limiter Combined with Circuit Breaker

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class InventoryApplicationService {
    
    private final InventorySystemApiClient systemApiClient;
    
    /**
     * Combined protection:
     * 1. RateLimiter - Prevent overwhelming the System API
     * 2. CircuitBreaker - Stop calls when System API is failing
     * 
     * Order: RateLimiter (outer) -> CircuitBreaker (inner) -> Actual call
     */
    @RateLimiter(name = "inventoryService")
    @CircuitBreaker(name = "inventoryService", fallbackMethod = "checkStockFallback")
    public StockLevel checkStock(String productId) {
        log.debug("Checking stock for product: {}", productId);
        return systemApiClient.getStockLevel(productId);
    }
    
    private StockLevel checkStockFallback(String productId, Exception ex) {
        log.warn("Inventory service unavailable for product: {}. Error: {}", 
                 productId, ex.getClass().getSimpleName());
        return StockLevel.UNKNOWN;
    }
}
```

**Annotation Order:**
- `@RateLimiter` → `@CircuitBreaker` → Actual method
- Rate limiting happens BEFORE circuit breaker evaluation
- Prevents counting rate-limited requests as failures

---

### Pattern 4: Per-User Rate Limiting (Advanced)

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class ApiGatewayService {
    
    private final RateLimiterRegistry rateLimiterRegistry;
    private final BackendClient backendClient;
    
    /**
     * Per-user rate limiting using programmatic approach.
     */
    public ApiResponse processRequest(String userId, ApiRequest request) {
        // Get or create rate limiter for this user
        RateLimiter rateLimiter = rateLimiterRegistry.rateLimiter(
            "user-" + userId,
            RateLimiterConfig.custom()
                .limitForPeriod(100)
                .limitRefreshPeriod(Duration.ofMinutes(1))
                .timeoutDuration(Duration.ZERO)
                .build()
        );
        
        try {
            return RateLimiter.decorateSupplier(rateLimiter, 
                () -> backendClient.process(request)
            ).get();
        } catch (RequestNotPermitted ex) {
            log.warn("Rate limit exceeded for user: {}", userId);
            throw new TooManyRequestsException("Rate limit exceeded", ex);
        }
    }
}
```

---

## Handling Rate Limit Exceeded

### Controller Exception Handler

```java
@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {
    
    @ExceptionHandler(RequestNotPermitted.class)
    public ResponseEntity<ErrorResponse> handleRateLimitExceeded(
            RequestNotPermitted ex, HttpServletRequest request) {
        
        log.warn("Rate limit exceeded: {}", request.getRequestURI());
        
        ErrorResponse error = ErrorResponse.builder()
            .status(HttpStatus.TOO_MANY_REQUESTS.value())
            .error("Too Many Requests")
            .message("Rate limit exceeded. Please try again later.")
            .path(request.getRequestURI())
            .build();
        
        return ResponseEntity
            .status(HttpStatus.TOO_MANY_REQUESTS)
            .header("Retry-After", "1")  // Suggest retry in 1 second
            .body(error);
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
    void getCustomer_withinRateLimit_succeeds() {
        // Arrange
        Customer expected = new Customer("cust-123", "John Doe");
        when(systemApiClient.findById("cust-123")).thenReturn(expected);
        
        // Act
        Customer result = service.getCustomer("cust-123");
        
        // Assert
        assertEquals(expected, result);
    }
    
    @Test
    void getCustomer_exceedsRateLimit_throwsException() {
        // Arrange
        when(systemApiClient.findById(anyString()))
            .thenReturn(new Customer("cust", "Test"));
        
        // Act - Call many times to exceed limit
        // Note: This test requires integration test setup with actual rate limiter
        
        // Assert
        // assertThrows(RequestNotPermitted.class, ...)
    }
}

// Integration test with actual rate limiter
@SpringBootTest
class CustomerApplicationServiceIntegrationTest {
    
    @Autowired
    private CustomerApplicationService service;
    
    @Test
    void rateLimiter_exceedsLimit_throwsException() {
        // Call 51 times (limit is 50)
        for (int i = 0; i < 50; i++) {
            service.getCustomer("cust-" + i);
        }
        
        // 51st call should be rate limited
        assertThrows(RequestNotPermitted.class, 
            () -> service.getCustomer("cust-51"));
    }
}
```

---

## Monitoring and Observability

### Health Endpoint

Rate limiter state exposed via `/actuator/health`:

```json
{
  "status": "UP",
  "components": {
    "rateLimiters": {
      "status": "UP",
      "details": {
        "systemApiClient": {
          "availablePermissions": 45,
          "numberOfWaitingThreads": 0
        }
      }
    }
  }
}
```

### Metrics Endpoint

```bash
curl http://localhost:8080/actuator/metrics/resilience4j.ratelimiter.available.permissions
curl http://localhost:8080/actuator/metrics/resilience4j.ratelimiter.waiting.threads
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `resilience4j_ratelimiter_available_permissions` | Available permits | < 10% of limit |
| `resilience4j_ratelimiter_waiting_threads` | Threads waiting | > 0 sustained |
| `resilience4j_ratelimiter_calls{kind="failed"}` | Rate-limited calls | > 1% of total |

---

## Best Practices

### ✅ DO

1. **Set rate limits based on downstream capacity** not arbitrary numbers
2. **Use timeoutDuration=0** for fail-fast behavior
3. **Provide meaningful fallbacks** (queue, cache, default response)
4. **Return HTTP 429** with `Retry-After` header for APIs
5. **Monitor available permissions** to detect capacity issues
6. **Combine with Circuit Breaker** for comprehensive protection
7. **Document rate limits** for API consumers

### ❌ DON'T

1. **Don't set very high limits** that don't protect downstream
2. **Don't use long timeoutDuration** for user-facing requests
3. **Don't rate limit internal health checks**
4. **Don't forget to handle RequestNotPermitted** in controllers
5. **Don't use same rate limiter** for different downstream services
6. **Don't rely solely on rate limiting** for resilience

---

## Common Pitfalls

### Pitfall 1: Wrong Limit Calculation

```java
// ❌ WRONG - Limit per pod, not total
// If you have 10 pods with limit=100, actual limit = 1000
resilience4j:
  ratelimiter:
    instances:
      systemApi:
        limitForPeriod: 100  # Per pod!

// ✅ CORRECT - Calculate per-pod limit
// If System API handles 500 TPS and you have 10 pods: 500/10 = 50 per pod
resilience4j:
  ratelimiter:
    instances:
      systemApi:
        limitForPeriod: 50
```

### Pitfall 2: Ignoring RequestNotPermitted

```java
// ❌ WRONG - No handling, user gets ugly error
@RateLimiter(name = "api")
public Result process() { }

// ✅ CORRECT - Proper handling with fallback or exception handler
@RateLimiter(name = "api", fallbackMethod = "processFallback")
public Result process() { }

private Result processFallback(RequestNotPermitted ex) {
    return Result.rateLimited();
}
```

### Pitfall 3: Rate Limiter Inside Retry

```java
// ❌ WRONG - Each retry consumes rate limit
@Retry(name = "service")
@RateLimiter(name = "service")
public Result doSomething() { }
// 3 retries = 3 rate limit permits consumed

// ✅ CORRECT - Rate limiter outside retry
@RateLimiter(name = "service")
@Retry(name = "service")
public Result doSomething() { }
// 1 rate limit permit regardless of retries
```

---

## Related Patterns

| Pattern | Combination | Benefit |
|---------|-------------|---------|
| **Circuit Breaker** | `@RateLimiter` + `@CircuitBreaker` | Rate limit + failure tracking |
| **Bulkhead** | `@RateLimiter` + `@Bulkhead` | Rate limit + concurrency limit |
| **Retry** | `@RateLimiter` + `@Retry` | Rate limit protects during retries |

**Recommended order:**
```java
@RateLimiter(name = "systemApi")     // Outer - prevents flood
@CircuitBreaker(name = "systemApi")  // Middle - tracks failures
@Retry(name = "systemApi")           // Inner - retries transient failures
public Result callSystemApi() { }
```

---

## References

### Documentation
- [Resilience4j RateLimiter Docs](https://resilience4j.readme.io/docs/ratelimiter)
- [Spring Boot Integration](https://resilience4j.readme.io/docs/getting-started-3)

### Related
- **Implements:** ADR-004 (Resilience Patterns)
- **Module:** mod-code-004-rate-limiter-java-resilience4j
- **Complements:** ERI-008 (Circuit Breaker), ERI-009 (Retry), ERI-010 (Timeout)

---

## Changelog

### v1.0 (2025-11-28)
- Initial version
- Complete RateLimiter implementation for Java/Spring Boot
- Multiple patterns: basic, with fallback, combined with circuit breaker
- Per-user rate limiting example
- Unit test examples
- Best practices and common pitfalls

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-011-rate-limiter-constraints
  version: "1.0"
  eri_reference: eri-code-011-rate-limiter-java-resilience4j
  adr_reference: adr-004-resilience-patterns
  
  structural_constraints:
    # Annotation Placement
    - id: annotation-in-application-layer
      rule: "@RateLimiter annotation MUST be in application layer only (not in domain)"
      validation: "grep -r '@RateLimiter' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: application
      
    - id: annotation-not-in-controller
      rule: "@RateLimiter SHOULD NOT be directly on controller methods"
      validation: "grep -r '@RateLimiter' src/*/java/**/adapter/rest/controller/ returns empty"
      severity: WARNING
      layer: adapter
      
    # Naming
    - id: ratelimiter-name-matches-config
      rule: "RateLimiter name MUST match a configured instance in application.yml"
      validation: "Name parameter value exists in resilience4j.ratelimiter.instances"
      severity: ERROR
      
    # Combination Order
    - id: ratelimiter-before-circuitbreaker
      rule: "When combined, @RateLimiter SHOULD be declared before @CircuitBreaker"
      validation: "@RateLimiter annotation appears before @CircuitBreaker"
      severity: WARNING
      
    - id: ratelimiter-before-retry
      rule: "When combined, @RateLimiter MUST be declared before @Retry"
      validation: "@RateLimiter annotation appears before @Retry"
      severity: ERROR

  configuration_constraints:
    - id: resilience4j-ratelimiter-config-exists
      rule: "resilience4j.ratelimiter section MUST exist in application.yml"
      validation: "application.yml contains resilience4j.ratelimiter key"
      severity: ERROR
      
    - id: limit-for-period-configured
      rule: "limitForPeriod MUST be explicitly configured per instance"
      validation: "resilience4j.ratelimiter.instances.*.limitForPeriod is defined"
      severity: ERROR
      
    - id: limit-based-on-downstream
      rule: "Rate limit SHOULD be based on downstream service capacity"
      validation: "Manual verification required"
      severity: WARNING
      
    - id: timeout-duration-appropriate
      rule: "timeoutDuration SHOULD be 0 for fail-fast or reasonable wait time"
      validation: "timeoutDuration is 0 or between 100ms and 5s"
      severity: WARNING

  dependency_constraints:
    required:
      - groupId: io.github.resilience4j
        artifactId: resilience4j-spring-boot3
        minVersion: "2.0.0"
        reason: "RateLimiter implementation for Spring Boot 3.x"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-aop
        reason: "Required for @RateLimiter annotation processing"
        
    optional:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-actuator
        reason: "Required for rate limiter health and metrics endpoints"

  testing_constraints:
    - id: within-limit-tested
      rule: "Within rate limit scenario MUST be tested"
      validation: "Test exists that verifies success within limit"
      severity: ERROR
      
    - id: exceed-limit-tested
      rule: "Exceed rate limit scenario SHOULD be tested"
      validation: "Integration test verifies RequestNotPermitted when limit exceeded"
      severity: WARNING
      
    - id: exception-handling-tested
      rule: "RequestNotPermitted handling MUST be tested if no fallback"
      validation: "Controller test verifies 429 response"
      severity: ERROR
```

---

**Status:** ✅ Production-Ready  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0
