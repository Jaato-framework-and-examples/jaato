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
implements_pattern: circuit-breaker
tags:
  - java
  - spring-boot
  - resilience4j
  - circuit-breaker
  - fault-tolerance
  - microservices
related:
  - eri-code-009-retry-java-resilience4j
  - eri-code-010-bulkhead-java-resilience4j
  - eri-code-011-rate-limiter-java-resilience4j
automated_by:
  - skill-code-001-add-circuit-breaker-java-resilience4j
cross_domain_usage: qa
---

## Overview

This Enterprise Reference Implementation provides the standard way to implement the Circuit Breaker pattern in Java/Spring Boot microservices using Resilience4j.

**What this implements:**
- Circuit breaker for external service calls
- Fallback mechanisms for graceful degradation
- Configuration externalization
- Monitoring and metrics integration

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Spring Boot** | 2.7.x / 3.x | Application framework |
| **Resilience4j** | 2.1.0 | Circuit breaker library |
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
        <artifactId>resilience4j-spring-boot2</artifactId>
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

### Gradle (build.gradle)

```gradle
dependencies {
    implementation 'io.github.resilience4j:resilience4j-spring-boot2:2.1.0'
    implementation 'org.springframework.boot:spring-boot-starter-aop'
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
}
```

---

## Project Structure

```
src/main/java/com/company/{service}/
├── application/
│   └── service/
│       └── {Service}ApplicationService.java    # @CircuitBreaker annotations here
├── adapter/
│   └── client/
│       ├── {External}Client.java               # External service client
│       └── {External}ClientFallback.java       # Fallback implementations
└── infrastructure/
    └── config/
        └── ResilienceConfig.java               # Programmatic configuration (optional)

src/main/resources/
└── application.yml                             # Circuit breaker configuration

src/test/java/com/company/{service}/
└── application/
    └── service/
        └── {Service}ApplicationServiceTest.java  # Unit tests with mocked failures
```

**Key Placement Rules:**
- `@CircuitBreaker` annotations go on **Application Service** methods (not domain)
- Fallback classes are **Adapters** (they handle infrastructure concerns)
- Configuration is externalized in `application.yml`

---

## Configuration

### application.yml

```yaml
resilience4j:
  circuitbreaker:
    configs:
      # Default configuration for all circuit breakers
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
          - java.lang.IllegalStateException
    
    instances:
      # Specific instance for payment service
      paymentService:
        baseConfig: default
        failureRateThreshold: 40
        waitDurationInOpenState: 30000
      
      # Specific instance for customer service
      customerService:
        baseConfig: default
        slidingWindowSize: 50
        minimumNumberOfCalls: 5

# Expose circuit breaker metrics
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,circuitbreakers,circuitbreakerevents
  endpoint:
    health:
      show-details: always
  health:
    circuitbreakers:
      enabled: true
```

### Configuration Parameters Explained

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| **slidingWindowSize** | 100 | Number of calls recorded | 100 for high traffic, 20 for low traffic |
| **minimumNumberOfCalls** | 10 | Min calls before evaluation | 10 minimum, adjust per traffic |
| **failureRateThreshold** | 50 | Failure % to open circuit (0-100) | 50 for external APIs, 40 for critical services |
| **waitDurationInOpenState** | 60000ms | Time before half-open | 30-60s typical, 5-10s for fast recovery |
| **permittedNumberOfCallsInHalfOpenState** | 3 | Test calls in half-open | 3-5 calls sufficient |

---

## Implementation Patterns

### Pattern 1: Basic Circuit Breaker with Fallback

```java
package com.company.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class PaymentService {
    
    private final PaymentApiClient paymentApiClient;
    private final PaymentCacheService cacheService;
    
    public PaymentService(PaymentApiClient paymentApiClient, 
                         PaymentCacheService cacheService) {
        this.paymentApiClient = paymentApiClient;
        this.cacheService = cacheService;
    }
    
    /**
     * Process payment with circuit breaker protection.
     * Falls back to cached response or default error handling.
     */
    @CircuitBreaker(name = "paymentService", fallbackMethod = "processPaymentFallback")
    public PaymentResponse processPayment(PaymentRequest request) {
        log.debug("Processing payment for order: {}", request.getOrderId());
        return paymentApiClient.charge(request);
    }
    
    /**
     * Fallback method - MUST have same signature + Throwable parameter.
     * Called when circuit is OPEN or method throws exception.
     */
    private PaymentResponse processPaymentFallback(PaymentRequest request, 
                                                   Throwable throwable) {
        log.warn("Circuit breaker fallback triggered for order: {}. Reason: {}", 
                 request.getOrderId(), throwable.getMessage());
        
        // Strategy 1: Return cached response if available
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
- ✅ Logging for observability
- ✅ Graceful degradation (cache or meaningful error)

---

### Pattern 2: Circuit Breaker with Multiple Fallback Strategies

```java
@Service
@Slf4j
public class CustomerService {
    
    private final CustomerApiClient apiClient;
    private final CustomerCacheService cacheService;
    
    @CircuitBreaker(name = "customerService", fallbackMethod = "findCustomerFallback")
    public Optional<Customer> findCustomer(String customerId) {
        return apiClient.getCustomer(customerId);
    }
    
    /**
     * Multi-level fallback:
     * 1. Try cache
     * 2. Return empty Optional (acceptable degradation)
     */
    private Optional<Customer> findCustomerFallback(String customerId, Throwable throwable) {
        log.warn("Customer service unavailable, trying cache for: {}", customerId);
        
        // Level 1: Try cache
        Optional<Customer> cached = cacheService.get(customerId);
        if (cached.isPresent()) {
            log.info("Returning cached customer: {}", customerId);
            return cached;
        }
        
        // Level 2: Return empty (acceptable for non-critical reads)
        log.warn("No cached data for customer: {}, returning empty", customerId);
        return Optional.empty();
    }
}
```

---

### Pattern 3: Circuit Breaker Without Fallback (Fail Fast)

```java
@Service
@Slf4j
public class NotificationService {
    
    private final EmailApiClient emailClient;
    
    /**
     * Circuit breaker without fallback - fails fast.
     * Use when no acceptable degradation exists.
     */
    @CircuitBreaker(name = "emailService")
    public void sendEmail(EmailRequest email) {
        emailClient.send(email);
        // If fails, exception propagates to caller
        // Circuit prevents repeated calls to failing service
    }
}
```

**When to use:**
- ✅ When no fallback makes sense
- ✅ Non-critical operations that can simply fail
- ✅ When caller should handle failure differently

---

### Pattern 4: Programmatic Circuit Breaker (without annotation)

```java
@Service
public class OrderService {
    
    private final CircuitBreakerRegistry circuitBreakerRegistry;
    private final OrderApiClient apiClient;
    
    public OrderService(CircuitBreakerRegistry circuitBreakerRegistry, 
                       OrderApiClient apiClient) {
        this.circuitBreakerRegistry = circuitBreakerRegistry;
        this.apiClient = apiClient;
    }
    
    public Order createOrder(OrderRequest request) {
        CircuitBreaker circuitBreaker = circuitBreakerRegistry
            .circuitBreaker("orderService");
        
        return circuitBreaker.executeSupplier(() -> {
            return apiClient.createOrder(request);
        });
    }
}
```

**When to use:**
- When you need more control
- Dynamic circuit breaker names
- Complex fallback logic
- Integration with reactive programming

---

## Testing

### Unit Test Example

```java
@ExtendWith(MockitoExtension.class)
class PaymentServiceTest {
    
    @Mock
    private PaymentApiClient apiClient;
    
    @Mock
    private PaymentCacheService cacheService;
    
    @InjectMocks
    private PaymentService paymentService;
    
    @Test
    void processPayment_success() {
        // Arrange
        PaymentRequest request = new PaymentRequest("order-123", 100.0);
        PaymentResponse expected = new PaymentResponse("txn-456", "SUCCESS");
        when(apiClient.charge(request)).thenReturn(expected);
        
        // Act
        PaymentResponse result = paymentService.processPayment(request);
        
        // Assert
        assertEquals(expected, result);
        verify(apiClient).charge(request);
    }
    
    @Test
    void processPayment_fallbackTriggered() {
        // Arrange
        PaymentRequest request = new PaymentRequest("order-123", 100.0);
        when(apiClient.charge(request)).thenThrow(new ConnectException("Connection refused"));
        
        PaymentResponse cached = new PaymentResponse("cached-txn", "SUCCESS");
        when(cacheService.getLastSuccessfulPayment("order-123"))
            .thenReturn(Optional.of(cached));
        
        // Act
        PaymentResponse result = paymentService.processPayment(request);
        
        // Assert
        assertEquals(cached, result);
        verify(cacheService).getLastSuccessfulPayment("order-123");
    }
}
```

---

## Monitoring and Observability

### Health Endpoint

Circuit breaker state exposed via `/actuator/health`:

```json
{
  "status": "UP",
  "components": {
    "circuitBreakers": {
      "status": "UP",
      "details": {
        "paymentService": {
          "status": "UP",
          "details": {
            "state": "CLOSED",
            "failureRate": "15.0%",
            "slowCallRate": "0.0%",
            "bufferedCalls": 20,
            "failedCalls": 3
          }
        }
      }
    }
  }
}
```

### Metrics Endpoint

Circuit breaker metrics via `/actuator/metrics`:

```bash
# Query metrics
curl http://localhost:8080/actuator/metrics/resilience4j.circuitbreaker.calls

# Sample response
{
  "name": "resilience4j.circuitbreaker.calls",
  "measurements": [
    {"statistic": "COUNT", "value": 1250.0}
  ],
  "availableTags": [
    {"tag": "name", "values": ["paymentService", "customerService"]},
    {"tag": "kind", "values": ["successful", "failed", "not_permitted"]}
  ]
}
```

### Prometheus Integration

```yaml
# application.yml
management:
  metrics:
    export:
      prometheus:
        enabled: true
```

**Key metrics to monitor:**
- `resilience4j_circuitbreaker_state` - Current state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
- `resilience4j_circuitbreaker_calls_total` - Total calls by kind
- `resilience4j_circuitbreaker_failure_rate` - Current failure rate

---

## Best Practices

### ✅ DO

1. **Always provide meaningful fallbacks** for user-facing operations
2. **Log fallback triggers** with context for troubleshooting
3. **Use cached data** when acceptable as fallback strategy
4. **Configure per service** based on SLA and criticality
5. **Monitor circuit breaker states** in production
6. **Test fallback paths** explicitly in unit and integration tests
7. **Document fallback behavior** in API documentation

### ❌ DON'T

1. **Don't ignore the Throwable** in fallback method
2. **Don't put business logic** in fallback methods
3. **Don't use circuit breaker** for business rule failures
4. **Don't set thresholds too low** (causes false positives)
5. **Don't forget to test** with actual failures in staging
6. **Don't return null** - use Optional or throw meaningful exception
7. **Don't apply to local methods** - only external dependencies

---

## Common Pitfalls

### Pitfall 1: Fallback Signature Mismatch

```java
// ❌ WRONG - Missing Throwable parameter
@CircuitBreaker(name = "service", fallbackMethod = "fallback")
public String doSomething(String param) { }

private String fallback(String param) { } // WILL NOT WORK!

// ✅ CORRECT - Throwable parameter added
private String fallback(String param, Throwable throwable) { }
```

### Pitfall 2: Ignoring Configuration

```java
// ❌ WRONG - Using non-existent circuit breaker name
@CircuitBreaker(name = "nonExistentService", ...)
// Will use default config, which may not be appropriate
```

### Pitfall 3: Fallback Throws Exception

```java
// ⚠️ RISKY - Fallback itself throws exception
private Response fallback(Request req, Throwable throwable) {
    // If this throws, no further fallback exists
    return riskyOperation(); // May throw!
}

// ✅ BETTER - Defensive fallback
private Response fallback(Request req, Throwable throwable) {
    try {
        return riskyOperation();
    } catch (Exception e) {
        log.error("Even fallback failed", e);
        return Response.error("Service unavailable");
    }
}
```

---

## Related Patterns

This ERI should be combined with other resilience patterns:

| Pattern | Combination | Benefit |
|---------|-------------|---------|
| **Retry** | Circuit Breaker + Retry | Retry transient failures, circuit breaker for persistent failures |
| **Timeout** | Circuit Breaker + Timeout | Timeout prevents hanging, circuit breaker tracks pattern |
| **Bulkhead** | Bulkhead + Circuit Breaker | Isolate failing services, prevent resource exhaustion |
| **Rate Limiter** | Rate Limiter + Circuit Breaker | Control outbound traffic, protect from our overload |

**See:** ERI-009 (Retry), ERI-010 (Bulkhead), ERI-011 (Rate Limiter), ERI-012 (Timeout)

---

## Migration from Hystrix

If migrating from Netflix Hystrix:

| Hystrix | Resilience4j Equivalent |
|---------|------------------------|
| `@HystrixCommand` | `@CircuitBreaker` |
| `fallbackMethod` | `fallbackMethod` (same) |
| `commandKey` | `name` |
| Thread pool isolation | Use `@Bulkhead` separately |

**Migration guide:** [Resilience4j Migration Docs](https://resilience4j.readme.io/docs/migration-guide)

---

## Automated Application

This ERI can be applied automatically using:

**Skill:** skill-code-001-add-circuit-breaker-java-resilience4j
- Analyzes existing code
- Adds @CircuitBreaker annotation
- Generates fallback method
- Updates configuration
- Creates tests

**Usage:**
```bash
# Via MCP/ADK
apply-skill skill-code-001-add-circuit-breaker-java-resilience4j \
  --target-class com.company.service.PaymentService \
  --target-method processPayment
```

---

## References

### Documentation
- [Resilience4j Official Docs](https://resilience4j.readme.io/)
- [Spring Boot Integration](https://resilience4j.readme.io/docs/getting-started-3)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

### Related
- **Implements:** ADR-004 (Resilience Patterns)
- **Automated by:** skill-code-001-add-circuit-breaker-java-resilience4j
- **Complements:** ERI-009 (Retry), ERI-010 (Bulkhead), ERI-012 (Timeout)

---

## Changelog

### v2.1 (2025-11-27)
- Added domain prefix to ID (eri-code-008)
- Added cross_domain_usage metadata
- Updated front matter format

### v2.0 (2025-11-20)
- Renamed to ERI-008-java-resilience4j (framework-specific)
- Added framework and library version details
- Added multiple implementation patterns
- Enhanced monitoring and observability section
- Added best practices and common pitfalls
- Added migration guide from Hystrix

### v1.0 (2024-05-15)
- Initial version as ERI-008

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-008-circuit-breaker-constraints
  version: "1.0"
  eri_reference: eri-code-008-circuit-breaker-java-resilience4j
  adr_reference: adr-004-resilience-patterns
  
  structural_constraints:
    # Annotation Placement
    - id: annotation-in-application-layer
      rule: "@CircuitBreaker annotation MUST be in application layer only (not in domain)"
      validation: "grep -r '@CircuitBreaker' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: application
      
    - id: annotation-not-in-controller
      rule: "@CircuitBreaker SHOULD NOT be directly on controller methods"
      validation: "grep -r '@CircuitBreaker' src/*/java/**/adapter/rest/controller/ returns empty"
      severity: WARNING
      layer: adapter
      
    # Fallback Requirements
    - id: fallback-method-required
      rule: "Every @CircuitBreaker annotation MUST have fallbackMethod defined"
      validation: "All @CircuitBreaker annotations include fallbackMethod parameter"
      severity: ERROR
      
    - id: fallback-signature-throwable
      rule: "Fallback method MUST have Throwable as last parameter"
      validation: "Fallback methods signature includes Throwable parameter at the end"
      severity: ERROR
      
    - id: fallback-same-return-type
      rule: "Fallback method MUST have same return type as protected method"
      validation: "Fallback method return type matches annotated method return type"
      severity: ERROR
      
    - id: fallback-in-same-class
      rule: "Fallback method MUST be in same class as @CircuitBreaker method"
      validation: "Fallback method exists in same class file"
      severity: ERROR
      
    # Naming
    - id: circuit-breaker-name-matches-config
      rule: "Circuit breaker name MUST match a configured instance in application.yml"
      validation: "Name parameter value exists in resilience4j.circuitbreaker.instances"
      severity: ERROR
      
    - id: circuit-breaker-naming-convention
      rule: "Circuit breaker name SHOULD follow camelCase convention matching service name"
      validation: "Name follows pattern: {serviceName}Service or {serviceName}Client"
      severity: WARNING

  configuration_constraints:
    - id: resilience4j-config-exists
      rule: "resilience4j.circuitbreaker section MUST exist in application.yml"
      validation: "application.yml contains resilience4j.circuitbreaker key"
      severity: ERROR
      
    - id: default-config-defined
      rule: "A default circuit breaker configuration SHOULD be defined"
      validation: "resilience4j.circuitbreaker.configs.default exists"
      severity: WARNING
      
    - id: actuator-endpoints-exposed
      rule: "Circuit breaker actuator endpoints MUST be exposed for monitoring"
      validation: "management.endpoints.web.exposure.include contains 'circuitbreakers' or '*'"
      severity: ERROR
      
    - id: health-indicator-enabled
      rule: "Circuit breaker health indicator SHOULD be enabled"
      validation: "management.health.circuitbreakers.enabled is true"
      severity: WARNING
      
    - id: failure-rate-threshold-range
      rule: "Failure rate threshold SHOULD be between 25-75%"
      validation: "failureRateThreshold >= 25 AND <= 75"
      severity: WARNING
      
    - id: sliding-window-size-minimum
      rule: "Sliding window size SHOULD be at least 10 for meaningful statistics"
      validation: "slidingWindowSize >= 10"
      severity: WARNING
      
    - id: wait-duration-reasonable
      rule: "Wait duration in open state SHOULD be between 10s and 120s"
      validation: "waitDurationInOpenState >= 10000 AND <= 120000"
      severity: WARNING

  dependency_constraints:
    required:
      - groupId: io.github.resilience4j
        artifactId: resilience4j-spring-boot3
        minVersion: "2.0.0"
        reason: "Circuit breaker implementation for Spring Boot 3.x"
        alternativeFor: "resilience4j-spring-boot2 for Spring Boot 2.x"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-aop
        reason: "Required for @CircuitBreaker annotation processing"
        
    optional:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-actuator
        reason: "Required for circuit breaker health and metrics endpoints"
        
      - groupId: io.micrometer
        artifactId: micrometer-registry-prometheus
        reason: "Export circuit breaker metrics to Prometheus"

  testing_constraints:
    - id: fallback-path-tested
      rule: "Fallback execution path MUST be tested"
      validation: "Test exists that verifies fallback is called when exception thrown"
      severity: ERROR
      
    - id: circuit-states-tested
      rule: "Circuit breaker state transitions SHOULD be tested"
      validation: "Tests verify CLOSED -> OPEN -> HALF_OPEN transitions"
      severity: WARNING
      
    - id: mock-external-failures
      rule: "Tests MUST mock external service failures to trigger circuit breaker"
      validation: "Test uses when().thenThrow() or similar to simulate failures"
      severity: ERROR
      
    - id: test-class-exists
      rule: "Unit test class MUST exist for service with @CircuitBreaker"
      validation: "Test class exists matching *ServiceTest.java pattern"
      severity: ERROR
```

---

**Status:** ✅ Production-Ready  
**Framework:** Java/Spring Boot  
**Library:** Resilience4j 2.1.0
