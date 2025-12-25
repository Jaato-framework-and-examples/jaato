---
id: mod-code-001-circuit-breaker-java-resilience4j
title: "MOD-001: Circuit Breaker - Java/Resilience4j"
version: 2.1
date: 2025-11-21
updated: 2025-11-24
status: Active
derived_from: eri-code-008-circuit-breaker-java-resilience4j
domain: code
tags:
  - java
  - resilience4j
  - circuit-breaker
  - fault-tolerance
used_by:
  - skill-code-001-add-circuit-breaker-java-resilience4j
  - skill-code-020-generate-microservice-java-spring
---

# MOD-001: Circuit Breaker - Java/Resilience4j

**Module ID:** mod-code-001-circuit-breaker-java-resilience4j  
**Version:** 2.1  
**Date:** 2025-11-21  
**Updated:** 2025-11-24  
**Source ERI:** eri-code-008-circuit-breaker-java-resilience4j  
**Framework:** Java 17+ / Spring Boot 3.x  
**Library:** Resilience4j 2.1.0  
**Used by:** skill-code-001-add-circuit-breaker-java-resilience4j, skill-code-020-generate-microservice-java-spring

---

## Purpose

Provides reusable code templates for implementing circuit breaker patterns in Java/Spring Boot applications using Resilience4j. Templates use `{{placeholder}}` variables that are replaced dynamically during code transformation or generation.

**Use this module when:**
- Adding fault tolerance to external service calls
- Implementing graceful degradation
- Preventing cascade failures

**Do NOT use when:**
- Internal method calls (no external dependency)
- Database operations (use retry instead)
- Message queue consumers (different pattern)

---

## Template Variables

### Common Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{circuitBreakerName}}` | Unique circuit breaker identifier | `paymentServiceCB` |
| `{{methodName}}` | Method being protected | `processPayment` |
| `{{returnType}}` | Method return type | `PaymentResult` |
| `{{methodParameters}}` | Method parameters with types | `String orderId, BigDecimal amount` |
| `{{originalMethodBody}}` | Original method implementation | `return client.call(orderId);` |

### Fallback Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{fallbackMethodName}}` | Fallback method name | `processPaymentFallback` |
| `{{fallbackLogic}}` | Return statement in fallback | `return PaymentResult.failed("unavailable");` |
| `{{fallbackStrategy}}` | Type: empty_object, failed_object, cached, default_value | `failed_object` |

### Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `{{failureRateThreshold}}` | % failures to open circuit | `50` |
| `{{waitDurationInOpenState}}` | Seconds before half-open | `30` |
| `{{slidingWindowSize}}` | Number of calls to evaluate | `100` |
| `{{minimumNumberOfCalls}}` | Min calls before evaluation | `10` |

### Chain Fallback Variables (Template 2)

| Variable | Description | Example |
|----------|-------------|---------|
| `{{primaryFallbackName}}` | First fallback method | `tryAlternativeService` |
| `{{secondaryFallbackName}}` | Second fallback method | `useCachedData` |
| `{{tertiaryFallbackName}}` | Final fallback method | `returnDefault` |
| `{{alternativeServiceCall}}` | Backup service call | `backupClient.call(id)` |
| `{{cacheKey}}` | Cache lookup key | `payment-{orderId}` |
| `{{defaultValue}}` | Final fallback value | `PaymentResult.unavailable()` |

---

## Templates

### Template 1: Basic with Single Fallback

**Use case:** Most common scenario (80% of cases). Single fallback to graceful degradation.

#### Code

```java
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Add to protected method
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{fallbackMethodName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}

// Add fallback method (MUST have same params + Throwable)
private {{returnType}} {{fallbackMethodName}}({{methodParameters}}, Throwable throwable) {
    log.warn("Circuit breaker fallback for {}: {}", "{{methodName}}", throwable.getMessage());
    {{fallbackLogic}}
}
```

#### Fallback Strategy Options

```java
// Strategy: empty_object
return Optional.empty();
return Collections.emptyList();
return new {{returnType}}();

// Strategy: failed_object
return {{returnType}}.failed("Service unavailable");
return {{returnType}}.error(throwable.getMessage());

// Strategy: cached
return cache.get("{{cacheKey}}").orElse(defaultValue);

// Strategy: default_value
return 0;        // for int/long
return false;    // for boolean
return "";       // for String
```

#### Configuration

```yaml
resilience4j:
  circuitbreaker:
    instances:
      {{circuitBreakerName}}:
        failure-rate-threshold: {{failureRateThreshold}}
        wait-duration-in-open-state: {{waitDurationInOpenState}}s
        sliding-window-size: {{slidingWindowSize}}
        minimum-number-of-calls: {{minimumNumberOfCalls}}
        permitted-number-of-calls-in-half-open-state: 10
        automatic-transition-from-open-to-half-open-enabled: true
```

---

### Template 2: Multiple Fallbacks Chain

**Use case:** High availability requirements. Tiered fallback with alternative service → cache → default.

#### Code

```java
@CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "{{primaryFallbackName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) {
    {{originalMethodBody}}
}

// Level 1: Try alternative service
private {{returnType}} {{primaryFallbackName}}({{methodParameters}}, Throwable throwable) {
    log.warn("Primary service failed, trying alternative: {}", throwable.getMessage());
    try {
        return {{alternativeServiceCall}};
    } catch (Exception e) {
        return {{secondaryFallbackName}}({{parameterNames}}, e);
    }
}

// Level 2: Use cached data
private {{returnType}} {{secondaryFallbackName}}({{methodParameters}}, Throwable throwable) {
    log.warn("Alternative service failed, using cached data: {}", throwable.getMessage());
    return cache.get("{{cacheKey}}")
        .orElseGet(() -> {{tertiaryFallbackName}}({{parameterNames}}, throwable));
}

// Level 3: Return default
private {{returnType}} {{tertiaryFallbackName}}({{methodParameters}}, Throwable throwable) {
    log.error("All fallbacks exhausted, returning default: {}", throwable.getMessage());
    return {{defaultValue}};
}
```

---

### Template 3: Fail Fast (No Fallback)

**Use case:** Non-critical operations where failure is acceptable. Metrics, analytics, logging.

#### Code

```java
@CircuitBreaker(name = "{{circuitBreakerName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) throws {{exceptionType}} {
    {{originalMethodBody}}
}
```

**Note:** No `fallbackMethod` parameter. Circuit throws `CallNotPermittedException` when open.

#### Caller Handling

```java
try {
    {{returnType}} result = service.{{methodName}}({{arguments}});
    // Process result
} catch (CallNotPermittedException e) {
    log.warn("Circuit breaker open, skipping operation: {}", e.getMessage());
    // Graceful handling - operation skipped
}
```

---

### Template 4: Programmatic (No Annotations)

**Use case:** Dynamic configuration, testing, conditional circuit breaker application.

#### Code

```java
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;

@Service
public class {{serviceName}} {
    
    private static final Logger log = LoggerFactory.getLogger({{serviceName}}.class);
    private final CircuitBreaker circuitBreaker;
    private final {{clientType}} client;
    
    public {{serviceName}}(CircuitBreakerRegistry registry, {{clientType}} client) {
        this.circuitBreaker = registry.circuitBreaker("{{circuitBreakerName}}");
        this.client = client;
    }
    
    public {{returnType}} {{methodName}}({{methodParameters}}) {
        return circuitBreaker.executeSupplier(() -> {
            {{originalMethodBody}}
        });
    }
    
    // With fallback
    public {{returnType}} {{methodName}}WithFallback({{methodParameters}}) {
        try {
            return circuitBreaker.executeSupplier(() -> {
                {{originalMethodBody}}
            });
        } catch (Exception e) {
            log.warn("Circuit breaker fallback triggered: {}", e.getMessage());
            return {{fallbackLogic}};
        }
    }
}
```

---

## Dependencies

### Maven (pom.xml)

```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Gradle (build.gradle)

```groovy
implementation 'io.github.resilience4j:resilience4j-spring-boot3:2.1.0'
```

---

## Testing Template

```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {{serviceName}}Test {
    
    @Mock
    private {{clientType}} client;
    
    @InjectMocks
    private {{serviceName}} service;
    
    @Test
    void {{methodName}}_WhenServiceAvailable_ReturnsResult() {
        // Given
        when(client.call(any())).thenReturn(expectedResult);
        
        // When
        {{returnType}} result = service.{{methodName}}(params);
        
        // Then
        assertThat(result).isEqualTo(expectedResult);
    }
    
    @Test
    void {{methodName}}_WhenServiceUnavailable_ReturnsFallback() {
        // Given
        when(client.call(any())).thenThrow(new RuntimeException("Service down"));
        
        // When
        {{returnType}} result = service.{{methodName}}(params);
        
        // Then
        assertThat(result).isNotNull();
        // Verify fallback behavior based on strategy
    }
}
```

---

## Best Practices

1. **Always log fallback execution** - Essential for debugging and monitoring
2. **Use meaningful circuit breaker names** - Include service + operation, e.g., `paymentService-processPayment`
3. **Configure realistic thresholds** - Based on actual SLAs and traffic patterns
4. **Test circuit breaker behavior** - Include tests for open/closed/half-open states
5. **Monitor circuit breaker metrics** - Expose to Prometheus/Grafana
6. **Keep fallback logic simple** - Fallback should not throw exceptions
7. **Use Optional for nullable returns** - Prefer `Optional.empty()` over `null`
8. **Consider timeout with circuit breaker** - Combine patterns for better resilience

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Missing Throwable parameter | Fallback not called | Add `Throwable throwable` as last parameter |
| Fallback throws exception | Cascade failure | Catch all exceptions in fallback |
| No logging in fallback | Silent failures | Always log fallback execution |
| Circular fallback calls | Stack overflow | Chain fallbacks linearly |
| Same name for multiple circuits | Shared state | Use unique names per circuit |
| Wrong method visibility | Fallback not found | Fallback must be same or less restrictive visibility |
| Different return type | Compilation error | Fallback must return same type |

---

## Usage Notes

1. **Variable replacement:** All `{{variable}}` placeholders are replaced by skills during code transformation
2. **Import statements:** Add required imports to the class
3. **Configuration merge:** Merge YAML configuration with existing application.yml
4. **Testing:** Generated tests use Mockito, no Spring context needed for unit tests

---

## Related

- **Source ERI:** eri-code-008-circuit-breaker-java-resilience4j
- **Used by Skills:** 
  - skill-code-001-add-circuit-breaker-java-resilience4j (TRANSFORMATION)
  - skill-code-020-generate-microservice-java-spring (CREATION)
- **Complements:** 
  - mod-code-002-retry-java-resilience4j (future)
  - mod-code-003-timeout-java-resilience4j (future)
- **Feature:** resilience.circuit_breaker

---

**Module Version:** 2.1  
**Last Updated:** 2025-11-24
