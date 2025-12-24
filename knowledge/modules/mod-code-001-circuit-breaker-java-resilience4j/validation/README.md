# Circuit Breaker Validation

## Purpose

Validates that Circuit Breaker pattern is correctly implemented using Resilience4j library.

## Module

**Module:** mod-code-001-circuit-breaker-java-resilience4j  
**Pattern:** Circuit Breaker (Resilience)  
**Library:** Resilience4j  
**Implements:** ADR-004 (Resilience Patterns)

## What This Validates

### 1. Code Annotations
- ✅ `@CircuitBreaker` annotation present in code
- ✅ Annotation is in **application layer** (not domain)
- ✅ Fallback methods defined
- ✅ Fallback methods implemented (not just declared)

### 2. Configuration
- ✅ `resilience4j:` configuration in application.yml
- ✅ Circuit breaker instances configured
- ✅ Actuator endpoints expose circuit breaker metrics

### 3. Dependencies
- ✅ `resilience4j-spring-boot` dependency in pom.xml
- ⚠️ `spring-boot-starter-aop` dependency (required for AOP)

## Usage

```bash
./circuit-breaker-check.sh /path/to/service
```

## Checks Performed

| Check | Type | Description |
|-------|------|-------------|
| @CircuitBreaker present | CRITICAL | Annotation exists in code |
| Fallback methods defined | CRITICAL | fallbackMethod attribute set |
| Fallback methods implemented | CRITICAL | Methods actually exist |
| Resilience4j config | CRITICAL | application.yml has config |
| Circuit breaker instances | WARN | Instances explicitly configured |
| Dependencies | CRITICAL | pom.xml has resilience4j |
| AOP dependency | WARN | spring-boot-starter-aop present |
| Actuator metrics | WARN | /actuator/circuitbreakers exposed |
| Layer placement | CRITICAL | @CircuitBreaker in application, not domain |

## Exit Codes

- `0`: All critical checks passed
- `1`: One or more critical checks failed

Warnings do not cause failure.

## Example Output

**Success:**
```
✅ PASS: @CircuitBreaker annotation found in 1 file(s)
     - CustomerApplicationService.java
✅ PASS: Fallback methods defined (2 reference(s))
✅ PASS: Fallback method implemented: registerCustomerFallback()
✅ PASS: Fallback method implemented: getCustomerFallback()
✅ PASS: Resilience4j configuration present in application.yml
✅ PASS: Circuit breaker instances configured
✅ PASS: resilience4j-spring-boot dependency in pom.xml
⚠️  WARN: spring-boot-starter-aop dependency missing (CB may not work)
✅ PASS: Circuit breaker metrics exposed via actuator
✅ PASS: @CircuitBreaker correctly placed in application layer

✅ Circuit Breaker Feature: VALIDATED
```

**Failure:**
```
❌ FAIL: @CircuitBreaker annotation not found
❌ FAIL: Fallback methods not defined
❌ FAIL: Resilience4j configuration missing in application.yml

❌ Circuit Breaker Feature: VALIDATION FAILED
   Errors: 3
```

## Code Examples

### Expected Annotation Usage (Application Layer)

```java
// ✅ CORRECT: In application layer
@Service
public class CustomerApplicationService {
    
    @CircuitBreaker(name = "customerService", fallbackMethod = "registerCustomerFallback")
    public CustomerResponse registerCustomer(RegisterCustomerRequest request) {
        // Business logic
    }
    
    private CustomerResponse registerCustomerFallback(RegisterCustomerRequest request, Exception ex) {
        log.error("Circuit breaker activated: {}", ex.getMessage());
        throw new RuntimeException("Service temporarily unavailable");
    }
}
```

```java
// ❌ WRONG: In domain layer
public class CustomerDomainService {
    
    @CircuitBreaker(name = "customerService") // ❌ Should NOT be in domain
    public Customer registerCustomer(...) {
        // ...
    }
}
```

### Expected Configuration (application.yml)

```yaml
resilience4j:
  circuitbreaker:
    instances:
      customerService:
        registerHealthIndicator: true
        slidingWindowSize: 10
        minimumNumberOfCalls: 5
        permittedNumberOfCallsInHalfOpenState: 3
        waitDurationInOpenState: 10s
        failureRateThreshold: 50

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,circuitbreakers
```

### Expected Dependencies (pom.xml)

```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

## When This Runs

This validation runs when:
- A skill generates code using mod-code-001-circuit-breaker
- Feature `circuit_breaker` is enabled in input configuration

**Skills that use this:**
- skill-code-001-add-circuit-breaker
- skill-code-020-generate-microservice (when circuit_breaker enabled)

## Related

- **Module:** mod-code-001-circuit-breaker-java-resilience4j.md
- **ADR:** adr-004-resilience-patterns
- **Skill:** skill-code-001-add-circuit-breaker

---

**Validation Script Version:** 1.0  
**Last Updated:** 2025-11-25
