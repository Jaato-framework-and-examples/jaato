// Template: full-resilience-stack.java.tpl
// Output: Applied to existing service class
// Purpose: Full resilience stack (Rate Limiter + Circuit Breaker + Retry)

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.extern.slf4j.Slf4j;

/**
 * Annotation order (execution is reverse):
 * @RateLimiter (outer) -> @CircuitBreaker -> @Retry (inner) -> Actual call
 * 
 * This means:
 * 1. Rate limiter checks if request is allowed
 * 2. Circuit breaker checks if circuit is open
 * 3. Retry handles transient failures
 * 4. Actual call is made
 */
@RateLimiter(name = "{{serviceName}}")
@CircuitBreaker(name = "{{serviceName}}", fallbackMethod = "{{methodName}}Fallback")
@Retry(name = "{{serviceName}}")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}

private {{ReturnType}} {{methodName}}Fallback({{ParamType}} {{paramName}}, Exception ex) {
    log.warn("Service unavailable for {}. Error: {}", {{paramName}}, ex.getClass().getSimpleName());
    return {{defaultValue}};
}
