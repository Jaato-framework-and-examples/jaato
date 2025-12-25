// Template: retry-with-circuitbreaker.java.tpl
// Output: Applied to existing service class
// Purpose: Retry combined with Circuit Breaker (proper order)

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Annotation order: CircuitBreaker (outer) -> Retry (inner) -> Actual call
 * This means: Retry happens inside the circuit breaker.
 * If retries exhaust, circuit breaker counts it as a failure.
 */
@CircuitBreaker(name = "{{serviceName}}", fallbackMethod = "{{methodName}}Fallback")
@Retry(name = "{{serviceName}}")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}

private {{ReturnType}} {{methodName}}Fallback({{ParamType}} {{paramName}}, Exception ex) {
    log.warn("Service unavailable for {}. Error: {}", {{paramName}}, ex.getMessage());
    return {{defaultValue}};
}
