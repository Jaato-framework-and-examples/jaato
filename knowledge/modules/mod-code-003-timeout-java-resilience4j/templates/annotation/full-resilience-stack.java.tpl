// Template: full-resilience-stack.java.tpl
// Output: Applied to existing service class
// Purpose: Full resilience stack (Circuit Breaker + Timeout + Retry)

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CompletableFuture;

/**
 * Annotation order (execution is reverse):
 * @CircuitBreaker (outer) -> @TimeLimiter -> @Retry (inner) -> Actual call
 * 
 * This means:
 * 1. Call is made
 * 2. Retry handles transient failures
 * 3. TimeLimiter enforces timeout
 * 4. CircuitBreaker tracks failures and provides fallback
 */
@CircuitBreaker(name = "{{serviceName}}", fallbackMethod = "{{methodName}}Fallback")
@TimeLimiter(name = "{{serviceName}}")
@Retry(name = "{{serviceName}}")
public CompletableFuture<{{ReturnType}}> {{methodName}}({{ParamType}} {{paramName}}) {
    return CompletableFuture.supplyAsync(() -> {
        log.debug("Calling external service: {}", {{paramName}});
        return client.{{clientMethod}}({{paramName}});
    });
}

private CompletableFuture<{{ReturnType}}> {{methodName}}Fallback({{ParamType}} {{paramName}}, Exception ex) {
    log.warn("Service unavailable for {}. Error: {}", {{paramName}}, ex.getClass().getSimpleName());
    return CompletableFuture.completedFuture({{defaultValue}});
}
