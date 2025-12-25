// Template: fail-fast.java.tpl
// Output: Applied to existing service class
// Purpose: Circuit breaker without fallback (fail fast for non-critical operations)

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;

// No fallbackMethod - throws CallNotPermittedException when circuit is open
@CircuitBreaker(name = "{{circuitBreakerName}}")
public {{returnType}} {{methodName}}({{methodParameters}}) throws {{exceptionType}} {
    {{originalMethodBody}}
}

// Caller must handle CallNotPermittedException:
//
// try {
//     {{returnType}} result = service.{{methodName}}({{arguments}});
//     // Process result
// } catch (CallNotPermittedException e) {
//     log.warn("Circuit breaker open, skipping operation: {}", e.getMessage());
//     // Graceful handling - operation skipped
// }
