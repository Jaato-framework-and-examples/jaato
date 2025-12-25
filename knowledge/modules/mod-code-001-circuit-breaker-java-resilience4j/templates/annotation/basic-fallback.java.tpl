// Template: basic-fallback.java.tpl
// Output: Applied to existing service class
// Purpose: Circuit breaker with single fallback (most common - 80% of cases)

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
