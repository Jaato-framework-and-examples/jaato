// Template: chain-fallback.java.tpl
// Output: Applied to existing service class
// Purpose: Circuit breaker with tiered fallback chain (alternative service → cache → default)

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
