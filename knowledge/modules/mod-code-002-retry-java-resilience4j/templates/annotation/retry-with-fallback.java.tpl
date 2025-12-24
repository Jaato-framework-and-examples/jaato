// Template: retry-with-fallback.java.tpl
// Output: Applied to existing service class
// Purpose: Retry pattern with fallback method

import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Retry(name = "{{retryName}}", fallbackMethod = "{{methodName}}Fallback")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}

private {{ReturnType}} {{methodName}}Fallback({{ParamType}} {{paramName}}, Exception ex) {
    log.warn("All retries exhausted for {}. Error: {}", {{paramName}}, ex.getMessage());
    return cacheService.getCached({{paramName}})
        .orElseThrow(() -> new ServiceUnavailableException("Service unavailable", ex));
}
