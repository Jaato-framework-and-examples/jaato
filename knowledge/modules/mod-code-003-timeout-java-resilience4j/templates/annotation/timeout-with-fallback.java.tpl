// Template: timeout-with-fallback.java.tpl
// Output: Applied to existing service class
// Purpose: Timeout pattern with fallback method

import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeoutException;

@TimeLimiter(name = "{{timelimiterName}}", fallbackMethod = "{{methodName}}Fallback")
public CompletableFuture<{{ReturnType}}> {{methodName}}({{ParamType}} {{paramName}}) {
    return CompletableFuture.supplyAsync(() -> {
        log.debug("Calling external service: {}", {{paramName}});
        return client.{{clientMethod}}({{paramName}});
    });
}

private CompletableFuture<{{ReturnType}}> {{methodName}}Fallback({{ParamType}} {{paramName}}, TimeoutException ex) {
    log.warn("Timeout for {}. Error: {}", {{paramName}}, ex.getMessage());
    return CompletableFuture.completedFuture({{defaultValue}});
}
