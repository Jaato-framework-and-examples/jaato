// Template: basic-timeout.java.tpl
// Output: Applied to existing service class
// Purpose: Basic timeout pattern (MUST return CompletableFuture)

import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CompletableFuture;

/**
 * IMPORTANT: Methods with @TimeLimiter MUST return CompletableFuture<T>
 */
@TimeLimiter(name = "{{timelimiterName}}")
public CompletableFuture<{{ReturnType}}> {{methodName}}({{ParamType}} {{paramName}}) {
    return CompletableFuture.supplyAsync(() -> {
        log.debug("Calling external service: {}", {{paramName}});
        return client.{{clientMethod}}({{paramName}});
    });
}
