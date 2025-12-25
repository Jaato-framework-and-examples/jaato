// Template: ratelimiter-with-fallback.java.tpl
// Output: Applied to existing service class
// Purpose: Rate limiter with fallback (queue for later processing)

import io.github.resilience4j.ratelimiter.RequestNotPermitted;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import lombok.extern.slf4j.Slf4j;

@RateLimiter(name = "{{rateLimiterName}}", fallbackMethod = "{{methodName}}Fallback")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}

private {{ReturnType}} {{methodName}}Fallback({{ParamType}} {{paramName}}, RequestNotPermitted ex) {
    log.warn("Rate limit exceeded for {}. Queueing.", {{paramName}});
    queueService.enqueue({{paramName}});
    return {{queuedResponse}};
}
