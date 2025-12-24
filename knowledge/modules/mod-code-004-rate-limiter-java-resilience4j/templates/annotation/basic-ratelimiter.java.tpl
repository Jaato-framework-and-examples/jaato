// Template: basic-ratelimiter.java.tpl
// Output: Applied to existing service class
// Purpose: Basic rate limiter pattern

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import lombok.extern.slf4j.Slf4j;

@RateLimiter(name = "{{rateLimiterName}}")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}
