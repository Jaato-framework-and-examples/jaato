// Template: basic-retry.java.tpl
// Output: Applied to existing service class
// Purpose: Basic retry pattern without fallback

import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Retry(name = "{{retryName}}")
public {{ReturnType}} {{methodName}}({{ParamType}} {{paramName}}) {
    log.debug("Calling external service: {}", {{paramName}});
    return client.{{clientMethod}}({{paramName}});
}
