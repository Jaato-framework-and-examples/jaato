# Template: application-systemapi.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: System API configuration with resilience settings

# System API Configuration
system-api:
  {{serviceName}}:
    base-url: ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}

# Feign Configuration (if using Feign)
{{#feign}}
feign:
  client:
    config:
      default:
        connectTimeout: 5000
        readTimeout: 5000
        loggerLevel: BASIC
{{/feign}}

# Resilience4j Configuration (MANDATORY for System API)
resilience4j:
  circuitbreaker:
    instances:
      {{serviceName}}:
        slidingWindowSize: 100
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        permittedNumberOfCallsInHalfOpenState: 10
        slidingWindowType: COUNT_BASED
        
  retry:
    instances:
      {{serviceName}}:
        maxAttempts: 3
        waitDuration: 1s
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2
        retryExceptions:
          - java.io.IOException
          - java.net.SocketTimeoutException
          - org.springframework.web.client.ResourceAccessException
        ignoreExceptions:
          - {{basePackage}}.domain.exception.{{Entity}}NotFoundException

  timelimiter:
    instances:
      {{serviceName}}:
        timeoutDuration: 10s
        cancelRunningFuture: true

# Logging
logging:
  level:
    {{basePackage}}.adapter.systemapi: DEBUG
    {{#feign}}
    feign: DEBUG
    {{/feign}}
