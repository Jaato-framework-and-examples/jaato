# Template: application-ratelimiter.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: Resilience4j rate limiter configuration

resilience4j:
  ratelimiter:
    configs:
      default:
        limitForPeriod: 50
        limitRefreshPeriod: 1s
        timeoutDuration: 0
        registerHealthIndicator: true
    
    instances:
      {{rateLimiterName}}:
        baseConfig: default
        limitForPeriod: {{limitForPeriod}}
        limitRefreshPeriod: {{limitRefreshPeriod}}

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,ratelimiters,ratelimiterevents
  health:
    ratelimiters:
      enabled: true
