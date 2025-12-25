# Template: application-timeout.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: Resilience4j time limiter configuration

resilience4j:
  timelimiter:
    configs:
      default:
        timeoutDuration: 5s
        cancelRunningFuture: true
    
    instances:
      {{timelimiterName}}:
        baseConfig: default
        timeoutDuration: {{timeoutDuration}}s

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,timelimiters,timelimiterevents
  health:
    timelimiters:
      enabled: true
