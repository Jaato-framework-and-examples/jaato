# Template: application.yml.tpl
# Output: src/main/resources/application.yml
# Purpose: Main application configuration

spring:
  application:
    name: {{serviceName}}

server:
  port: 8080

management:
  server:
    port: 8081
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  endpoint:
    health:
      show-details: always
      probes:
        enabled: true

logging:
  level:
    {{basePackage}}: INFO
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} [%X{correlationId}] %-5level %logger{36} - %msg%n"
