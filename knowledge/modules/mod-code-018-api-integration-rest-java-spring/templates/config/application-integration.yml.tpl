# Template: application-integration.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: Integration configuration for REST APIs

integration:
  {{apiName}}:
    base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
    timeout:
      connect: 5s
      read: 10s

{{#feign}}
# Feign Configuration
feign:
  client:
    config:
      default:
        connectTimeout: 5000
        readTimeout: 10000
        loggerLevel: BASIC
{{/feign}}

logging:
  level:
    {{basePackage}}.adapter.integration: DEBUG
    {{#feign}}
    feign: DEBUG
    {{/feign}}
