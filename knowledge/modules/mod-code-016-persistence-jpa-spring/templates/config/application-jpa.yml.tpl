# Template: application-jpa.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: JPA/Database configuration

spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/{{databaseName}}
    username: ${DB_USERNAME:postgres}
    password: ${DB_PASSWORD:postgres}
    driver-class-name: org.postgresql.Driver
  
  jpa:
    hibernate:
      ddl-auto: validate  # MUST be 'validate' in production
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        format_sql: true
    open-in-view: false  # MUST be disabled
    show-sql: false

logging:
  level:
    org.hibernate.SQL: DEBUG
    org.hibernate.type.descriptor.sql.BasicBinder: TRACE
