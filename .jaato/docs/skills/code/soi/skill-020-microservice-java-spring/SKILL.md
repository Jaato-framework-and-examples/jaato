---
id: skill-020-microservice-java-spring
title: "SKILL-020: Generate Microservice - Java/Spring Boot"
version: 2.0
domain: code
layer: soi
flow: GENERATE
framework: java
library: spring-boot
complexity: complex
time_estimate: 5-10 minutes
modules:
  - mod-code-015-hexagonal-base-java-spring
implements:
  - adr-009-service-architecture-patterns
  - adr-001-api-design-standards
---

# SKILL-020: Generate Microservice - Java/Spring Boot

## Purpose

Generate production-ready Spring Boot microservice using Hexagonal Light architecture.

## Output Structure

```
{service-name}/
├── pom.xml
├── src/main/java/{package}/
│   ├── Application.java
│   ├── domain/           # Pure POJOs - NO framework annotations
│   ├── application/      # @Service orchestration
│   ├── adapter/          # REST controllers, JPA entities
│   └── infrastructure/   # Config, exception handling
└── src/test/java/
```

## Validation

- [ ] Maven build succeeds
- [ ] Domain layer has NO framework annotations
- [ ] Tests pass
