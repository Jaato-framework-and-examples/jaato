---
id: skill-020-microservice-java-spring
version: 2.1.0
tags:
  artifact-type: service
  runtime-model: request-response
  stack: java-spring
---

# skill-020-microservice-java-spring

## Overview

**Skill ID:** skill-020-microservice-java-spring  
**Type:** GENERATE  
**Framework:** Java 17+ / Spring Boot 3.2.x  
**Architecture:** Hexagonal Light  
**Role:** Base skill for Java/Spring microservices (extensible)

---

## Purpose

Generates a complete, production-ready Spring Boot microservice with Hexagonal Light architecture. This is the **base skill** for Java/Spring microservices that can be **extended** by specialized skills (REST APIs, gRPC, Async).

**Use this skill for:**
- Internal microservices implementing DDD bounded contexts
- Backend services not directly exposed as public APIs
- Base generation that specialized API skills extend

**For public APIs, use skills that extend this one:**
- **skill-021** for REST APIs (pagination, HATEOAS, compensation)
- **skill-022** for gRPC APIs (planned)
- **skill-023** for Async/Event APIs (planned)

---

## When to Use

✅ **Use this skill when:**
- Creating an **internal microservice** (not a public API)
- Implementing a **DDD bounded context** (entities, aggregates, domain services)
- Building **backend services** consumed only by other internal services
- Need **Hexagonal Light** architecture without API-specific patterns
- Want base microservice that you'll customize manually

❌ **Do NOT use when:**
- Building APIs that follow the **4-layer model** → use **skill-021**
- Need **pagination**, **HATEOAS**, or **filtering** → use **skill-021**
- Building a **Domain API** for SAGA orchestration → use **skill-021**
- Creating **gRPC** services → use **skill-022** (planned)
- Building **async/event-driven** services → use **skill-023** (planned)
- OpenAPI contract is a **primary deliverable** → use **skill-021**

---

## Extensibility

This skill serves as the **base** for specialized API skills:

```
skill-020-microservice-java-spring (this skill)
│
│   Base capabilities:
│   ├── Hexagonal Light structure (mod-015)
│   ├── Resilience patterns (mod-001-004)
│   ├── Persistence patterns (mod-016/017)
│   └── [future: observability, caching, etc.]
│
├── skill-021-api-rest-java-spring (extends)
│   └── Adds: pagination, HATEOAS, compensation
│
├── skill-022-api-grpc-java-spring (extends, planned)
│   └── Adds: proto, stubs, interceptors
│
└── skill-023-api-async-java-spring (extends, planned)
    └── Adds: Kafka, event schemas, consumers/producers
```

When you add capabilities to skill-020, all extending skills inherit them automatically.

---

## Capabilities

| Capability | Description |
|------------|-------------|
| **Project generation** | Complete Maven project with all dependencies |
| **Hexagonal Light structure** | Domain, Application, Adapter, Infrastructure layers |
| **Entity generation** | Domain entities, Value Objects, Repositories |
| **Resilience patterns** | Circuit breaker, retry, timeout, rate limiter |
| **Persistence patterns** | JPA or System API client |
| **Test generation** | Unit tests for domain layer |
| **Extensible base** | Foundation for API-specific skills |

---

## Input Summary

```json
{
  "serviceName": "customer-service",
  "basePackage": "com.company.customer",
  "entities": [
    {
      "name": "Customer",
      "fields": [
        { "name": "firstName", "type": "String" },
        { "name": "lastName", "type": "String" }
      ]
    }
  ],
  "features": {
    "resilience": { "enabled": true },
    "persistence": { "type": "jpa" }
  }
}
```

---

## Output Summary

```
customer-service/
├── pom.xml
├── src/main/java/.../
│   ├── domain/           # Pure POJOs (entities, value objects)
│   ├── application/      # @Service orchestration
│   ├── adapter/          # REST controller, persistence
│   └── infrastructure/   # Config, exception handlers
├── src/test/java/.../
├── src/main/resources/
│   └── application.yml
└── .enablement/
    └── manifest.json     # Traceability
```

---

## Dependencies

### Knowledge Dependencies
- **ADR-009:** Service Architecture Patterns (Hexagonal Light)
- **ERI-001:** Hexagonal Light Java Spring (reference implementation)

### Module Dependencies
- **mod-015:** hexagonal-base-java-spring (always)
- **mod-001-004:** resilience patterns (if enabled)
- **mod-016:** persistence-jpa-spring (if persistence.type=jpa)
- **mod-017:** persistence-systemapi (if persistence.type=system_api)

---

## Extended By

| Skill | Purpose | Adds |
|-------|---------|------|
| skill-021-api-rest-java-spring | REST APIs (4-layer model) | mod-019, mod-020 |
| skill-022-api-grpc-java-spring | gRPC APIs | (planned) |
| skill-023-api-async-java-spring | Async/Event APIs | (planned) |

---

## Tags

> Tags are defined in YAML frontmatter at the top of this file.
> See `model/domains/code/TAG-TAXONOMY.md` for CODE domain taxonomy.

---

## Version

**Current:** 2.1.0  
**Status:** Active  
**Last Updated:** 2025-12-24
