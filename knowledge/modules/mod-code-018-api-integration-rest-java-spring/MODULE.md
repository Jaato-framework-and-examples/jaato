---
id: mod-code-018-api-integration-rest-java-spring
title: "MOD-018: API Integration REST - Java/Spring"
version: 1.1
date: 2025-12-22
status: Active
derived_from: eri-code-013-api-integration-rest-java-spring
domain: code
tags:
  - java
  - spring-boot
  - rest-client
  - feign
  - resttemplate
  - integration
used_by:
  - skill-code-020-generate-microservice-java-spring
  - mod-code-017-persistence-systemapi

# Variant Configuration (v1.1)
variants:
  enabled: true
  selection_mode: explicit  # Don't auto-suggest, let user choose
  
  default:
    id: restclient
    name: "RestClient (Spring 6.1+)"
    description: "Modern REST client, recommended for Spring Boot 3.2+"
    templates:
      - client/restclient.java.tpl
      - config/restclient-config.java.tpl
    
  alternatives:
    - id: feign
      name: "OpenFeign (Declarative)"
      description: "Declarative REST client with interface-based definition"
      templates:
        - client/feign.java.tpl
        - config/feign-config.java.tpl
      recommend_when:
        - condition: "Existing codebase uses Feign extensively"
          reason: "Maintain consistency with existing patterns"
        - condition: "Team prefers declarative interface style"
          reason: "Simpler API definition"
          
    - id: resttemplate
      name: "RestTemplate (Legacy)"
      description: "Traditional REST client for legacy compatibility"
      templates:
        - client/resttemplate.java.tpl
        - config/resttemplate-config.java.tpl
      deprecated: true
      deprecation_reason: "RestClient is the modern replacement. Use only for legacy compatibility."
---

# MOD-018: API Integration REST - Java/Spring

## Overview

Reusable templates for implementing REST API integration in Java/Spring Boot applications. Supports three functionally equivalent client implementations.

**Source ERI:** [ERI-CODE-013](../../../ERIs/eri-code-013-api-integration-rest-java-spring/ERI.md)

**Use when:** Service needs to call external REST APIs (internal services, System APIs, third-party)

**Client variants:** RestClient (default), Feign, RestTemplate

---

## Structure

```
mod-code-018-api-integration-rest-java-spring/
├── MODULE.md
├── templates/
│   ├── client/
│   │   ├── restclient.java.tpl      # Default - Spring 3.2+
│   │   ├── feign.java.tpl           # Declarative
│   │   └── resttemplate.java.tpl    # Legacy
│   ├── config/
│   │   ├── restclient-config.java.tpl
│   │   ├── feign-config.java.tpl
│   │   ├── resttemplate-config.java.tpl
│   │   └── application-integration.yml.tpl
│   ├── exception/
│   │   └── IntegrationException.java.tpl
│   └── test/
│       └── ClientTest.java.tpl
└── validation/
    ├── README.md
    └── integration-check.sh
```

---

## Client Selection

| Input Parameter | Template Selected |
|-----------------|-------------------|
| `integration.client = "restclient"` | `client/restclient.java.tpl` |
| `integration.client = "feign"` | `client/feign.java.tpl` |
| `integration.client = "resttemplate"` | `client/resttemplate.java.tpl` |

**Default:** `restclient`

---

## Template Variables

### Common Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{basePackage}}` | Base Java package | `com.company.customer` |
| `{{serviceName}}` | Service name | `customer-domain-api` |
| `{{ApiName}}` | API name (PascalCase) | `PartiesApi` |
| `{{apiName}}` | API name (camelCase) | `partiesApi` |
| `{{Entity}}` | Entity name | `Party` |
| `{{baseUrlEnv}}` | Env var for base URL | `PARTIES_API_URL` |
| `{{resourcePath}}` | API resource path | `/parties` |

### Endpoint Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{#endpoints}}` | List of endpoints | - |
| `{{httpMethod}}` | HTTP method | `GET`, `POST` |
| `{{path}}` | Endpoint path | `/{id}` |
| `{{pathParams}}` | Path parameters | `@PathVariable String id` |
| `{{requestBody}}` | Request body type | `PartyDto` |
| `{{responseType}}` | Response type | `PartyDto` |

---

## Templates Reference

Templates are in separate `.tpl` files. See:

- `templates/client/restclient.java.tpl` - RestClient implementation
- `templates/client/feign.java.tpl` - Feign interface
- `templates/client/resttemplate.java.tpl` - RestTemplate implementation
- `templates/config/*.tpl` - Configuration for each client type
- `templates/exception/IntegrationException.java.tpl` - Common exception

---

## Usage by Other Modules

### mod-code-017-persistence-systemapi

When persistence uses System API, it depends on this module for the REST client:

```
mod-017 (persistence) 
    └── uses mod-018 (integration) for REST client
```

The adapter in mod-017 wraps the client from mod-018 with resilience patterns.

---

## Validation

See [validation/README.md](validation/README.md) for:

- Correlation headers check
- Base URL externalization check
- Error handling check

---

## Related

- **Source ERI:** [ERI-CODE-013](../../../ERIs/eri-code-013-api-integration-rest-java-spring/ERI.md)
- **ADR:** [ADR-012](../../../ADRs/adr-012-api-integration-patterns/ADR.md)
- **Used by:** mod-code-017-persistence-systemapi, skill-code-020
- **Capability:** integration.api.rest
