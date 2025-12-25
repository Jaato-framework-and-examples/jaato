# skill-021-api-rest-java-spring

> **Extends skill-020** to generate REST APIs following the 4-layer API model

## Quick Start

```bash
# Generate a Domain API
enablement generate --skill skill-021-api-rest-java-spring \
  --input generation-request.json \
  --output ./generated
```

## What This Skill Does

**Extends skill-020-microservice-java-spring** with REST API patterns:

- **Inherits from skill-020**: Hexagonal Light, resilience, persistence, etc.
- **Adds**: Pagination, HATEOAS, Compensation (for Domain APIs)

### Inheritance Model

```
skill-020-microservice-java-spring (base)
│   ├── Hexagonal Light (mod-015)
│   ├── Resilience (mod-001-004)
│   ├── Persistence (mod-016/017)
│   └── [future: observability, caching]
│
└── skill-021-api-rest-java-spring (this - delta)
    ├── Pagination (mod-019)
    ├── HATEOAS (mod-019)
    └── Compensation (mod-020, Domain only)
```

## API Layers

| Layer | Use Case | Key Features |
|-------|----------|--------------|
| **Experience** | BFF for mobile/web | HATEOAS, caching, aggregation |
| **Composable** | Multi-domain workflows | Orchestration, pagination |
| **Domain** | Business capabilities | HATEOAS, compensation, business rules |
| **System** | SoR integration | Data transformation, simple contracts |

## Example Input

```json
{
  "serviceName": "customer-management-api",
  "basePackage": "com.bank.customer",
  "apiLayer": "domain",
  "entities": [
    {
      "name": "Customer",
      "fields": [
        { "name": "firstName", "type": "String", "required": true },
        { "name": "lastName", "type": "String", "required": true },
        { "name": "email", "type": "String", "required": true }
      ]
    }
  ],
  "features": {
    "resilience": { "enabled": true },
    "persistence": { "type": "system_api" }
  }
}
```

## Generated Output

```
customer-management-api/
├── pom.xml
├── src/main/java/.../
│   ├── domain/           # Pure business logic
│   ├── application/      # Use cases
│   ├── adapter/in/rest/  # REST controllers
│   │   ├── dto/PageResponse.java
│   │   └── assembler/    # HATEOAS
│   └── infrastructure/
├── src/main/resources/
│   ├── application.yml
│   └── openapi.yaml
└── .enablement/manifest.json
```

## Features by Layer

| Feature | Experience | Composable | Domain | System |
|---------|------------|------------|--------|--------|
| Pagination | ✅ | ✅ | ✅ | ✅ |
| HATEOAS | ✅ | ❌ | ✅ | ❌ |
| Compensation | ❌ | ❌ | ✅ | ❌ |
| Caching hints | ✅ | ❌ | ❌ | ❌ |

## Related Skills

- **skill-020**: **Parent** - base microservice (inherited)
- **skill-022**: **Sibling** - gRPC APIs (planned, extends skill-020)
- **skill-023**: **Sibling** - Async/Event APIs (planned, extends skill-020)

## Documentation

- [SKILL.md](./SKILL.md) - Full specification
- [OVERVIEW.md](./OVERVIEW.md) - Quick reference
- [Examples](./prompts/examples/) - Sample inputs/outputs

## Version

- **Current**: 2.0.0
- **Status**: Active
- **Last Updated**: 2025-12-19
