---
skill_id: skill-code-020-generate-microservice-java-spring
skill_name: Generate Spring Boot Microservice
skill_type: creation
skill_domain: code
complexity: high
priority: high
version: 1.3.0
date: 2025-12-22
author: Fusion C4E Team
status: active
tags:
  - creation
  - generation
  - spring-boot
  - java
  - hexagonal
  - microservice
---

# Skill: Generate Spring Boot Microservice

## Overview

### Purpose

Generates a complete, production-ready Spring Boot microservice with Hexagonal Light architecture from a JSON configuration. This is the **foundation skill** for creating new microservices in the Fusion platform.

### Scope

This skill generates:
1. Complete Maven project structure
2. Domain layer (pure POJOs - entities, services, repositories)
3. Application layer (Spring @Service orchestration)
4. Adapter layer (REST controllers, DTOs, JPA persistence)
5. Infrastructure (configuration, exception handling)
6. Unit and integration tests
7. OpenAPI specification (generated from config)
8. Docker and deployment files (optional)

### Architecture

Implements **Hexagonal Light** architecture as defined in ADR-009:
- Domain layer has NO framework annotations
- Application layer bridges domain and adapters
- Adapters contain all framework-specific code
- Dependencies point inward (Adapters → Application → Domain)

---

## Knowledge Dependencies

### Implements ADRs
- **ADR-009:** Service Architecture Patterns (Hexagonal Light structure)
- **ADR-001:** API Design Standards (API types and constraints)
- **ADR-004:** Resilience Patterns (Circuit Breaker, Retry, Timeout, Rate Limiter)
- **ADR-011:** Persistence Patterns (JPA, System API)

### References ERIs
- **ERI-001:** Hexagonal Light Java Spring (reference implementation)
- **ERI-008:** Circuit Breaker Java Resilience4j
- **ERI-009:** Retry Java Resilience4j
- **ERI-010:** Timeout Java Resilience4j
- **ERI-011:** Rate Limiter Java Resilience4j
- **ERI-012:** Persistence Patterns Java Spring (JPA + System API)

### Uses Modules
- **mod-code-015:** hexagonal-base-java-spring (always - base templates)
- **mod-code-001:** circuit-breaker-java-resilience4j (if resilience.circuit_breaker.enabled)
- **mod-code-002:** retry-java-resilience4j (if resilience.retry.enabled)
- **mod-code-003:** timeout-java-resilience4j (if resilience.timeout.enabled)
- **mod-code-004:** rate-limiter-java-resilience4j (if resilience.rate_limiter.enabled)
- **mod-code-016:** persistence-jpa-spring (if persistence.type = "jpa")
- **mod-code-017:** persistence-systemapi (if persistence.type = "system_api")

---

## Input Specification

### JSON Config Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Microservice Generation Config",
  "type": "object",
  "required": ["serviceName", "groupId", "basePackage", "apiType", "entities"],
  "properties": {
    
    "serviceName": {
      "type": "string",
      "description": "Service name in kebab-case",
      "pattern": "^[a-z][a-z0-9-]*$",
      "examples": ["customer-service", "order-management-api"]
    },
    
    "groupId": {
      "type": "string",
      "description": "Maven group ID",
      "pattern": "^[a-z][a-z0-9]*(\\.[a-z][a-z0-9]*)*$",
      "examples": ["com.company", "com.company.customers"]
    },
    
    "artifactId": {
      "type": "string",
      "description": "Maven artifact ID (defaults to serviceName)",
      "pattern": "^[a-z][a-z0-9-]*$"
    },
    
    "basePackage": {
      "type": "string",
      "description": "Java base package",
      "pattern": "^[a-z][a-z0-9]*(\\.[a-z][a-z0-9]*)*$",
      "examples": ["com.company.customer"]
    },
    
    "javaVersion": {
      "type": "string",
      "description": "Java version",
      "enum": ["17", "21"],
      "default": "17"
    },
    
    "springBootVersion": {
      "type": "string",
      "description": "Spring Boot version",
      "default": "3.2.0"
    },
    
    "apiType": {
      "type": "string",
      "description": "Type of API (defines constraints and validations)",
      "enum": ["domain_api", "composable_api", "system_api", "experience_api"]
    },
    
    "entities": {
      "type": "array",
      "description": "List of domain entities to generate",
      "minItems": 1,
      "items": {
        "$ref": "#/definitions/Entity"
      }
    },
    
    "features": {
      "type": "object",
      "description": "Optional features to enable",
      "$ref": "#/definitions/Features"
    }
  },
  
  "definitions": {
    
    "Entity": {
      "type": "object",
      "required": ["name", "fields"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Entity name in PascalCase",
          "pattern": "^[A-Z][a-zA-Z0-9]*$",
          "examples": ["Customer", "Order", "Product"]
        },
        "fields": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/Field"
          }
        },
        "isAggregateRoot": {
          "type": "boolean",
          "description": "Whether this is the main entity (aggregate root)",
          "default": true
        }
      }
    },
    
    "Field": {
      "type": "object",
      "required": ["name", "type"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Field name in camelCase",
          "pattern": "^[a-z][a-zA-Z0-9]*$"
        },
        "type": {
          "type": "string",
          "description": "Java type",
          "enum": ["String", "int", "Integer", "long", "Long", "boolean", "Boolean", 
                   "double", "Double", "BigDecimal", "LocalDate", "LocalDateTime", 
                   "UUID", "List", "Set"]
        },
        "required": {
          "type": "boolean",
          "default": false
        },
        "unique": {
          "type": "boolean",
          "default": false
        },
        "minLength": {
          "type": "integer",
          "description": "Minimum length for String fields"
        },
        "maxLength": {
          "type": "integer",
          "description": "Maximum length for String fields"
        },
        "min": {
          "type": "number",
          "description": "Minimum value for numeric fields"
        },
        "max": {
          "type": "number",
          "description": "Maximum value for numeric fields"
        },
        "format": {
          "type": "string",
          "description": "Format hint (email, phone, etc.)",
          "enum": ["email", "phone", "url", "uuid"]
        }
      }
    },
    
    "Features": {
      "type": "object",
      "properties": {
        "resilience": {
          "type": "object",
          "properties": {
            "circuit_breaker": {
              "type": "object",
              "properties": {
                "enabled": { "type": "boolean", "default": false },
                "pattern": { 
                  "type": "string", 
                  "enum": ["basic_fallback", "fail_fast", "multiple_fallbacks"],
                  "default": "basic_fallback"
                }
              }
            },
            "retry": {
              "type": "object",
              "properties": {
                "enabled": { "type": "boolean", "default": false },
                "maxAttempts": { "type": "integer", "default": 3 },
                "strategy": {
                  "type": "string",
                  "enum": ["fixed_delay", "exponential_backoff"],
                  "default": "exponential_backoff"
                }
              }
            },
            "timeout": {
              "type": "object",
              "properties": {
                "enabled": { "type": "boolean", "default": false },
                "duration": { "type": "string", "default": "5s" }
              }
            },
            "rate_limiter": {
              "type": "object",
              "properties": {
                "enabled": { "type": "boolean", "default": false },
                "limitForPeriod": { "type": "integer", "default": 50 },
                "limitRefreshPeriod": { "type": "string", "default": "1s" }
              }
            }
          }
        },
        "persistence": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true },
            "type": {
              "type": "string",
              "enum": ["jpa", "system_api"],
              "default": "jpa",
              "description": "jpa for local DB, system_api for mainframe delegation"
            },
            "database": { 
              "type": "string", 
              "enum": ["postgresql", "mysql", "h2"],
              "default": "postgresql",
              "description": "Only applies when type=jpa"
            },
            "system_api": {
              "type": "object",
              "description": "Only applies when type=system_api",
              "properties": {
                "client": {
                  "type": "string",
                  "enum": ["feign", "resttemplate", "restclient"],
                  "default": "feign"
                },
                "base_url_env": {
                  "type": "string",
                  "description": "Environment variable for System API base URL",
                  "default": "SYSTEM_API_${SERVICE_NAME}_URL"
                }
              }
            }
          }
        },
        "health_checks": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true }
          }
        },
        "structured_logging": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true },
            "format": { 
              "type": "string", 
              "enum": ["json", "logfmt"],
              "default": "json"
            }
          }
        },
        "docker": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": true }
          }
        },
        "kubernetes": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean", "default": false }
          }
        }
      }
    }
  }
}
```

### Example Input

#### Example 1: Domain API with JPA (Data Owner)

```json
{
  "serviceName": "customer-service",
  "groupId": "com.company",
  "artifactId": "customer-service",
  "basePackage": "com.company.customer",
  "javaVersion": "17",
  "springBootVersion": "3.2.0",
  
  "apiType": "domain_api",
  
  "entities": [
    {
      "name": "Customer",
      "isAggregateRoot": true,
      "fields": [
        { "name": "name", "type": "String", "required": true, "minLength": 2, "maxLength": 100 },
        { "name": "email", "type": "String", "required": true, "format": "email", "unique": true },
        { "name": "age", "type": "int", "required": true, "min": 18, "max": 120 },
        { "name": "tier", "type": "String", "required": false }
      ]
    }
  ],
  
  "features": {
    "resilience": {
      "circuit_breaker": { "enabled": true, "pattern": "basic_fallback" }
    },
    "persistence": { 
      "enabled": true, 
      "type": "jpa",
      "database": "postgresql" 
    },
    "health_checks": { "enabled": true },
    "structured_logging": { "enabled": true, "format": "json" },
    "docker": { "enabled": true }
  }
}
```

#### Example 2: Domain API with System API (Mainframe Delegation)

```json
{
  "serviceName": "account-service",
  "groupId": "com.company",
  "artifactId": "account-service",
  "basePackage": "com.company.account",
  "javaVersion": "17",
  "springBootVersion": "3.2.0",
  
  "apiType": "domain_api",
  
  "entities": [
    {
      "name": "Account",
      "isAggregateRoot": true,
      "fields": [
        { "name": "accountNumber", "type": "String", "required": true },
        { "name": "balance", "type": "BigDecimal", "required": true },
        { "name": "status", "type": "String", "required": true },
        { "name": "customerId", "type": "String", "required": true }
      ]
    }
  ],
  
  "features": {
    "resilience": {
      "circuit_breaker": { "enabled": true, "pattern": "basic_fallback" },
      "retry": { "enabled": true, "maxAttempts": 3, "strategy": "exponential_backoff" },
      "timeout": { "enabled": true, "duration": "5s" }
    },
    "persistence": { 
      "enabled": true, 
      "type": "system_api",
      "system_api": {
        "client": "feign",
        "base_url_env": "SYSTEM_API_ACCOUNT_URL"
      }
    },
    "health_checks": { "enabled": true },
    "structured_logging": { "enabled": true, "format": "json" },
    "docker": { "enabled": true }
  }
}
```

---

## Output Specification

### Generated Structure

```
{{serviceName}}/
├── pom.xml
├── README.md
├── Dockerfile                                    (if docker.enabled)
├── .gitignore
│
├── src/main/java/{{basePackagePath}}/
│   │
│   ├── {{ServiceName}}Application.java
│   │
│   ├── domain/                                   # DOMAIN LAYER (Pure POJOs)
│   │   ├── model/
│   │   │   ├── {{Entity}}.java
│   │   │   ├── {{Entity}}Id.java
│   │   │   └── {{Entity}}Registration.java
│   │   ├── service/
│   │   │   └── {{Entity}}DomainService.java
│   │   ├── repository/
│   │   │   └── {{Entity}}Repository.java
│   │   └── exception/
│   │       └── {{Entity}}NotFoundException.java
│   │
│   ├── application/                              # APPLICATION LAYER
│   │   └── service/
│   │       └── {{Entity}}ApplicationService.java
│   │
│   ├── adapter/                                  # ADAPTER LAYER
│   │   ├── rest/
│   │   │   ├── controller/
│   │   │   │   └── {{Entity}}Controller.java
│   │   │   ├── dto/
│   │   │   │   ├── {{Entity}}DTO.java
│   │   │   │   ├── Create{{Entity}}Request.java
│   │   │   │   └── Update{{Entity}}Request.java
│   │   │   └── mapper/
│   │   │       └── {{Entity}}DtoMapper.java
│   │   │
│   │   └── persistence/
│   │       ├── entity/
│   │       │   └── {{Entity}}Entity.java
│   │       ├── repository/
│   │       │   └── {{Entity}}JpaRepository.java
│   │       ├── adapter/
│   │       │   └── {{Entity}}RepositoryAdapter.java
│   │       └── mapper/
│   │           └── {{Entity}}EntityMapper.java
│   │
│   └── infrastructure/
│       ├── config/
│       │   └── ApplicationConfig.java
│       └── exception/
│           ├── GlobalExceptionHandler.java
│           └── ErrorResponse.java
│
├── src/main/resources/
│   ├── application.yml
│   ├── application-dev.yml
│   ├── application-prod.yml
│   └── openapi.yaml                              # Generated OpenAPI spec
│
└── src/test/java/{{basePackagePath}}/
    ├── domain/service/
    │   └── {{Entity}}DomainServiceTest.java
    └── adapter/rest/controller/
        └── {{Entity}}ControllerIntegrationTest.java
```

### File Count Estimate

| Entity Count | Java Files | Test Files | Config Files | Total |
|--------------|------------|------------|--------------|-------|
| 1 entity | ~25 | ~5 | ~8 | ~38 |
| 2 entities | ~45 | ~10 | ~8 | ~63 |
| 3 entities | ~65 | ~15 | ~8 | ~88 |

---

## API Type Constraints

### domain_api

**Constraints applied:**
- ✅ No HTTP clients to other domain APIs generated
- ✅ Can include HTTP clients to System APIs (within same domain)
- ✅ Compensation endpoints generated for SAGA support
- ✅ Domain owns its data

**Validation warnings if:**
- Config includes external domain API calls

### composable_api

**Constraints applied:**
- ✅ HTTP clients for calling Domain APIs generated
- ✅ SAGA orchestration support included
- ✅ No direct database persistence (stateless)
- ✅ Circuit breaker required for external calls

**Validation warnings if:**
- persistence.enabled = true (composable should be stateless)

### system_api

**Constraints applied:**
- ✅ Designed for SoR integration
- ✅ Data transformation focus
- ✅ Error mapping to standard format

### experience_api

**Constraints applied:**
- ✅ BFF pattern support
- ✅ HTTP clients for Composable/Domain APIs
- ✅ Response transformation for UI
- ✅ Caching support

---

## Validation

### Input Validation

```yaml
serviceName:
  - pattern: "^[a-z][a-z0-9-]*$"
  - message: "Service name must be kebab-case"

basePackage:
  - pattern: "^[a-z][a-z0-9]*(\\.[a-z][a-z0-9]*)*$"
  - message: "Base package must be valid Java package"

entities:
  - minItems: 1
  - message: "At least one entity required"

entity.name:
  - pattern: "^[A-Z][a-zA-Z0-9]*$"
  - message: "Entity name must be PascalCase"

entity.fields:
  - minItems: 1
  - message: "Entity must have at least one field"
```

### Output Validation

```yaml
compilation:
  check: "Maven build succeeds"
  command: "mvn clean compile"
  severity: CRITICAL

tests:
  check: "All tests pass"
  command: "mvn test"
  severity: CRITICAL

structure:
  check: "Hexagonal Light structure valid"
  validations:
    - "Domain layer has no Spring annotations"
    - "Domain layer has no JPA annotations"
    - "Repository interface in domain layer"
    - "Repository implementation in adapter layer"
  severity: HIGH

adr_009_compliance:
  check: "Follows ADR-009 Hexagonal Light"
  severity: HIGH

adr_001_compliance:
  check: "API type constraints respected"
  severity: HIGH

adr_011_compliance:
  check: "Persistence patterns followed"
  validations:
    - "JPA entities only in adapter/persistence/entity/"
    - "System API clients only in adapter/systemapi/client/"
  severity: HIGH

adr_004_compliance:
  check: "Resilience patterns applied correctly"
  validations:
    - "System API adapters have @CircuitBreaker and @Retry"
    - "Annotation order: @RateLimiter > @CircuitBreaker > @TimeLimiter > @Retry"
  severity: HIGH
```

---

## Execution Flow

This skill follows the **GENERATE** execution flow defined at domain level.

**See:** [`runtime/flows/code/GENERATE.md`](../../runtime/flows/code/GENERATE.md)

The GENERATE flow provides the canonical 8-step process for code generation skills.
The steps below are skill-specific refinements of that flow.

---

## Execution Steps

### Step 1: Validate Input

```
1. Parse JSON config
2. Validate against schema
3. Check API type constraints
4. Validate entity definitions
5. Resolve default values
```

### Step 2: Load Modules

```
1. Load mod-code-015-hexagonal-base-java-spring (always)

2. Load persistence module based on config:
   - If features.persistence.type = "jpa":
     - Load mod-code-016-persistence-jpa-spring
   - If features.persistence.type = "system_api":
     - Load mod-code-017-persistence-systemapi
     - Select client variant: feign | resttemplate | restclient

3. Load resilience modules based on config:
   - If features.resilience.circuit_breaker.enabled:
     - Load mod-code-001-circuit-breaker-java-resilience4j
   - If features.resilience.retry.enabled:
     - Load mod-code-002-retry-java-resilience4j
   - If features.resilience.timeout.enabled:
     - Load mod-code-003-timeout-java-resilience4j
   - If features.resilience.rate_limiter.enabled:
     - Load mod-code-004-rate-limiter-java-resilience4j
```

### Step 3: Generate Project Structure

```
1. Create directory structure
2. Generate pom.xml with dependencies
3. Generate Application.java
4. Generate application.yml configs
```

### Step 4: Generate Domain Layer

```
For each entity:
  1. Generate {{Entity}}.java (domain model)
  2. Generate {{Entity}}Id.java (value object)
  3. Generate {{Entity}}Registration.java (command)
  4. Generate {{Entity}}DomainService.java (business logic)
  5. Generate {{Entity}}Repository.java (port interface)
  6. Generate {{Entity}}NotFoundException.java
```

### Step 5: Generate Application Layer

```
For each entity:
  1. Generate {{Entity}}ApplicationService.java
```

### Step 6: Generate Adapter Layer

```
For each entity:
  1. Generate REST adapter:
     - {{Entity}}Controller.java
     - {{Entity}}DTO.java
     - Create{{Entity}}Request.java
     - Update{{Entity}}Request.java
     - {{Entity}}DtoMapper.java
  
  2. Generate Persistence adapter:
     
     If persistence.type = "jpa":
       - {{Entity}}JpaEntity.java (adapter/persistence/entity/)
       - {{Entity}}JpaRepository.java (adapter/persistence/repository/)
       - {{Entity}}PersistenceAdapter.java (adapter/persistence/)
       - {{Entity}}PersistenceMapper.java (adapter/persistence/mapper/)
     
     If persistence.type = "system_api":
       - {{Entity}}Dto.java (adapter/systemapi/dto/)
       - {{Entity}}SystemApiClient.java (adapter/systemapi/client/)
         - Variant based on system_api.client: feign | resttemplate | restclient
       - {{Entity}}SystemApiAdapter.java (adapter/systemapi/)
         - Include @CircuitBreaker, @Retry if resilience enabled
       - {{Entity}}SystemApiMapper.java (adapter/systemapi/mapper/)
       - SystemApiFeignConfig.java (if using feign)
```

### Step 7: Generate Infrastructure

```
1. Generate ApplicationConfig.java
2. Generate GlobalExceptionHandler.java
3. Generate ErrorResponse.java
```

### Step 8: Generate Tests

```
For each entity:
  1. Generate {{Entity}}DomainServiceTest.java (unit test)
  2. Generate {{Entity}}ControllerIntegrationTest.java
```

### Step 9: Generate OpenAPI Spec

```
1. Generate openapi.yaml from entities and endpoints
2. Include all DTOs as schemas
3. Document all endpoints
```

### Step 10: Apply Features

```
If resilience.circuit_breaker.enabled:
  - Add Resilience4j dependencies
  - Add circuit breaker config to application.yml
  - Apply patterns from mod-code-001

If resilience.retry.enabled:
  - Add retry config to application.yml
  - Apply patterns from mod-code-002

If resilience.timeout.enabled:
  - Add timelimiter config to application.yml
  - Apply patterns from mod-code-003
  - Ensure CompletableFuture return types

If resilience.rate_limiter.enabled:
  - Add ratelimiter config to application.yml
  - Apply patterns from mod-code-004

If persistence.type = "system_api":
  - Resilience patterns MUST be applied to SystemApiAdapter
  - Add System API client dependencies (Feign/RestTemplate/RestClient)
  - Configure base URL via environment variable

**IMPORTANT: Resilience Implementation Requirements**

When resilience features are enabled, they MUST be implemented using Resilience4j annotations, 
NOT just documented in comments. The annotations must be applied to:

1. **System API Adapters/Clients**: All methods that call external System APIs
2. **Application Services**: Methods that orchestrate domain operations

Required annotations (when corresponding feature is enabled):
- `@CircuitBreaker(name = "...", fallbackMethod = "...")`
- `@Retry(name = "...")`
- `@TimeLimiter(name = "...")`
- `@RateLimiter(name = "...")`

Annotation order (per ADR-004): @RateLimiter > @CircuitBreaker > @TimeLimiter > @Retry

Each annotated method MUST have a corresponding fallback method for @CircuitBreaker.

If docker.enabled:
  - Generate Dockerfile
  - Generate .dockerignore
```

### Step 11: Validate Output

```
1. Run mvn clean compile
2. Run mvn test
3. Verify structure compliance
4. Report any warnings
```

---

## Success Metrics

| Metric | Threshold |
|--------|-----------|
| Build succeeds | 100% |
| Tests pass | 100% |
| Domain layer Spring-free | 100% |
| Structure compliance | 100% |
| Code coverage (domain) | ≥80% |

---

## Template Mapping Reference

This section provides the **explicit mapping** between output files and module templates.
When generating, Claude MUST use these templates - not improvise.

### Base Templates (mod-code-015-hexagonal-base-java-spring) - ALWAYS

| Output File | Template | Notes |
|-------------|----------|-------|
| `{{ServiceName}}Application.java` | `templates/Application.java.tpl` | Main class |
| `pom.xml` | `templates/config/pom.xml.tpl` | Maven config |
| `application.yml` | `templates/config/application.yml.tpl` | Spring config |

### Domain Layer (mod-code-015)

| Output File | Template | Notes |
|-------------|----------|-------|
| `domain/model/{{Entity}}.java` | `templates/domain/Entity.java.tpl` | Pure POJO |
| `domain/model/{{Entity}}Id.java` | `templates/domain/EntityId.java.tpl` | Value object |
| `domain/model/{{EnumName}}.java` | `templates/domain/Enum.java.tpl` | Domain enum |
| `domain/repository/{{Entity}}Repository.java` | `templates/domain/Repository.java.tpl` | Port interface |
| `domain/service/{{Entity}}DomainService.java` | `templates/domain/DomainService.java.tpl` | POJO service |
| `domain/exception/{{Entity}}NotFoundException.java` | `templates/domain/NotFoundException.java.tpl` | Domain exception |

### Application Layer (mod-code-015)

| Output File | Template | Notes |
|-------------|----------|-------|
| `application/{{Entity}}ApplicationService.java` | `templates/application/ApplicationService.java.tpl` | @Service |
| `application/dto/Create{{Entity}}Request.java` | `templates/application/dto/CreateRequest.java.tpl` | Input DTO |
| `application/dto/Update{{Entity}}Request.java` | `templates/application/dto/UpdateRequest.java.tpl` | Input DTO |
| `application/dto/{{Entity}}Response.java` | `templates/application/dto/Response.java.tpl` | Output DTO |

### Adapter Layer - REST Inbound (mod-code-015)

| Output File | Template | Notes |
|-------------|----------|-------|
| `adapter/rest/{{Entity}}Controller.java` | `templates/adapter/RestController.java.tpl` | @RestController |

### Adapter Layer - Persistence JPA (mod-code-016-persistence-jpa-spring)

| Output File | Template | When |
|-------------|----------|------|
| `adapter/persistence/entity/{{Entity}}JpaEntity.java` | `templates/entity/JpaEntity.java.tpl` | persistence.type = "jpa" |
| `adapter/persistence/repository/{{Entity}}JpaRepository.java` | `templates/repository/JpaRepository.java.tpl` | persistence.type = "jpa" |
| `adapter/persistence/{{Entity}}PersistenceAdapter.java` | `templates/adapter/PersistenceAdapter.java.tpl` | persistence.type = "jpa" |
| `adapter/persistence/mapper/{{Entity}}PersistenceMapper.java` | `templates/mapper/PersistenceMapper.java.tpl` | persistence.type = "jpa" |

### Adapter Layer - System API (mod-code-017 + mod-code-018)

| Output File | Template | Module | When |
|-------------|----------|--------|------|
| `adapter/systemapi/dto/{{Api}}Dto.java` | `templates/dto/Dto.java.tpl` | mod-code-017 | persistence.type = "system_api" |
| `adapter/systemapi/mapper/{{Entity}}Mapper.java` | `templates/mapper/SystemApiMapper.java.tpl` | mod-code-017 | persistence.type = "system_api" |
| `adapter/systemapi/{{Entity}}SystemApiAdapter.java` | `templates/adapter/SystemApiAdapter.java.tpl` | mod-code-017 | persistence.type = "system_api" |
| `adapter/systemapi/SystemApiUnavailableException.java` | `templates/exception/SystemApiUnavailableException.java.tpl` | mod-code-017 | persistence.type = "system_api" |
| `adapter/systemapi/client/{{Api}}Client.java` | `templates/client/restclient.java.tpl` | mod-code-018 | system_api.client = "restclient" |
| `adapter/systemapi/client/{{Api}}Client.java` | `templates/client/feign.java.tpl` | mod-code-018 | system_api.client = "feign" |
| `adapter/systemapi/client/{{Api}}Client.java` | `templates/client/resttemplate.java.tpl` | mod-code-018 | system_api.client = "resttemplate" |
| `infrastructure/config/RestClientConfig.java` | `templates/config/restclient-config.java.tpl` | mod-code-018 | system_api.client = "restclient" |
| `infrastructure/config/FeignConfig.java` | `templates/config/feign-config.java.tpl` | mod-code-018 | system_api.client = "feign" |

### Infrastructure (mod-code-015)

| Output File | Template | Notes |
|-------------|----------|-------|
| `infrastructure/config/ApplicationConfig.java` | `templates/infrastructure/ApplicationConfig.java.tpl` | @Bean for domain |
| `infrastructure/GlobalExceptionHandler.java` | `templates/infrastructure/GlobalExceptionHandler.java.tpl` | RFC 7807 |
| `infrastructure/CorrelationIdFilter.java` | `templates/infrastructure/CorrelationIdFilter.java.tpl` | MDC filter |

### Tests (mod-code-015 + module-specific)

| Output File | Template | Module |
|-------------|----------|--------|
| `test/.../{{Entity}}DomainServiceTest.java` | `templates/test/DomainServiceTest.java.tpl` | mod-code-015 |
| `test/.../{{Entity}}PersistenceAdapterTest.java` | `templates/test/PersistenceAdapterTest.java.tpl` | mod-code-016 |
| `test/.../{{Entity}}SystemApiAdapterTest.java` | `templates/test/SystemApiAdapterTest.java.tpl` | mod-code-017 |

### Resilience Configuration (mod-code-001, mod-code-002, mod-code-003, mod-code-004)

These modules provide **configuration snippets** to merge into `application.yml`:

| Feature | Module | Config Template |
|---------|--------|-----------------|
| Circuit Breaker | mod-code-001 | `templates/config/application-circuitbreaker.yml.tpl` |
| Retry | mod-code-002 | `templates/config/application-retry.yml.tpl` |
| Timeout | mod-code-003 | `templates/config/application-timeout.yml.tpl` |
| Rate Limiter | mod-code-004 | `templates/config/application-ratelimiter.yml.tpl` |

### Resilience Annotations

Resilience annotations are applied to **SystemApiAdapter** methods when using System API persistence:

| Pattern | Annotation | Source Module |
|---------|------------|---------------|
| Circuit Breaker | `@CircuitBreaker(name=..., fallbackMethod=...)` | mod-code-001 |
| Retry | `@Retry(name=...)` | mod-code-002 |
| Timeout | Client-level (RestClient factory) | mod-code-018 |

> **Note:** @TimeLimiter requires async methods (CompletableFuture). For sync methods,
> use client-level timeout configuration in RestClientConfig.

---

## Related Skills

### Complements
- **skill-code-001:** add-circuit-breaker-java-resilience4j
- **skill-code-002:** add-retry-java-resilience4j
- **skill-code-003:** add-timeout-java-resilience4j
- **skill-code-004:** add-rate-limiter-java-resilience4j
- **skill-code-005:** add-jpa-persistence-java-spring (future)
- **skill-code-006:** add-systemapi-persistence-java-spring (future)

### Builds Toward
- **skill-code-021:** generate-domain-api-java-spring (extends with Domain API specifics)
- **skill-code-022:** generate-composable-api-java-spring (extends with SAGA support)

---

## Changelog

### Version 1.2.0 (2025-12-01)
- Added full resilience support: circuit_breaker, retry, timeout, rate_limiter
- Added persistence type selection: jpa vs system_api
- Added System API client options: feign, resttemplate, restclient
- Updated module references: mod-code-001 through mod-code-004, mod-code-016, mod-code-017
- Added ADR-004 and ADR-011 compliance validation
- Added Example 2: Domain API with System API persistence

### Version 1.1.0 (2025-11-26)
- Updated to new skill naming convention (skill-code-020)
- Added skill_domain field
- Updated references to other skills

### Version 1.0.0 (2025-11-24)
- Initial skill definition
- Hexagonal Light architecture
- JSON config as input
- OpenAPI generation as output
- Support for domain_api, composable_api, system_api, experience_api
- Feature support: resilience, persistence, health_checks, logging, docker
