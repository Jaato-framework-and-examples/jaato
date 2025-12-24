---
id: adr-011-persistence-patterns
title: "ADR-011: Persistence Patterns"
sidebar_label: Persistence Patterns
version: 1.0
date: 2025-11-28
status: Accepted
author: Architecture Team
domain: architecture
tags:
  - persistence
  - jpa
  - system-api
  - data-access
  - hexagonal
related:
  - adr-009-service-architecture-patterns
  - adr-004-resilience-patterns
  - adr-001-api-design-standards
---

# ADR-011: Persistence Patterns

## Status

**Accepted**

## Context

Our microservices architecture requires clear patterns for data persistence. The organization has a specific context:

1. **System of Record (SoR)** resides in the **mainframe** (Z/OS, DB2, CICS)
2. **System APIs** wrap mainframe transactions and expose them via REST/JSON
3. **Domain APIs** consume System APIs rather than accessing data directly
4. Some services do require **local persistence** for caches, audit logs, or service-specific data

We need standardized patterns for both scenarios that integrate with our Hexagonal Light architecture (ADR-009).

### Current Challenges

- Inconsistent data access patterns across teams
- Unclear guidance on when to use JPA vs. System API delegation
- Missing resilience patterns in System API calls
- Repository implementations leaking into domain layer

## Decision

We will standardize on **two persistence patterns** based on data ownership:

### Pattern 1: JPA Persistence (Data Owner)

For services that **own their data**:

```
Domain API owns data → JPA Repository → Local Database
```

**Use when:**
- Service is the System of Record for this data
- Data is local to the service domain
- Need complex queries, joins, or aggregations
- Data lifecycle is managed by this service

**Examples:**
- Audit logging service (owns audit logs)
- Configuration service (owns config data)
- Local cache with persistence

### Pattern 2: System API Delegation (Data Consumer)

For services that **consume data** from the mainframe:

```
Domain API → Repository Interface → System API Adapter → System API → Mainframe
```

**Use when:**
- Data resides in mainframe (Z/OS, DB2)
- Service is a Domain API, not a System API
- Need transactional integrity with legacy systems
- Data lifecycle is managed by mainframe

**Examples:**
- Customer service (customer data in mainframe)
- Account service (account data in mainframe)
- Transaction service (transaction data in mainframe)

## Architecture

### JPA Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Domain API                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                         DOMAIN                                │  │
│  │  ┌─────────────┐    ┌────────────────────┐                   │  │
│  │  │   Entity    │    │    Repository      │                   │  │
│  │  │  (Domain)   │    │   (Interface)      │                   │  │
│  │  └─────────────┘    └────────────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                   ▲                                 │
│                                   │ implements                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    ADAPTER (Persistence)                      │  │
│  │  ┌─────────────┐    ┌────────────────────┐                   │  │
│  │  │  JPA Entity │    │  JPA Repository    │                   │  │
│  │  │  (Adapter)  │    │   (Spring Data)    │                   │  │
│  │  └─────────────┘    └────────────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │ JDBC
                                    ▼
                            ┌───────────────┐
                            │   Database    │
                            │  (PostgreSQL) │
                            └───────────────┘
```

**Key Points:**
- Domain entities are **pure domain objects** (no JPA annotations)
- JPA entities are in `adapter/persistence/entity/` with mapping to domain
- Repository interface is in `domain/repository/`
- JPA implementation is in `adapter/persistence/repository/`

### System API Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Domain API                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                         DOMAIN                                │  │
│  │  ┌─────────────┐    ┌────────────────────┐                   │  │
│  │  │   Entity    │    │    Repository      │                   │  │
│  │  │  (Domain)   │    │   (Interface)      │                   │  │
│  │  └─────────────┘    └────────────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                   ▲                                 │
│                                   │ implements                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   ADAPTER (System API)                        │  │
│  │  ┌─────────────┐    ┌────────────────────┐                   │  │
│  │  │  Client DTO │    │  SystemAPI Adapter │                   │  │
│  │  │  (Adapter)  │    │   (REST Client)    │                   │  │
│  │  └─────────────┘    └────────────────────┘                   │  │
│  │                              │                                │  │
│  │                     ┌────────┴────────┐                      │  │
│  │                     │  Feign Client   │                      │  │
│  │                     │  + Resilience   │                      │  │
│  │                     └─────────────────┘                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │ REST/JSON
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          System API                                 │
├─────────────────────────────────────────────────────────────────────┤
│                    (Wraps mainframe transactions)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ CICS/MQ
                                    ▼
                            ┌───────────────┐
                            │   Mainframe   │
                            │    (Z/OS)     │
                            └───────────────┘
```

**Key Points:**
- Repository interface is in `domain/repository/` (same as JPA)
- System API adapter is in `adapter/systemapi/`
- Client DTOs map to/from domain entities
- **Resilience patterns are MANDATORY** (Circuit Breaker, Retry, Timeout)

## REST Client Options

For System API calls, three client options are supported:

### Option A: Feign (Recommended)

Declarative, clean interface. Best for most cases.

```java
@FeignClient(name = "customer-system-api", url = "${system-api.customer.base-url}")
public interface CustomerSystemApiClient {
    
    @GetMapping("/customers/{id}")
    CustomerDto findById(@PathVariable String id);
    
    @PostMapping("/customers")
    CustomerDto save(@RequestBody CustomerDto customer);
}
```

**Pros:** Clean interface, automatic serialization, integrates with Resilience4j
**Cons:** Additional dependency

### Option B: RestTemplate (Legacy)

Synchronous, widely understood. For simple cases or legacy code.

```java
@Component
public class CustomerSystemApiClient {
    
    private final RestTemplate restTemplate;
    
    public CustomerDto findById(String id) {
        return restTemplate.getForObject(
            baseUrl + "/customers/{id}", 
            CustomerDto.class, 
            id
        );
    }
}
```

**Pros:** No additional dependencies, well-known
**Cons:** Verbose, imperative style

### Option C: RestClient (Modern)

New in Spring 6.1 / Boot 3.2. Fluent API, recommended for new projects.

```java
@Component
public class CustomerSystemApiClient {
    
    private final RestClient restClient;
    
    public CustomerDto findById(String id) {
        return restClient.get()
            .uri("/customers/{id}", id)
            .retrieve()
            .body(CustomerDto.class);
    }
}
```

**Pros:** Modern API, fluent style, good defaults
**Cons:** Requires Spring Boot 3.2+

### Selection Criteria

| Criteria | Feign | RestTemplate | RestClient |
|----------|-------|--------------|------------|
| **Spring Boot Version** | Any | Any | 3.2+ |
| **Style** | Declarative | Imperative | Fluent |
| **Complexity** | Low | Medium | Low |
| **Testing** | Easy (mock interface) | Medium | Medium |
| **Resilience4j** | Built-in | Manual | Manual |
| **Recommended for** | Most cases | Legacy | New projects |

## Resilience Requirements

**System API calls MUST include resilience patterns:**

```java
@Component
public class CustomerSystemApiAdapter implements CustomerRepository {
    
    private final CustomerSystemApiClient client;
    
    @CircuitBreaker(name = "customerSystemApi", fallbackMethod = "findByIdFallback")
    @TimeLimiter(name = "customerSystemApi")
    @Retry(name = "customerSystemApi")
    public CompletableFuture<Customer> findById(String id) {
        return CompletableFuture.supplyAsync(() -> {
            CustomerDto dto = client.findById(id);
            return mapper.toDomain(dto);
        });
    }
    
    private CompletableFuture<Customer> findByIdFallback(String id, Exception ex) {
        // Fallback strategy: cache, default, or error
    }
}
```

**Minimum required patterns:**
- `@CircuitBreaker` - Prevent cascade failures
- `@Retry` - Handle transient failures
- `@TimeLimiter` - Prevent hanging calls (for async methods)

See ADR-004 for detailed resilience pattern guidance.

## Project Structure

### JPA Pattern Structure

```
src/main/java/com/company/{service}/
├── domain/
│   ├── model/
│   │   └── Customer.java              # Domain entity (no JPA)
│   └── repository/
│       └── CustomerRepository.java    # Repository interface
│
└── adapter/
    └── persistence/
        ├── entity/
        │   └── CustomerJpaEntity.java # JPA entity
        ├── repository/
        │   └── CustomerJpaRepository.java  # Spring Data JPA
        ├── mapper/
        │   └── CustomerPersistenceMapper.java
        └── CustomerPersistenceAdapter.java # Implements domain repo
```

### System API Pattern Structure

```
src/main/java/com/company/{service}/
├── domain/
│   ├── model/
│   │   └── Customer.java              # Domain entity
│   └── repository/
│       └── CustomerRepository.java    # Repository interface
│
└── adapter/
    └── systemapi/
        ├── client/
        │   └── CustomerSystemApiClient.java  # Feign/RestTemplate/RestClient
        ├── dto/
        │   └── CustomerDto.java       # API contract DTO
        ├── mapper/
        │   └── CustomerSystemApiMapper.java
        └── CustomerSystemApiAdapter.java # Implements domain repo
```

## Configuration

### JPA Configuration

```yaml
spring:
  datasource:
    url: jdbc:postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME}
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
    hikari:
      maximum-pool-size: 10
      minimum-idle: 5
      connection-timeout: 30000
      
  jpa:
    hibernate:
      ddl-auto: validate  # MUST be 'validate' in production
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        format_sql: true
    show-sql: false
```

### System API Client Configuration

```yaml
system-api:
  customer:
    base-url: ${SYSTEM_API_CUSTOMER_URL:http://localhost:8081}
    
feign:
  client:
    config:
      customer-system-api:
        connectTimeout: 5000
        readTimeout: 10000
        loggerLevel: basic
        
resilience4j:
  circuitbreaker:
    instances:
      customerSystemApi:
        slidingWindowSize: 10
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        
  retry:
    instances:
      customerSystemApi:
        maxAttempts: 3
        waitDuration: 500ms
        enableExponentialBackoff: true
        
  timelimiter:
    instances:
      customerSystemApi:
        timeoutDuration: 5s
```

## Consequences

### Positive

- **Clear separation** between data ownership patterns
- **Consistent structure** across all services
- **Resilience built-in** for System API calls
- **Testability** improved with interface-based repositories
- **Domain isolation** maintained (no JPA in domain layer)

### Negative

- **Mapping overhead** between domain and persistence/DTO layers
- **Additional complexity** for simple CRUD operations
- **Learning curve** for teams unfamiliar with Hexagonal architecture

### Neutral

- **Technology choice** (Feign vs RestTemplate vs RestClient) requires team decision
- **Migration effort** for existing services to adopt new patterns

## Compliance

### MUST

- Repository interfaces MUST be in `domain/repository/`
- JPA entities MUST NOT be in domain layer
- System API calls MUST have resilience patterns
- Base URLs MUST be externalized (environment variables)

### SHOULD

- Domain entities SHOULD NOT have framework annotations
- System API adapters SHOULD use Feign for new projects
- JPA configuration SHOULD use `ddl-auto: validate` in production

### MAY

- Services MAY use hybrid pattern (JPA cache + System API source)
- Services MAY use RestClient for Spring Boot 3.2+ projects

## Related Decisions

- **ADR-009:** Hexagonal Light architecture (adapter placement)
- **ADR-004:** Resilience patterns (required for System API calls)
- **ADR-001:** API design standards (contract definitions)

## References

- [Spring Data JPA Documentation](https://spring.io/projects/spring-data-jpa)
- [Spring Cloud OpenFeign](https://spring.io/projects/spring-cloud-openfeign)
- [Resilience4j Documentation](https://resilience4j.readme.io/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)

---

## Changelog

### v1.0 (2025-11-28)
- Initial version
- Defined JPA and System API persistence patterns
- Documented REST client options (Feign, RestTemplate, RestClient)
- Integrated with Hexagonal Light architecture (ADR-009)
- Mandated resilience patterns for System API calls
