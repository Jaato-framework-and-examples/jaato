---
id: adr-009-service-architecture-patterns
title: "ADR-009: Service Architecture Patterns"
sidebar_label: Service Architecture Patterns
version: 2.0
date: 2025-11-19
updated: 2025-11-24
status: Accepted
author: Architecture Team
framework: agnostic
architecture_styles:
  - hexagonal-light
  - full-hexagonal
  - traditional-layered
tags:
  - architecture
  - hexagonal
  - clean-architecture
  - testing
  - microservices
  - domain-driven-design
related:
  - adr-001-api-design-standards
  - adr-004-resilience-patterns
  - eri-code-001-hexagonal-light-java-spring
  - skill-code-020-generate-microservice-java-spring
---

# ADR-009: Service Architecture Patterns

**Status:** Accepted  
**Date:** 2025-11-19  
**Updated:** 2025-11-24 (v2.0 - Framework-agnostic)  
**Deciders:** Architecture Team

---

## Context

### Current Situation

Our organization develops and maintains 400+ microservices with the following challenges:

1. **Tightly coupled to frameworks**
   - Business logic embedded in framework-annotated classes
   - Difficult to unit test without framework context
   - Framework upgrades are painful and risky
   - Vendor lock-in concerns

2. **Testing challenges**
   - Heavy reliance on integration tests (slow)
   - Test suites taking 15-30 minutes
   - Flaky tests due to external dependencies
   - Low developer productivity

3. **Inconsistent architecture**
   - Each team implements differently
   - Mix of patterns: layered, transaction script, anemic domain
   - Difficult knowledge transfer
   - High code review overhead

4. **Future flexibility concerns**
   - Potential framework migrations
   - Multi-channel support (REST, gRPC, messaging)
   - Domain logic reusability across contexts

### Business Context

- **Team size:** 400+ developers across multiple squads
- **Services:** 400+ microservices in production
- **Skill levels:** Junior to Senior (need clear patterns)
- **Domain complexity:** Ranges from simple CRUD to complex business logic

### Problem Statement

We need an architectural pattern that:
- ✅ Enables fast unit testing of business logic
- ✅ Reduces framework coupling
- ✅ Maintains simplicity for straightforward services
- ✅ Scales to complex domain logic when needed
- ✅ Is adoptable by developers of all levels
- ✅ Provides clear boundaries and responsibilities

---

## Decision

We adopt **"Hexagonal Light" architecture as the default pattern** for new microservices, with flexibility for simpler or more complex cases based on defined criteria.

---

## Architecture Styles

### Style 1: Hexagonal Light (DEFAULT)

**When to use:**
- ✅ Standard business services (majority of cases)
- ✅ Services with 3-10 business rules
- ✅ Need good testability
- ✅ Expected to evolve over time

**Structure:**

```
src/main/java/{basePackage}/
├── domain/                          # DOMAIN LAYER (Pure POJOs)
│   ├── model/                       # Domain entities and value objects
│   │   ├── {Entity}.java           # Domain entity (POJO)
│   │   ├── {Entity}Id.java         # Value object for ID
│   │   └── {Enum}.java             # Domain enums
│   ├── service/                     # Domain services (POJOs - NO framework)
│   │   └── {Entity}DomainService.java
│   ├── repository/                  # Repository interfaces (ports)
│   │   └── {Entity}Repository.java
│   └── exception/                   # Domain exceptions
│       └── {Entity}NotFoundException.java
│
├── application/                     # APPLICATION LAYER (Framework integration)
│   └── service/
│       └── {Entity}ApplicationService.java  # @Service, @Transactional
│
├── adapter/                         # ADAPTER LAYER (Framework-specific)
│   ├── rest/                        # REST adapter (driving/in)
│   │   ├── controller/
│   │   │   └── {Entity}Controller.java
│   │   ├── dto/
│   │   │   ├── {Entity}DTO.java
│   │   │   ├── Create{Entity}Request.java
│   │   │   └── Update{Entity}Request.java
│   │   └── mapper/
│   │       └── {Entity}DtoMapper.java
│   │
│   └── persistence/                 # Persistence adapter (driven/out)
│       ├── entity/
│       │   └── {Entity}Entity.java
│       ├── repository/
│       │   └── {Entity}JpaRepository.java
│       ├── adapter/
│       │   └── {Entity}RepositoryAdapter.java
│       └── mapper/
│           └── {Entity}EntityMapper.java
│
└── infrastructure/                  # INFRASTRUCTURE (Configuration)
    ├── config/
    │   └── {Entity}Config.java      # Bean wiring
    └── exception/
        ├── GlobalExceptionHandler.java
        └── ErrorResponse.java
```

**Key Characteristics:**
- **Domain layer is pure POJOs** - No framework annotations
- **Application layer bridges** domain and adapters
- **Adapters contain** all framework-specific code
- **Fast unit testing** of domain layer (no framework needed)
- **Clear separation** of concerns

**Dependency Direction:**

```
┌─────────────────────────────────────────────────────────────────┐
│                         ADAPTERS                                 │
│  (REST controllers, JPA repositories, messaging, etc.)          │
│  Framework-specific code lives here                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ depends on
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       APPLICATION                                │
│  (Application services - orchestration, transactions)            │
│  Thin layer that coordinates domain and adapters                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ depends on
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                          DOMAIN                                  │
│  (Business logic, domain services, entities)                    │
│  Pure POJOs - NO framework dependencies                         │
│  Defines repository interfaces (ports)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

### Style 2: Full Hexagonal (COMPLEX CASES)

**When to use:**
- ✅ Complex domain logic (>10 business rules)
- ✅ Multiple adapters (REST + Kafka + gRPC)
- ✅ Critical business services
- ✅ Planned framework migration
- ✅ Domain-driven design approach

**Additional structure:**

```
domain/
├── model/
├── port/
│   ├── in/                          # Driving ports (use cases)
│   │   └── RegisterCustomerUseCase.java
│   └── out/                         # Driven ports (repositories, external)
│       ├── CustomerRepository.java
│       └── NotificationPort.java
└── service/
    └── CustomerService.java         # Implements use cases
```

**Characteristics:**
- Explicit use case interfaces (driving ports)
- Explicit repository/external interfaces (driven ports)
- Maximum decoupling from framework
- Higher code overhead but maximum flexibility

---

### Style 3: Traditional Layered (SIMPLE CRUD)

**When to use:**
- ✅ Pure CRUD operations (<3 business rules)
- ✅ Simple proxy/BFF services
- ✅ Short-lived services or prototypes
- ✅ Very simple data transformations

**Structure:**

```
src/main/java/{basePackage}/
├── controller/
│   └── {Entity}Controller.java
├── service/
│   └── {Entity}Service.java         # @Service with business logic
├── repository/
│   └── {Entity}Repository.java
└── dto/
    └── {Entity}DTO.java
```

**Characteristics:**
- Business logic in @Service classes
- Direct coupling to framework
- Simpler structure, less code
- Acceptable for truly simple services

---

## Decision Matrix

| Criteria | Traditional | Hexagonal Light | Full Hexagonal |
|----------|-------------|-----------------|----------------|
| **Business Rules** | 0-2 | 3-10 | 10+ |
| **Adapters** | 1 (REST only) | 1-2 | 3+ |
| **Domain Complexity** | Low | Medium | High |
| **Expected Lifespan** | <1 year | 1-5 years | 5+ years |
| **Team Size** | 1-2 devs | 2-5 devs | 5+ devs |
| **Framework Migration Risk** | Low | Medium | High |
| **Code Overhead** | 0% | +30% | +60% |
| **Testability** | ★☆☆ | ★★★ | ★★★ |
| **Framework Independence** | ★☆☆ | ★★☆ | ★★★ |
| **Learning Curve** | ★☆☆ | ★★☆ | ★★★ |

**Default choice: Hexagonal Light** (optimal trade-off for most services)

---

## Rationale

### Why Hexagonal Light as Default?

#### 1. Fast Unit Testing

**Problem:** Current tests are slow (15-30 minutes) because they require framework context.

**Solution with Hexagonal Light:**
```
Domain layer tests:
- Pure POJO tests
- No framework startup
- Run in milliseconds
- 40-60x faster than integration tests
```

**Impact:**
- Test execution: 15 min → 2 min
- Developer productivity: +30%
- Better TDD experience

#### 2. Framework Independence

**Problem:** Framework upgrades are painful (e.g., Spring 5→6 took 6 months).

**Solution:**
- Domain logic has zero framework dependencies
- Framework code isolated in adapters
- Can evaluate alternatives (Quarkus, Micronaut)

**Impact:**
- Framework upgrade impact reduced by 60%
- Domain logic portable across frameworks

#### 3. Clear Boundaries

**Problem:** Business logic scattered across layers, hard to find and maintain.

**Solution:**
```
domain/          ← Business logic HERE (easy to find)
application/     ← Orchestration
adapter/         ← Framework details
```

**Impact:**
- Onboarding time: -40%
- Code review efficiency: +50%
- Easier knowledge transfer

#### 4. Pragmatic Balance

Hexagonal Light provides the **optimal trade-off**:
- More testable than Traditional
- Less complex than Full Hexagonal
- Suitable for 80% of our services
- Adoptable by developers of all levels

### Why NOT Full Hexagonal as Default?

1. **Overhead for simple services** - 60% more code for simple CRUD
2. **Cognitive load** - Too complex for junior developers
3. **Diminishing returns** - Most services don't need maximum decoupling

### Why NOT Keep Traditional?

1. **Testing bottleneck** - Current test suites are too slow
2. **Framework coupling** - Painful migrations
3. **Lack of standards** - Inconsistency across teams

---

## Consequences

### Positive

- ✅ Fast development cycle (quick feedback from tests)
- ✅ Better code quality (clear separation of concerns)
- ✅ Framework flexibility (easier upgrades and migrations)
- ✅ Knowledge sharing (consistent structure across services)
- ✅ Testing culture (fast tests encourage more testing)
- ✅ Domain logic explicit and easy to find

### Negative

- ⚠️ Learning curve for teams new to hexagonal concepts
- ⚠️ ~30% more code than traditional approach
- ⚠️ More files and directories to navigate
- ⚠️ Migration effort for existing services

### Mitigations

1. **Training & Documentation** - Comprehensive examples (ERIs)
2. **Gradual Adoption** - New services use Hexagonal Light, existing migrate opportunistically
3. **Automation** - Code generators (skills) and templates
4. **Support** - Architecture guild for questions

---

## Implementation

### Technology-Specific ERIs

| Framework | ERI | Description |
|-----------|-----|-------------|
| Java/Spring | eri-code-001-hexagonal-light-java-spring | Base Hexagonal Light template |
| Java/Spring | eri-002-domain-api-java-spring | Domain API with constraints |
| NodeJS (future) | eri-050-hexagonal-light-nodejs | NodeJS implementation |
| Quarkus (future) | eri-100-hexagonal-light-quarkus | Quarkus implementation |

### Automated Skills

| Skill | Purpose |
|-------|---------|
| skill-code-020-generate-microservice-java-spring | Generate Hexagonal Light microservice |

### Success Metrics

| Metric | Baseline | Target (6 months) |
|--------|----------|-------------------|
| Unit test execution time | 15-30 min | 2-5 min |
| Integration test ratio | 70% | 30% |
| Code coverage | 60% | 80% |
| Framework coupling (LOC) | 40% | 15% |
| Developer satisfaction | 6/10 | 8/10 |
| Onboarding time | 4 weeks | 2.5 weeks |

---

## Validation

### Success Criteria

- ✅ Test suite execution < 5 minutes
- ✅ Domain logic testable without framework
- ✅ Clear architectural boundaries
- ✅ 80%+ adoption for new services

### Compliance Checks

- Automated structure validation
- Layer boundary checks (no framework in domain)
- Naming convention enforcement

---

## References

### Related ADRs
- **ADR-001:** API Design Standards
- **ADR-004:** Resilience Patterns
- **ADR-006:** Error Handling Standards

### External Resources
- [Hexagonal Architecture (Alistair Cockburn)](https://alistair.cockburn.us/hexagonal-architecture/)
- [Clean Architecture (Robert Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Growing Object-Oriented Software, Guided by Tests](http://www.growing-object-oriented-software.com/)

---

## Notes

### Migration Strategy for Existing Services

**Don't migrate if:**
- Service is stable (< 5 changes/year)
- Pure CRUD with no business logic
- Service being decommissioned

**Do migrate if:**
- High churn (> 10 changes/year)
- Complex business logic
- Testing pain points
- Framework upgrade planned

### Exceptions

Some services may use Traditional architecture:
- Internal tools
- One-off utilities
- Prototypes
- Services with <3 month lifespan

Requires architecture review approval.

---

## Changelog

### v2.0 (2025-11-24)
- Made framework-agnostic (moved Java specifics to ERIs)
- Clarified decision matrix criteria
- Added rationale section
- Referenced new ERIs and Skills

### v1.0 (2025-11-19)
- Initial version with three architecture styles
- Established Hexagonal Light as default

---

**Decision Status:** ✅ Accepted and Active  
**Review Date:** Q2 2025
