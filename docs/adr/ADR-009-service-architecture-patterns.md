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
related:
  - adr-001-api-design-standards
  - adr-004-resilience-patterns
  - eri-code-001-hexagonal-light-java-spring
---

# ADR-009: Service Architecture Patterns

**Status:** Accepted
**Date:** 2025-11-19
**Updated:** 2025-11-24 (v2.0 - Framework-agnostic)
**Deciders:** Architecture Team

---

## Context

Our organization develops and maintains 400+ microservices with the following challenges:

1. **Tightly coupled to frameworks** - Business logic embedded in framework-annotated classes
2. **Testing challenges** - Heavy reliance on integration tests (slow)
3. **Inconsistent architecture** - Each team implements differently
4. **Future flexibility concerns** - Potential framework migrations

---

## Decision

We adopt **"Hexagonal Light" architecture as the default pattern** for new microservices.

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
│   ├── service/                     # Domain services (POJOs - NO framework)
│   ├── repository/                  # Repository interfaces (ports)
│   └── exception/                   # Domain exceptions
│
├── application/                     # APPLICATION LAYER (Framework integration)
│   └── service/                     # @Service, @Transactional
│
├── adapter/                         # ADAPTER LAYER (Framework-specific)
│   ├── rest/                        # REST adapter (driving/in)
│   └── persistence/                 # Persistence adapter (driven/out)
│
└── infrastructure/                  # INFRASTRUCTURE (Configuration)
    ├── config/                      # Bean wiring
    └── exception/                   # Global exception handling
```

**Key Characteristics:**
- **Domain layer is pure POJOs** - No framework annotations
- **Application layer bridges** domain and adapters
- **Adapters contain** all framework-specific code
- **Fast unit testing** of domain layer (no framework needed)

---

### Style 2: Full Hexagonal (COMPLEX CASES)

**When to use:**
- ✅ Complex domain logic (>10 business rules)
- ✅ Multiple adapters (REST + Kafka + gRPC)
- ✅ Critical business services

### Style 3: Traditional Layered (SIMPLE CRUD)

**When to use:**
- ✅ Pure CRUD operations (<3 business rules)
- ✅ Simple proxy/BFF services
- ✅ Short-lived services or prototypes

---

## Decision Matrix

| Criteria | Traditional | Hexagonal Light | Full Hexagonal |
|----------|-------------|-----------------|----------------|
| **Business Rules** | 0-2 | 3-10 | 10+ |
| **Code Overhead** | 0% | +30% | +60% |
| **Testability** | ★☆☆ | ★★★ | ★★★ |
| **Framework Independence** | ★☆☆ | ★★☆ | ★★★ |

**Default choice: Hexagonal Light**

---

## Consequences

### Positive
- ✅ Fast development cycle (quick feedback from tests)
- ✅ Better code quality (clear separation of concerns)
- ✅ Framework flexibility (easier upgrades and migrations)
- ✅ Domain logic explicit and easy to find

### Negative
- ⚠️ Learning curve for teams new to hexagonal concepts
- ⚠️ ~30% more code than traditional approach

---

**Decision Status:** ✅ Accepted and Active
**Review Date:** Q2 2025
