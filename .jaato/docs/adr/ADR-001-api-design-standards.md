---
id: adr-001-api-design-standards
title: "ADR-001: API Design Standards"
sidebar_label: API Design Standards
version: 2.0
date: 2025-05-27
updated: 2025-11-24
status: Accepted
author: Architecture Team
framework: agnostic
api_layers:
  - experience-api
  - composable-api
  - domain-api
  - system-api
tags:
  - api
  - architecture
  - microservices
  - api-led-connectivity
  - transaction
  - saga
  - orchestration
related:
  - adr-009-service-architecture-patterns
  - eri-code-001-hexagonal-light-java-spring
---

# ADR-001: API Design Standards

**Status:** Accepted
**Date:** 2025-05-27
**Updated:** 2025-11-24 (v2.0 - Framework-agnostic)
**Deciders:** Architecture Team

---

## Context

In our microservices-based application architecture, we need a consistent approach to API design that enables:

- Reliable orchestration of complex business workflows spanning multiple domains
- Clear separation of concerns between orchestration, business logic, and integrations
- Reusability and consistency across a multinational, distributed organization
- Flexible, resilient, and maintainable transaction management

**Problems we face:**
- Inconsistent API structures across teams
- Unclear boundaries between API responsibilities
- Difficulty coordinating multi-domain transactions
- Tight coupling between UI requirements and business logic
- Complex integration management with Systems of Record (SoR)

**Business impact:**
- Slower time-to-market for new features
- Higher maintenance costs
- Difficulty scaling across locations
- Increased risk of cascade failures

---

## Decision

We adopt a **4-layer API architecture** (API-led Connectivity) with clear responsibilities per layer.

### API Layer Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIENCE / BFF LAYER                       │
│  Purpose: Channel-specific data transformation and optimization  │
│  Consumers: UI applications (Web, Mobile, etc.)                  │
│  Calls: Composable APIs                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       COMPOSABLE API LAYER                       │
│  Purpose: Orchestration of multi-domain workflows                │
│  Responsibility: Transaction management (SAGA pattern)           │
│  Calls: Multiple Domain APIs                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DOMAIN API LAYER                          │
│  Purpose: Atomic business capabilities per domain                │
│  Responsibility: Business logic, data ownership                  │
│  Calls: System APIs (within same domain only)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM API LAYER                          │
│  Purpose: Abstraction of Systems of Record (SoR)                 │
│  Responsibility: Integration contracts, data transformation      │
│  Calls: External systems, databases, third-party APIs            │
└─────────────────────────────────────────────────────────────────┘
```

---

### Layer 1: Experience API (BFF)

**Purpose:** Interface between UI and backend, optimized for specific channels.

**Characteristics:**
- Channel-specific (Mobile BFF, Web BFF, etc.)
- Data aggregation and transformation for UI needs
- No business logic
- Caching for UI performance
- Authentication/session management

**Constraints:**
- ✅ CAN call Composable APIs
- ✅ CAN call Domain APIs directly (simple read operations)
- ❌ CANNOT call System APIs directly
- ❌ CANNOT implement business logic
- ❌ CANNOT own data

---

### Layer 2: Composable API

**Purpose:** Orchestration of multi-domain business workflows.

**Characteristics:**
- Coordinates calls to multiple Domain APIs
- Implements transaction management (SAGA pattern)
- Manages compensation logic for failures
- Stateless orchestration (state in Domain APIs)

**Constraints:**
- ✅ CAN call multiple Domain APIs
- ✅ CAN implement SAGA orchestration
- ✅ CAN manage cross-domain transactions
- ❌ CANNOT call System APIs directly
- ❌ CANNOT own data (delegates to Domain APIs)
- ❌ CANNOT bypass Domain APIs

---

### Layer 3: Domain API

**Purpose:** Atomic business capabilities for a specific business domain (bounded context).

**Characteristics:**
- Domain API owns its data (single source of truth for the domain)
- Implements domain business logic
- Provides CRUD + business operations
- Exposes compensation endpoints for SAGA

**Constraints:**
- ✅ CAN call System APIs (within same domain)
- ✅ CAN call domain microservices (within same bounded context)
- ✅ CAN implement business rules
- ✅ CAN own and manage domain data
- ❌ CANNOT call Domain APIs from **other** domains (use Composable layer)
- ❌ CANNOT call Composable APIs
- ❌ CANNOT call Experience APIs

---

### Layer 4: System API

**Purpose:** Abstraction layer for Systems of Record (SoR) and external integrations.

**Characteristics:**
- Standardizes access to backend systems
- Transforms data between SoR and domain models
- Handles integration complexity (protocols, formats)
- Provides consistent contracts regardless of SoR variations

**Constraints:**
- ✅ CAN call external systems, databases, third-party APIs
- ✅ CAN transform data formats
- ✅ CAN handle integration protocols
- ❌ CANNOT implement business logic
- ❌ CANNOT be called by Experience or Composable APIs directly

---

## Technical Standards

### API Contract Standards

| Aspect | Standard |
|--------|----------|
| **Specification** | OpenAPI 3.0+ |
| **Versioning** | URL path versioning (`/api/v1/`, `/api/v2/`) |
| **Naming** | RESTful conventions, kebab-case for paths |
| **Methods** | Standard HTTP verbs (GET, POST, PUT, PATCH, DELETE) |
| **Status Codes** | Standard HTTP status codes |

### Error Response Format

```json
{
  "timestamp": "2025-11-24T10:15:30.123Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Validation failed",
  "path": "/api/v1/customers",
  "correlationId": "abc-123-def-456",
  "details": [
    {
      "field": "email",
      "message": "must be a valid email address"
    }
  ]
}
```

---

## Consequences

### Positive

- ✅ Clear separation of concerns across layers
- ✅ Reusable Domain APIs across multiple Composable flows
- ✅ Consistent transaction management via SAGA
- ✅ Decoupled UI from business logic
- ✅ Standardized integration contracts

### Negative

- ⚠️ More layers = more network hops (latency)
- ⚠️ Increased complexity for simple use cases
- ⚠️ Requires discipline to maintain layer boundaries

---

**Decision Status:** ✅ Accepted and Active
**Review Date:** Q2 2025
