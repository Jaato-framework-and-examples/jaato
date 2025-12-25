---
id: adr-001-api-design
title: "ADR-001: API Design - Fusion API Model, Types & Standards"
sidebar_label: Fusion API Model
version: 3.1
date: 2025-05-27
updated: 2025-12-22
status: Accepted
author: Architecture Team
decision_type: pattern
scope: organization
tags:
  - api
  - architecture
  - microservices
  - fusion-api-model
  - api-led-connectivity
  - rest
  - grpc
  - async-api
related:
  - adr-009-service-architecture-patterns
  - adr-004-resilience-patterns
  - adr-013-distributed-transactions
implemented_by:
  - eri-code-001-hexagonal-light-java-spring
  - eri-code-014-api-public-exposure-java-spring
---

# ADR-001: API Design - Fusion API Model, Types & Standards

**Status:** Accepted  
**Date:** 2025-05-27  
**Updated:** 2025-12-22 (v3.1 - Named model as "Fusion API Model", added inference rules)  
**Deciders:** Architecture Team

---

## Context

In our microservices-based application architecture, we need a consistent approach to API design that enables:

- Reliable orchestration of complex business workflows spanning multiple domains
- Clear separation of concerns between orchestration, business logic, and integrations
- Reusability and consistency across a multinational, distributed organization
- Flexible, resilient, and maintainable transaction management
- Support for multiple API technologies (REST, gRPC, async events)

**Problems we face:**
- Inconsistent API structures across teams
- Unclear boundaries between API responsibilities
- Difficulty coordinating multi-domain transactions
- Tight coupling between UI requirements and business logic
- Complex integration management with Systems of Record (SoR)
- No clear standards for pagination, filtering, or hypermedia

**Business impact:**
- Slower time-to-market for new features
- Higher maintenance costs
- Difficulty scaling across locations
- Increased risk of cascade failures
- Poor developer experience consuming APIs

---

## Decision

We adopt the **Fusion API Model**: a **4-layer API architecture** with support for **multiple API types** and **standardized patterns per type**.

> **Naming Convention:** The term "Fusion" identifies our opinionated API model. When referencing 
> APIs that follow this architecture, use the prefix "Fusion" to distinguish them from generic 
> API terminology (e.g., "Fusion Domain API" instead of just "Domain API").

---

## Part 1: Fusion API Layer Model

### Model Overview

The **Fusion API Model** defines four distinct API layers, each with specific responsibilities and constraints. This model enables consistent API design across the organization while supporting the complexity of enterprise integrations.

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIENCE / BFF LAYER                       │
│  Purpose: Channel-specific data transformation and optimization  │
│  Consumers: UI applications (Web, Mobile, etc.)                  │
│  Calls: Composable APIs, Domain APIs (simple reads)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       COMPOSABLE API LAYER                       │
│  Purpose: Orchestration of multi-domain workflows                │
│  Responsibility: Transaction coordination (see ADR-013)          │
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
- Implements transaction coordination (see ADR-013)
- Manages compensation logic for failures
- Stateless orchestration (state in Domain APIs)

**Constraints:**
- ✅ CAN call multiple Domain APIs
- ✅ CAN implement orchestration patterns
- ✅ CAN manage cross-domain transactions
- ❌ CANNOT call System APIs directly
- ❌ CANNOT own data (delegates to Domain APIs)
- ❌ CANNOT bypass Domain APIs

**Example use cases:**
- Order creation: validate customer → reserve inventory → process payment → create order
- Account opening: KYC check → create account → setup products → send notifications

---

### Layer 3: Domain API

**Purpose:** Atomic business capabilities for a specific business domain (bounded context).

#### Domain API vs Domain Microservices

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUSTOMER DOMAIN (Bounded Context)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │       DOMAIN API                     │  ← Exposes domain    │
│   │   (customer-management-api)          │    capabilities      │
│   │                                      │    externally        │
│   │   - Aggregate root operations        │                      │
│   │   - External contract (OpenAPI)      │                      │
│   │   - Called by Composable APIs        │                      │
│   └──────────────┬──────────────────────┘                       │
│                  │ internal calls                                │
│                  ↓                                               │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │           DOMAIN MICROSERVICES (internal)                 │  │
│   │                                                           │  │
│   │   ┌─────────────────┐    ┌─────────────────┐             │  │
│   │   │ customer-events │    │ customer-search │             │  │
│   │   │   (async jobs)  │    │  (read model)   │             │  │
│   │   └─────────────────┘    └─────────────────┘             │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Domain API owns its data (single source of truth)
- Implements domain business logic
- Provides CRUD + business operations
- Exposes compensation endpoints when participates in distributed transactions

**Constraints:**
- ✅ CAN call System APIs (within same domain)
- ✅ CAN call domain microservices (within same bounded context)
- ✅ CAN implement business rules
- ✅ CAN own and manage domain data
- ❌ CANNOT call Domain APIs from other domains (use Composable layer)
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

### Fusion API Identification

The 4-layer API architecture defined in this ADR is named the **Fusion API Model**. This naming convention helps distinguish our opinionated API patterns from generic industry terminology.

#### Naming Convention

When referencing APIs that follow this architecture, use the prefix **"Fusion"** to explicitly indicate adherence to this model:

- "Fusion Domain API" (not just "Domain API")
- "Fusion System API" (not just "System API")
- "Fusion Experience API" or "Fusion BFF"
- "Fusion Composable API"

This explicit naming avoids ambiguity with common industry terms like "Domain API" or "BFF" which may have different meanings in other contexts.

#### Fusion API vs Internal Microservices

| Type | Description | API Exposure |
|------|-------------|--------------|
| **Fusion API** | APIs following the 4-layer model | Public exposure with HATEOAS, pagination |
| **Internal Microservice** | Services within a bounded context | No public exposure patterns |

#### Layer-Feature Matrix

When implementing a Fusion API, the following features apply based on layer:

| Fusion API Layer | HATEOAS | Pagination | Compensation |
|------------------|---------|------------|--------------|
| Experience/BFF   | ✅ Required | ✅ Required | ❌ N/A |
| Composable       | ❌ Optional | ✅ Required | ❌ N/A |
| Domain           | ✅ Required | ✅ Required | ⚠️ Opt-in |
| System           | ❌ Optional | ✅ Required | ❌ N/A |

> **Note:** Operational rules for identifying Fusion APIs from user requests are defined in the 
> runtime discovery configuration (`runtime/discovery/skill-index.yaml`), not in this ADR.

---

## Part 2: API Types

### Supported API Types

| Type | Protocol | Communication | Primary Use |
|------|----------|---------------|-------------|
| **REST** | HTTP/HTTPS | Synchronous | External APIs, CRUD operations |
| **gRPC** | HTTP/2 | Synchronous | Internal APIs, high-performance |
| **AsyncAPI** | AMQP/Kafka | Asynchronous | Events, eventual consistency |

### Layer × Type Matrix

| Layer | REST | gRPC | AsyncAPI |
|-------|------|------|----------|
| **Experience (BFF)** | ✅ Primary | ⚠️ Rare | ❌ Not applicable |
| **Composable** | ✅ Common | ✅ Performance | ⚠️ Choreography |
| **Domain** | ✅ External contract | ✅ Internal | ✅ Domain events |
| **System** | ✅ Legacy wrap | ⚠️ Rare | ✅ CDC/Events |

**Legend:**
- ✅ Recommended/Common
- ⚠️ Valid but less common
- ❌ Not applicable

### Type Selection Criteria

| Criterion | REST | gRPC | AsyncAPI |
|-----------|------|------|----------|
| Browser clients | ✅ Native | ❌ Requires proxy | ❌ WebSocket |
| Low latency | ⚠️ Acceptable | ✅ Optimal | ⚠️ Variable |
| Strong contracts | ✅ OpenAPI | ✅ Proto | ✅ AsyncAPI spec |
| Streaming | ⚠️ SSE/WebSocket | ✅ Native | ✅ Native |
| Tooling maturity | ✅ Excellent | ✅ Good | ⚠️ Evolving |

---

## Part 3: REST API Standards

> These standards apply to all REST APIs. Other types (gRPC, AsyncAPI) will have their own standards sections.

### 3.1 Contract Standards

| Aspect | Standard |
|--------|----------|
| **Specification** | OpenAPI 3.0+ |
| **Versioning** | URL path versioning (`/api/v1/`, `/api/v2/`) |
| **Naming** | kebab-case for paths, camelCase for JSON properties |
| **Methods** | Standard HTTP verbs (GET, POST, PUT, PATCH, DELETE) |
| **Content-Type** | `application/json` (default), support for `application/problem+json` |

### 3.2 Resource Design

#### URL Structure

```
/{api-version}/{resource-collection}/{resource-id}/{sub-resource}
```

**Examples:**
- `GET /api/v1/customers` - List customers
- `GET /api/v1/customers/123` - Get customer by ID
- `GET /api/v1/customers/123/orders` - Get customer's orders
- `POST /api/v1/customers` - Create customer
- `PUT /api/v1/customers/123` - Replace customer
- `PATCH /api/v1/customers/123` - Partial update
- `DELETE /api/v1/customers/123` - Delete customer

#### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Path segments | kebab-case, plural nouns | `/api/v1/order-items` |
| Query parameters | camelCase | `?pageSize=20&sortBy=createdAt` |
| JSON properties | camelCase | `{ "firstName": "John" }` |
| Headers | Kebab-Case | `X-Correlation-ID` |

### 3.3 Pagination

All collection endpoints MUST support pagination.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 0 | Zero-based page number |
| `size` | integer | 20 | Items per page (max 100) |
| `sort` | string | - | Sort field and direction: `field,asc` or `field,desc` |

**Example:**
```
GET /api/v1/customers?page=0&size=20&sort=lastName,asc
```

#### Response Structure

```json
{
  "content": [
    { "id": "123", "firstName": "John", "lastName": "Doe" },
    { "id": "456", "firstName": "Jane", "lastName": "Smith" }
  ],
  "page": {
    "number": 0,
    "size": 20,
    "totalElements": 142,
    "totalPages": 8
  },
  "_links": {
    "self": { "href": "/api/v1/customers?page=0&size=20" },
    "next": { "href": "/api/v1/customers?page=1&size=20" },
    "last": { "href": "/api/v1/customers?page=7&size=20" }
  }
}
```

### 3.4 HATEOAS (Hypermedia)

APIs exposed for external consumption SHOULD include hypermedia links.

#### When to Use HATEOAS

| API Type | HATEOAS | Rationale |
|----------|---------|-----------|
| Public APIs | ✅ Required | Discoverability, client decoupling |
| Partner APIs | ✅ Recommended | Reduces integration documentation |
| Internal APIs | ⚠️ Optional | May add overhead without benefit |

#### Link Structure

```json
{
  "id": "123",
  "firstName": "John",
  "lastName": "Doe",
  "status": "ACTIVE",
  "_links": {
    "self": { "href": "/api/v1/customers/123" },
    "orders": { "href": "/api/v1/customers/123/orders" },
    "update": { "href": "/api/v1/customers/123", "method": "PUT" },
    "deactivate": { "href": "/api/v1/customers/123/deactivate", "method": "POST" }
  }
}
```

#### Link Relations

| Relation | Description |
|----------|-------------|
| `self` | Link to current resource |
| `collection` | Link to parent collection |
| `next`, `prev` | Pagination navigation |
| `first`, `last` | Pagination boundaries |
| `{action}` | Available actions (domain-specific) |

### 3.5 Filtering and Sorting

#### Filtering

Support filtering via query parameters:

```
GET /api/v1/customers?status=ACTIVE&country=ES
GET /api/v1/orders?createdAfter=2025-01-01&status=PENDING
```

**Operators (optional advanced filtering):**

| Operator | Example | Description |
|----------|---------|-------------|
| equals | `status=ACTIVE` | Exact match (default) |
| like | `name=like:John*` | Pattern matching |
| gt, gte | `amount=gt:100` | Greater than |
| lt, lte | `createdAt=lt:2025-01-01` | Less than |
| in | `status=in:ACTIVE,PENDING` | Multiple values |

#### Sorting

```
GET /api/v1/customers?sort=lastName,asc
GET /api/v1/customers?sort=createdAt,desc&sort=lastName,asc
```

### 3.6 Error Handling

#### HTTP Status Codes

| Status Code | Usage |
|-------------|-------|
| **200** | Successful GET, PUT, PATCH |
| **201** | Successful POST (resource created) |
| **204** | Successful DELETE (no content) |
| **400** | Bad Request (validation errors) |
| **401** | Unauthorized (authentication required) |
| **403** | Forbidden (authorization failed) |
| **404** | Not Found (resource doesn't exist) |
| **409** | Conflict (business rule violation) |
| **422** | Unprocessable Entity (semantic errors) |
| **500** | Internal Server Error |
| **503** | Service Unavailable |

#### Error Response Format (RFC 7807)

```json
{
  "type": "https://api.company.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "Request validation failed",
  "instance": "/api/v1/customers",
  "correlationId": "abc-123-def-456",
  "timestamp": "2025-12-19T10:15:30.123Z",
  "errors": [
    {
      "field": "email",
      "code": "INVALID_FORMAT",
      "message": "must be a valid email address"
    }
  ]
}
```

### 3.7 Idempotency

For non-idempotent operations (POST), clients MAY provide an idempotency key:

```
POST /api/v1/orders
X-Idempotency-Key: client-generated-uuid-12345
```

The server:
- MUST return the same response for duplicate requests with the same key
- SHOULD retain idempotency keys for at least 24 hours
- MUST return `409 Conflict` if key reused with different payload

---

## Part 4: API Exposure Levels

### 4.1 Public APIs (Internet)

APIs exposed to external consumers over the internet.

**Requirements:**
- ✅ MUST use HTTPS
- ✅ MUST implement rate limiting
- ✅ MUST include HATEOAS
- ✅ MUST have comprehensive OpenAPI documentation
- ✅ MUST implement OAuth2/OIDC authentication
- ✅ MUST use API Gateway
- ✅ MUST version URLs

### 4.2 Partner APIs (B2B)

APIs exposed to trusted business partners.

**Requirements:**
- ✅ MUST use HTTPS
- ✅ MUST implement rate limiting
- ✅ SHOULD include HATEOAS
- ✅ MUST have OpenAPI documentation
- ✅ MUST implement mutual TLS or API keys
- ✅ SHOULD use API Gateway

### 4.3 Internal APIs (Intranet)

APIs for internal service-to-service communication.

**Requirements:**
- ✅ SHOULD use HTTPS (MUST in production)
- ⚠️ Rate limiting optional
- ⚠️ HATEOAS optional
- ✅ MUST have OpenAPI documentation
- ✅ MUST propagate correlation IDs
- ⚠️ May use Service Mesh instead of Gateway

---

## Part 5: gRPC Standards (Placeholder)

> To be defined when gRPC adoption begins.

**Planned sections:**
- Proto file conventions
- Service definition patterns
- Error handling (status codes)
- Streaming patterns
- Interceptors

---

## Part 6: AsyncAPI Standards (Placeholder)

> To be defined when async messaging patterns are standardized.

**Planned sections:**
- Event naming conventions
- Schema evolution strategy
- Idempotency patterns
- Dead letter handling
- Event sourcing considerations

---

## Part 7: Cross-Cutting Concerns

| Concern | Implementation |
|---------|----------------|
| **Correlation ID** | Propagate `X-Correlation-ID` header across all calls |
| **Authentication** | OAuth2 / JWT tokens (public), mTLS (internal) |
| **Authorization** | Role-based access control (RBAC) |
| **Rate Limiting** | Per-client limits on public/partner APIs |
| **Timeouts** | All external calls MUST have timeouts |
| **Resilience** | Circuit breaker on external dependencies (see ADR-004) |
| **Observability** | Structured logging, distributed tracing |

---

## Rationale

### Why 4-Layer Model

1. **Clear Separation of Concerns**: Each layer has distinct responsibilities
2. **Reusability**: Domain APIs can be reused across multiple Composable flows
3. **Scalability**: Layers can scale independently
4. **Testability**: Each layer can be tested in isolation

### Why REST as Primary

1. **Industry Standard**: Widest adoption, best tooling support
2. **Browser Compatibility**: Works natively in all clients
3. **Developer Familiarity**: Lower learning curve
4. **Infrastructure Support**: Load balancers, caches, gateways all support HTTP

### Why gRPC for Internal

1. **Performance**: Binary protocol, HTTP/2 multiplexing
2. **Strong Contracts**: Proto files provide strict typing
3. **Streaming**: Native support for bidirectional streaming

### Alternatives Considered

- **GraphQL**: Rejected for external APIs due to caching complexity and lack of standardized error handling. May be considered for BFF layer.
- **SOAP**: Legacy, not considered for new development

---

## Consequences

### Positive

- ✅ Clear separation of concerns across layers
- ✅ Reusable Domain APIs across multiple workflows
- ✅ Standardized patterns reduce decision fatigue
- ✅ Better developer experience with consistent APIs
- ✅ Support for multiple API technologies

### Negative

- ⚠️ More layers = more network hops (latency)
- ⚠️ Increased complexity for simple use cases
- ⚠️ Requires discipline to maintain layer boundaries
- ⚠️ Multiple standards to learn (REST, gRPC, AsyncAPI)

### Mitigations

- Use caching at Experience layer to reduce latency
- Allow Experience to call Domain directly for simple reads
- Provide templates and automation for API creation
- Clear guidelines and code review enforcement

---

## Implementation

### Reference Implementations

| Aspect | ERI | Status |
|--------|-----|--------|
| Base Architecture | eri-code-001-hexagonal-light-java-spring | ✅ Active |
| API Public Exposure | eri-code-014-api-public-exposure-java-spring | ⏳ Planned |

### Modules

| Module | Purpose | Status |
|--------|---------|--------|
| mod-code-015-hexagonal-base-java-spring | Base structure | ✅ Active |
| mod-code-019-api-public-exposure-java-spring | Pagination, HATEOAS | ⏳ Planned |

### Skills

| Skill | Purpose | Status |
|-------|---------|--------|
| skill-020-microservice-java-spring | Generate base microservice | ✅ Active |

---

## Validation

### Success Criteria

- [ ] All new APIs follow 4-layer model
- [ ] All REST APIs implement pagination for collections
- [ ] Public APIs include HATEOAS links
- [ ] Zero direct calls from Experience to System APIs
- [ ] OpenAPI specs for all APIs

### Compliance Checks

- Automated layer boundary validation
- API contract compliance checks (OpenAPI linting)
- Pagination structure validation
- HATEOAS link validation (public APIs)

---

## References

### Related ADRs

- **ADR-004:** Resilience Patterns
- **ADR-009:** Service Architecture Patterns (Hexagonal Light)
- **ADR-013:** Distributed Transactions (planned)

### External Resources

- [RFC 7807 - Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [JSON:API Specification](https://jsonapi.org/)
- [HATEOAS - Richardson Maturity Model](https://martinfowler.com/articles/richardsonMaturityModel.html)
- [OpenAPI Specification](https://swagger.io/specification/)
- [gRPC Documentation](https://grpc.io/docs/)
- [AsyncAPI Specification](https://www.asyncapi.com/)

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-05-27 | 1.0 | Initial version with 4-layer model | Architecture Team |
| 2025-11-24 | 2.0 | Made framework-agnostic, added error response format | Architecture Team |
| 2025-12-19 | 3.0 | Added API Types (REST/gRPC/AsyncAPI), REST Standards (Pagination, HATEOAS, Filtering), API Exposure Levels, cleaned orphan references | C4E Team |
| 2025-12-22 | 3.1 | Named model as "Fusion API Model", added Fusion API Identification section with inference rules for deterministic skill selection | C4E Team |

---

**Decision Status:** ✅ Accepted and Active  
**Next Review:** Q1 2026
