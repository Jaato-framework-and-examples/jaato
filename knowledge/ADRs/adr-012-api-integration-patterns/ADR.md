# ADR-012: API Integration Patterns

**Status:** Accepted  
**Date:** 2025-12-01  
**Deciders:** Architecture Team  
**Domain:** Integration  

---

## Context

Services in a distributed architecture need to integrate with other services, both internal (within the organization) and external (third-party, legacy systems). Integration patterns must support different protocols, communication styles, and reliability requirements.

### Integration Scenarios

| Scenario | Example | Requirements |
|----------|---------|--------------|
| Domain API → System API | Customer service calls mainframe | Resilience, mapping |
| Domain API → Domain API | Order calls Inventory | Service discovery, resilience |
| Composable API → Domain APIs | Orchestrates multiple domains | Correlation, timeout |
| BFF → Composable/Domain APIs | Mobile app backend | Aggregation, caching |
| Any → External APIs | Payment gateway, KYC provider | Security, resilience |

---

## Decision

Define **API Integration** as a first-class architectural capability with explicit types and implementation patterns.

### Integration Types

#### 1. API-Based Integration (Request/Response)

Synchronous or asynchronous request/response patterns via API contracts.

| Type | Protocol | Specification | Communication |
|------|----------|---------------|---------------|
| **REST** | HTTP/1.1, HTTP/2 | OpenAPI 3.x | Synchronous |
| **gRPC** | HTTP/2 | Protocol Buffers | Synchronous/Streaming |
| **Async Request/Reply** | AMQP, Kafka | AsyncAPI | Asynchronous |

#### 2. Event-Driven Integration (Pub/Sub)

Fire-and-forget or reactive patterns via events/commands.

| Type | Pattern | Use Case | Specification |
|------|---------|----------|---------------|
| **Choreography** | Events | Loose coupling, reactive flows | AsyncAPI |
| **Orchestration** | Commands | SAGA, distributed transactions | AsyncAPI |

### Current Scope (v1.x)

This ADR establishes the framework. Current implementation supports:

| Type | Status | ERI | MODULE |
|------|--------|-----|--------|
| REST Synchronous | ✅ Supported | ERI-013 | mod-code-018 |
| gRPC | 🔮 Planned | - | - |
| Async Request/Reply | 🔮 Planned | - | - |
| Event Choreography | 🔮 Planned | - | - |
| Event Orchestration (SAGA) | 🔮 Planned | - | - |

---

## Principles

### 1. Contract First

All integrations MUST have a formal contract specification:
- REST → OpenAPI 3.x
- gRPC → Protocol Buffers
- Event-Driven → AsyncAPI

### 2. Resilience by Default

All outbound integrations MUST implement resilience patterns:
- Circuit Breaker (mandatory for external/legacy)
- Retry (configurable)
- Timeout (mandatory)

See ADR-004 for resilience pattern details.

### 3. Observability

All integrations MUST support:
- Correlation ID propagation
- Distributed tracing (headers)
- Metrics exposure

### 4. Abstraction

Integration clients SHOULD be abstracted behind interfaces:
- Domain layer defines port (interface)
- Adapter layer provides implementation
- Enables testing and substitution

---

## REST Integration Guidelines

### Client Selection Criteria

When implementing REST integration, select client based on:

| Criteria | Feign | RestClient | RestTemplate |
|----------|-------|------------|--------------|
| API style | Well-defined, stable | Any | Any |
| Control needed | Low | High | High |
| Spring Boot version | Any | 3.2+ | Any |
| Declarative preference | ✅ Yes | ❌ No | ❌ No |
| Dynamic headers | Via interceptors | ✅ Inline | ✅ Inline |
| Recommended for | Internal modern APIs | External/Legacy APIs | Legacy codebases |

**Default Recommendation:** `RestClient` for maximum flexibility, especially with external/legacy APIs.

### Required Headers

All REST integrations MUST propagate:

| Header | Purpose | Source |
|--------|---------|--------|
| `X-Correlation-ID` | Request tracing | MDC or incoming request |
| `X-Source-System` | Origin identification | Service name |
| `Authorization` | Security token | Security context |

---

## Consequences

### Positive

- Clear taxonomy for integration patterns
- Consistent approach across services
- Resilience built-in by default
- Contract-first enables parallel development

### Negative

- Additional abstraction layer
- Learning curve for multiple integration types
- Contract management overhead

### Risks

- Contract drift between services
- Version compatibility issues

---

## Compliance

### Mandatory

- [ ] Integration type explicitly declared in service config
- [ ] Contract specification exists (OpenAPI/Protobuf/AsyncAPI)
- [ ] Resilience patterns applied to outbound calls
- [ ] Correlation headers propagated

### Recommended

- [ ] Integration tests with contract verification
- [ ] Circuit breaker dashboards configured
- [ ] Retry metrics exposed

---

## Related

- **ADR-001:** API Design Standards (API types)
- **ADR-004:** Resilience Patterns (Circuit Breaker, Retry, Timeout)
- **ADR-011:** Persistence Patterns (System API as persistence)
- **ERI-013:** REST Integration Java Spring (implementation)
- **CAP:** integration.api.rest

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-01 | Initial version - REST support |
