---
id: adr-004-resilience-patterns
title: "ADR-004: Resilience Patterns"
sidebar_label: Resilience Patterns
version: 2
date: 2025-11-20
updated: 2025-11-20
status: Accepted
author: Architecture Team
framework: agnostic
patterns:
  - circuit-breaker
  - retry
  - bulkhead
  - rate-limiter
  - timeout
  - fallback
tags:
  - resilience
  - circuit-breaker
  - retry
  - bulkhead
  - rate-limiter
  - timeout
  - fallback
  - fault-tolerance
  - microservices
  - distributed-systems
related:
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-009-retry-java-resilience4j
  - eri-010-bulkhead-java-resilience4j
  - eri-011-rate-limiter-java-resilience4j
  - eri-012-timeout-java-resilience4j
  - eri-013-fallback-java-resilience4j
  - skill-code-001-add-circuit-breaker-java-resilience4j
  - skill-code-002-add-retry-java-resilience4j
  - skill-code-003-add-bulkhead-java-resilience4j
---

# ADR-004: Resilience Patterns

**Status:** Accepted  
**Date:** 2024-11-20  
**Updated:** 2025-11-20 (v2.0 - Framework-agnostic)  
**Deciders:** Architecture Team

---

## Context

Microservices architectures introduce distributed system challenges where services depend on external APIs, databases, and other microservices. Network issues, service outages, or performance degradation in any dependency can cascade through the system, causing widespread failures.

**Problems we face:**
- External service failures causing cascade failures
- Slow dependencies impacting overall system performance  
- Resource exhaustion from retry storms
- Inability to handle partial outages gracefully
- Lack of fault tolerance in critical flows

**Business impact:**
- Service downtime and poor user experience
- Revenue loss during outages
- Increased operational costs
- Damage to brand reputation

---

## Decision

We will implement **resilience patterns** across all microservices to provide fault tolerance, graceful degradation, and system stability.

### Resilience Patterns Suite

We adopt the following complementary patterns:

#### 1. **Circuit Breaker**
**Purpose:** Prevent cascade failures by failing fast when a dependency is unhealthy

**How it works:**
- Monitors calls to external dependencies
- Opens circuit after threshold failures
- Fails fast while circuit is open
- Periodically tests if dependency recovered

**When to apply:**
- ✅ Calls to external APIs (payment gateways, third-party services)
- ✅ Database operations prone to timeouts
- ✅ Calls to other microservices
- ✅ Any operation that can fail independently

**Implementations:** See ERI-008 (Circuit Breaker)

---

#### 2. **Retry**
**Purpose:** Handle transient failures by automatically retrying failed operations

**How it works:**
- Retries operation after failure
- Uses exponential backoff
- Limits number of retries
- Can combine with circuit breaker

**When to apply:**
- ✅ Network glitches (transient failures)
- ✅ Temporary service unavailability
- ✅ Rate limiting responses (429)
- ❌ NOT for business logic errors
- ❌ NOT for authentication failures

**Implementations:** See ERI-009 (Retry Pattern)

---

#### 3. **Bulkhead**
**Purpose:** Isolate resources to prevent one failing component from exhausting all resources

**How it works:**
- Separates thread pools or connection pools
- Limits concurrent calls per dependency
- Isolates failures to specific partitions

**When to apply:**
- ✅ Multiple external dependencies with different SLAs
- ✅ Protection against slow dependencies
- ✅ When one service shouldn't starve others
- ✅ Critical vs non-critical flows separation

**Implementations:** See ERI-010 (Bulkhead Pattern)

---

#### 4. **Rate Limiter**
**Purpose:** Control rate of operations to prevent overload

**How it works:**
- Limits requests per time window
- Rejects or queues excess requests
- Protects downstream services

**When to apply:**
- ✅ Public-facing APIs
- ✅ Protection against DDoS or abuse
- ✅ Compliance with third-party rate limits
- ✅ Resource capacity management

**Implementations:** See ERI-011 (Rate Limiter Pattern)

---

#### 5. **Timeout**
**Purpose:** Prevent indefinite waiting for slow operations

**How it works:**
- Sets maximum wait time for operations
- Fails fast after timeout
- Releases resources promptly

**When to apply:**
- ✅ ALL external calls (mandatory)
- ✅ Database queries
- ✅ Any I/O operations
- ✅ Long-running computations

**Implementations:** See ERI-012 (Timeout Pattern)

---

#### 6. **Fallback**
**Purpose:** Provide alternative response when primary operation fails

**How it works:**
- Defines degraded but acceptable behavior
- Returns cached data or default values
- Maintains partial functionality

**When to apply:**
- ✅ When degraded service is acceptable
- ✅ Non-critical features
- ✅ Data that can be cached
- ✅ Read operations with reasonable defaults

**Implementations:** See ERI-013 (Fallback Pattern)

---

## Pattern Combination Matrix

Patterns should be **combined** based on use case:

| Use Case | Recommended Patterns | Why |
|----------|---------------------|-----|
| **External API calls** | Circuit Breaker + Retry + Timeout + Fallback | Full protection: fail fast, retry transient, timeout slow, fallback gracefully |
| **Database operations** | Bulkhead + Timeout + Retry | Isolate DB issues, timeout long queries, retry transient failures |
| **Public APIs** | Rate Limiter + Timeout | Control load, prevent abuse, timeout slow clients |
| **Critical flows** | Circuit Breaker + Fallback + Timeout | Ensure availability with degraded functionality |
| **Third-party with SLA** | Circuit Breaker + Retry + Rate Limiter | Respect their limits, handle their failures |
| **Internal microservices** | Circuit Breaker + Retry + Timeout | Standard inter-service resilience |

---

## Implementation Strategy

### Phase 1: Foundation (Current)
1. ✅ Circuit Breaker on all external API calls
2. ✅ Timeout on all external operations
3. ✅ Fallback for non-critical features

### Phase 2: Enhancement (Q1 2025)
4. Retry with exponential backoff
5. Bulkhead for resource isolation
6. Metrics and monitoring for all patterns

### Phase 3: Advanced (Q2 2025)
7. Rate limiting on public APIs
8. Adaptive patterns (adjust thresholds dynamically)
9. Chaos engineering validation

---

## Technology Choices

This ADR is **framework-agnostic**. Implementation varies by technology stack:

### Java/Spring Boot
- **Library:** Resilience4j 2.x
- **Why:** Spring Boot native integration, comprehensive patterns, production-proven
- **Reference:** See ERI-008-java, ERI-009-java, etc.

### NodeJS (Future)
- **Library:** Opossum (circuit breaker), async-retry
- **Reference:** See ERI-050-nodejs, ERI-051-nodejs, etc.

### Go (Future)
- **Library:** gobreaker, go-resiliency
- **Reference:** TBD

### Quarkus (Future)
- **Library:** SmallRye Fault Tolerance
- **Reference:** See ERI-100-quarkus, etc.

---

## Rationale

This decision was made based on the following considerations:

1. **Failure is Inevitable**: In distributed systems, partial failures are normal. Resilience patterns provide predictable behavior during degraded conditions rather than catastrophic cascading failures.

2. **Library Maturity**: Resilience4j is actively maintained, lightweight, and designed for functional programming paradigms. It integrates seamlessly with Spring Boot and provides comprehensive metrics out of the box.

3. **Pattern Composability**: The selected patterns (Circuit Breaker, Retry, Timeout, Bulkhead, Rate Limiter) can be composed declaratively, allowing fine-grained control over failure handling strategies.

4. **Observability**: All patterns expose metrics compatible with Micrometer/Prometheus, enabling proactive monitoring and alerting before failures impact users.

5. **Configuration Externalization**: Runtime-adjustable thresholds allow operations teams to tune resilience behavior without code changes or redeployments.

**Alternatives Considered:**
- **Hystrix**: Deprecated by Netflix. No longer maintained.
- **Sentinel (Alibaba)**: Feature-rich but heavier footprint and less adoption in Western enterprise environments.
- **Custom Implementation**: Rejected due to complexity and maintenance burden. Battle-tested libraries provide better reliability.

---

## Consequences

### Positive
- ✅ System resilient to dependency failures
- ✅ Reduced cascade failure risk
- ✅ Graceful degradation capability
- ✅ Better observability (circuit states, retry attempts)
- ✅ Improved user experience during partial outages
- ✅ Lower operational costs (faster failure recovery)

### Negative
- ⚠️ Increased code complexity
- ⚠️ Additional configuration required
- ⚠️ More comprehensive testing needed
- ⚠️ Learning curve for developers
- ⚠️ Potential for misconfiguration

### Mitigations
- Provide automated skills for pattern application (skill-001, skill-002, etc.)
- Comprehensive ERI documentation per framework
- Default configurations for common cases
- Validation scripts to verify correct implementation
- Training and best practices documentation

---

## Validation

### Success Criteria
- [ ] All external API calls protected by circuit breaker + timeout
- [ ] 90%+ of services implement at least 2 resilience patterns
- [ ] Mean Time To Recovery (MTTR) reduced by 50%
- [ ] Zero cascade failures in production (target)
- [ ] Circuit breaker metrics available in monitoring

### Compliance Checks
- Automated validation via compliance scripts
- Code review checklist includes resilience patterns
- Architecture review validates pattern selection

---

## References

### Enterprise Reference Implementations (ERIs)
- **ERI-008:** Circuit Breaker Pattern (Java, NodeJS, Quarkus)
- **ERI-009:** Retry Pattern (Java, NodeJS, Quarkus)
- **ERI-010:** Bulkhead Pattern (Java, NodeJS, Quarkus)
- **ERI-011:** Rate Limiter Pattern (Java, NodeJS, Quarkus)
- **ERI-012:** Timeout Pattern (Java, NodeJS, Quarkus)
- **ERI-013:** Fallback Pattern (Java, NodeJS, Quarkus)

### Automated Skills
- **skill-001:** add-circuit-breaker-java
- **skill-002:** add-retry-java
- **skill-003:** add-bulkhead-java
- **skill-004:** add-rate-limiter-java
- (Future: skill-050+ for NodeJS, skill-100+ for Quarkus)

### External Resources
- [Resilience4j Documentation](https://resilience4j.readme.io/)
- [Netflix Hystrix (archived)](https://github.com/Netflix/Hystrix)
- [Martin Fowler: Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Release It! - Michael Nygard](https://pragprog.com/titles/mnee2/release-it-second-edition/)

---

## Changelog

### v2.0 (2025-11-20)
- Made framework-agnostic (removed Resilience4j specifics)
- Added all 6 resilience patterns with decision matrix
- Added pattern combination guidance
- Separated framework implementations to ERIs
- Added future framework support (NodeJS, Go, Quarkus)

### v1.0 (2024-05-15)
- Initial version with Resilience4j specifics
- Covered circuit breaker and retry patterns only

---

**Decision Status:** ✅ Accepted and Active  
**Review Date:** Q4 2025
