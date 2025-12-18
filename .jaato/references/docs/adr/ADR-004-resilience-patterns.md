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
  - fault-tolerance
  - microservices
related:
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-code-009-retry-java-resilience4j
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

---

## Decision

We will implement **resilience patterns** across all microservices to provide fault tolerance, graceful degradation, and system stability.

### Resilience Patterns Suite

#### 1. **Circuit Breaker**
**Purpose:** Prevent cascade failures by failing fast when a dependency is unhealthy

**When to apply:**
- ✅ Calls to external APIs (payment gateways, third-party services)
- ✅ Database operations prone to timeouts
- ✅ Calls to other microservices
- ✅ Any operation that can fail independently

#### 2. **Retry**
**Purpose:** Handle transient failures by automatically retrying failed operations

**When to apply:**
- ✅ Network glitches (transient failures)
- ✅ Temporary service unavailability
- ✅ Rate limiting responses (429)
- ❌ NOT for business logic errors
- ❌ NOT for authentication failures

#### 3. **Bulkhead**
**Purpose:** Isolate resources to prevent one failing component from exhausting all resources

#### 4. **Rate Limiter**
**Purpose:** Control rate of operations to prevent overload

#### 5. **Timeout**
**Purpose:** Prevent indefinite waiting for slow operations

#### 6. **Fallback**
**Purpose:** Provide alternative response when primary operation fails

---

## Pattern Combination Matrix

| Use Case | Recommended Patterns |
|----------|---------------------|
| **External API calls** | Circuit Breaker + Retry + Timeout + Fallback |
| **Database operations** | Bulkhead + Timeout + Retry |
| **Public APIs** | Rate Limiter + Timeout |
| **Critical flows** | Circuit Breaker + Fallback + Timeout |
| **Internal microservices** | Circuit Breaker + Retry + Timeout |

---

## Technology Choices

This ADR is **framework-agnostic**. Implementation varies by technology stack:

### Java/Spring Boot
- **Library:** Resilience4j 2.x
- **Reference:** See ERI-008, ERI-009, etc.

---

## Consequences

### Positive
- ✅ System resilient to dependency failures
- ✅ Reduced cascade failure risk
- ✅ Graceful degradation capability
- ✅ Better observability (circuit states, retry attempts)

### Negative
- ⚠️ Increased code complexity
- ⚠️ Additional configuration required
- ⚠️ Learning curve for developers

---

**Decision Status:** ✅ Accepted and Active
**Review Date:** Q4 2025
