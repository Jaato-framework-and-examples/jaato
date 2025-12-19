---
id: skill-001-circuit-breaker-java-resilience4j
title: "SKILL-001: Add Circuit Breaker - Java/Resilience4j"
version: 2.1
domain: code
layer: soi
flow: ADD
framework: java
library: resilience4j
complexity: simple
time_estimate: 2-3 minutes
module: mod-code-001-circuit-breaker-java-resilience4j
implements:
  - adr-004-resilience-patterns
source_eri: eri-code-008-circuit-breaker-java-resilience4j
---

# SKILL-001: Add Circuit Breaker - Java/Resilience4j

## Purpose

Transform Java class by adding Resilience4j circuit breaker pattern to specified method, implementing ADR-004 resilience standards.

## Input

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `targetClass` | Yes | string | Fully qualified class name |
| `targetMethod` | Yes | string | Method name to protect |
| `circuitBreakerName` | No | string | CB instance name (default: derived) |
| `pattern.type` | No | enum | basic_fallback, multiple_fallbacks, fail_fast, programmatic |
| `config.failureRateThreshold` | No | int | Failure % to open (default: 50) |
| `config.waitDurationInOpenState` | No | int | Wait ms in open state (default: 60000) |
| `config.slidingWindowSize` | No | int | Calls for evaluation (default: 100) |

## Output

| Artifact | Description |
|----------|-------------|
| Modified Java file | `@CircuitBreaker` annotation + fallback method |
| pom.xml update | Resilience4j dependency (if missing) |
| application.yml | Circuit breaker configuration |

## Pattern Types

### basic_fallback (default)
Single fallback method for graceful degradation.

### multiple_fallbacks
Chain of fallbacks: primary → secondary → default.

### fail_fast
No fallback - throws `CallNotPermittedException` when open.

### programmatic
Inject `CircuitBreakerRegistry` for dynamic configuration.

## Validation

- [ ] `@CircuitBreaker` annotation present on method
- [ ] Fallback method exists with correct signature (if applicable)
- [ ] Imports added: `io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker`
- [ ] Logger present for fallback logging
- [ ] pom.xml contains resilience4j dependency
- [ ] application.yml contains circuit breaker config
- [ ] Code compiles successfully
