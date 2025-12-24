---
id: adr-013-distributed-transactions
title: "ADR-013: Distributed Transactions"
sidebar_label: Distributed Transactions
version: 1.0
date: 2025-12-19
status: Accepted
author: C4E Team
decision_type: pattern
scope: organization
tags:
  - transactions
  - saga
  - compensation
  - consistency
  - orchestration
  - choreography
related:
  - adr-001-api-design
  - adr-004-resilience-patterns
implemented_by:
  - eri-code-015-distributed-transactions-java-spring
---

# ADR-013: Distributed Transactions

**Status:** Accepted  
**Date:** 2025-12-19  
**Deciders:** C4E Team, Architecture Team

---

## Context

In our 4-layer API architecture (ADR-001), business workflows often span multiple Domain APIs. Each Domain API owns its data and operates independently, creating challenges for maintaining consistency across domains.

**Problems we face:**

- Multi-domain operations require coordination (e.g., order creation involves Customer, Inventory, Payment domains)
- Traditional ACID transactions don't work across service boundaries
- Failures in one domain must trigger compensations in others
- Need for eventual consistency while maintaining business integrity
- Complexity in implementing reliable rollback mechanisms

**Business impact:**

- Risk of data inconsistency across domains
- Orphaned records when partial operations fail
- Difficulty debugging distributed failures
- Customer-facing errors when coordination fails

**Constraints:**

- Each Domain API owns its data (no shared databases)
- Services communicate via REST/gRPC (no distributed transaction coordinators)
- Must support both synchronous and asynchronous patterns
- Must be resilient to partial failures

---

## Decision

We adopt the **SAGA pattern** for distributed transactions, with **Orchestration** as the primary approach and **Choreography** for specific use cases.

### Pattern Selection

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED TRANSACTION PATTERNS              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │   SAGA              │      │   2PC / XA          │          │
│  │   ORCHESTRATION     │      │   (Two-Phase Commit)│          │
│  │                     │      │                     │          │
│  │   ✅ PREFERRED      │      │   ❌ NOT USED       │          │
│  │                     │      │                     │          │
│  │   - Composable API  │      │   - Requires        │          │
│  │     coordinates     │      │     coordinator     │          │
│  │   - Explicit flow   │      │   - Blocking        │          │
│  │   - Compensation    │      │   - Tight coupling  │          │
│  │     on failure      │      │   - Not cloud-native│          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │   SAGA              │      │   OUTBOX            │          │
│  │   CHOREOGRAPHY      │      │   PATTERN           │          │
│  │                     │      │                     │          │
│  │   ⚠️ SPECIFIC CASES │      │   ✅ COMPLEMENTARY  │          │
│  │                     │      │                     │          │
│  │   - Event-driven    │      │   - Reliable event  │          │
│  │   - Loose coupling  │      │     publishing      │          │
│  │   - Complex to      │      │   - At-least-once   │          │
│  │     debug           │      │     delivery        │          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: SAGA Orchestration (Primary Pattern)

### How It Works

The **Composable API** acts as the orchestrator, coordinating calls to multiple Domain APIs and managing compensations on failure.

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPOSABLE API (Orchestrator)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   START ──► Step 1 ──► Step 2 ──► Step 3 ──► COMMIT             │
│              │          │          │                             │
│              ▼          ▼          ▼                             │
│           Domain A   Domain B   Domain C                         │
│                                                                  │
│   On failure at Step 3:                                          │
│                                                                  │
│   FAIL ◄── Compensate 2 ◄── Compensate 1 ◄── ROLLBACK           │
│                 │                │                               │
│                 ▼                ▼                               │
│              Domain B        Domain A                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Order Creation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORDER COMPOSABLE API                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Validate Customer ──────────► Customer Domain API            │
│         │                              │                         │
│         │ success                      │ getCustomer(id)         │
│         ▼                              ▼                         │
│  2. Reserve Inventory ──────────► Inventory Domain API           │
│         │                              │                         │
│         │ success                      │ reserveStock(items)     │
│         ▼                              ▼                         │
│  3. Process Payment ────────────► Payment Domain API             │
│         │                              │                         │
│         │ success                      │ processPayment(amount)  │
│         ▼                              ▼                         │
│  4. Create Order ───────────────► Order Domain API               │
│         │                              │                         │
│         │ success                      │ createOrder(data)       │
│         ▼                              │                         │
│      COMPLETE ◄─────────────────────────                         │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│  ON FAILURE (e.g., Payment fails at step 3):                     │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  3. Payment FAILED ─────────────► (no compensation needed)       │
│         │                                                        │
│         ▼                                                        │
│  2. Compensate Inventory ───────► Inventory Domain API           │
│         │                              │                         │
│         │                              │ releaseStock(items)     │
│         ▼                              ▼                         │
│  1. (Customer read-only) ───────► (no compensation needed)       │
│         │                                                        │
│         ▼                                                        │
│      ROLLBACK COMPLETE                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Orchestrator Responsibilities

| Responsibility | Description |
|----------------|-------------|
| **Sequence Management** | Execute steps in correct order |
| **State Tracking** | Track which steps completed successfully |
| **Failure Detection** | Detect failures and determine compensation scope |
| **Compensation Execution** | Call compensation endpoints in reverse order |
| **Idempotency** | Handle retries without duplicate effects |
| **Timeout Management** | Enforce timeouts per step and overall |

---

## Part 2: Domain API Compensation

### Compensation Interface

Domain APIs that participate in distributed transactions MUST implement compensation capabilities.

#### Compensation Endpoint Convention

```
POST /api/v1/{resource}/compensate
```

#### Request Structure

```json
{
  "transactionId": "saga-uuid-12345",
  "correlationId": "request-uuid-67890",
  "originalOperationId": "operation-uuid-11111",
  "reason": "PAYMENT_FAILED",
  "context": {
    "orderId": "order-123",
    "items": ["item-1", "item-2"]
  }
}
```

#### Response Structure

```json
{
  "status": "COMPENSATED",
  "transactionId": "saga-uuid-12345",
  "originalOperationId": "operation-uuid-11111",
  "compensatedAt": "2025-12-19T10:30:00Z",
  "message": "Stock reservation released successfully"
}
```

#### Compensation Status Values

| Status | Description |
|--------|-------------|
| `COMPENSATED` | Compensation executed successfully |
| `ALREADY_COMPENSATED` | Idempotency: operation was already compensated |
| `NOT_FOUND` | Original operation not found (may already be rolled back) |
| `FAILED` | Compensation failed (requires manual intervention) |
| `PENDING` | Compensation queued for async processing |

### Compensation Requirements

| Requirement | Description |
|-------------|-------------|
| **Idempotent** | Calling compensate multiple times has same effect as once |
| **Recorded** | Original operation ID stored for compensation lookup |
| **Reversible** | Domain logic supports undo operations |
| **Timeout-aware** | Compensation has its own timeout |

### When Compensation Is Required

| Operation Type | Compensation Required | Example |
|----------------|----------------------|---------|
| **Create** | ✅ Yes | Delete or mark as cancelled |
| **Update** | ✅ Yes | Restore previous state |
| **Delete** | ⚠️ Depends | Soft delete can restore; hard delete cannot |
| **Read** | ❌ No | No side effects |

### Compensation Strategies

| Strategy | Use When | Example |
|----------|----------|---------|
| **Semantic Undo** | Operation can be logically reversed | Cancel order, release reservation |
| **Counter-Operation** | Create inverse transaction | Credit after debit |
| **State Restoration** | Previous state is preserved | Restore from audit log |
| **Cancellation Flag** | Soft delete/cancel | Mark as `CANCELLED` |

---

## Part 3: SAGA Choreography (Alternative Pattern)

### When to Use Choreography

| Use Case | Rationale |
|----------|-----------|
| Loose coupling required | Services must not know about each other |
| High throughput | Event-driven scales better |
| Simple linear flows | No complex branching logic |
| Async-tolerant | Eventual consistency acceptable |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVENT-DRIVEN SAGA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Order Service                                                   │
│       │                                                          │
│       │ publishes                                                │
│       ▼                                                          │
│  [OrderCreated] ──────────────────────────────────►              │
│                                                     │            │
│                              Inventory Service ◄────┘            │
│                                     │                            │
│                                     │ publishes                  │
│                                     ▼                            │
│                              [StockReserved] ─────────►          │
│                                                       │          │
│                                    Payment Service ◄──┘          │
│                                           │                      │
│                                           │ publishes            │
│                                           ▼                      │
│                                    [PaymentProcessed] ──►        │
│                                                          │       │
│                                        Order Service ◄───┘       │
│                                              │                   │
│                                              ▼                   │
│                                       Order CONFIRMED            │
│                                                                  │
│  ════════════════════════════════════════════════════════════   │
│  ON FAILURE (PaymentFailed event):                               │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│  [PaymentFailed] ──────────────────────────────────►             │
│                                                     │            │
│                              Inventory Service ◄────┘            │
│                                     │                            │
│                                     │ compensates                │
│                                     ▼                            │
│                              [StockReleased] ────────►           │
│                                                      │           │
│                                   Order Service ◄────┘           │
│                                          │                       │
│                                          ▼                       │
│                                   Order CANCELLED                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Choreography Challenges

| Challenge | Mitigation |
|-----------|------------|
| Hard to visualize flow | Document with sequence diagrams |
| Difficult debugging | Correlation IDs, distributed tracing |
| Cyclic dependencies | Careful event design |
| No single source of truth | Event store for audit |

---

## Part 4: Outbox Pattern (Complementary)

### Purpose

Ensure reliable event publishing alongside database operations.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOMAIN SERVICE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   BEGIN TRANSACTION                                              │
│         │                                                        │
│         ├──► 1. Update Business Table (orders)                   │
│         │                                                        │
│         ├──► 2. Insert into Outbox Table                         │
│         │         {                                              │
│         │           "id": "event-123",                           │
│         │           "type": "OrderCreated",                      │
│         │           "payload": {...},                            │
│         │           "status": "PENDING"                          │
│         │         }                                              │
│         │                                                        │
│   COMMIT TRANSACTION                                             │
│                                                                  │
│   ════════════════════════════════════════════════════════════  │
│                                                                  │
│   OUTBOX PROCESSOR (async)                                       │
│         │                                                        │
│         ├──► 1. Read PENDING events from Outbox                  │
│         │                                                        │
│         ├──► 2. Publish to Message Broker                        │
│         │                                                        │
│         └──► 3. Mark as PUBLISHED (or delete)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Outbox Table Schema

```sql
CREATE TABLE outbox (
    id              UUID PRIMARY KEY,
    aggregate_type  VARCHAR(255) NOT NULL,
    aggregate_id    VARCHAR(255) NOT NULL,
    event_type      VARCHAR(255) NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    published_at    TIMESTAMP,
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING'
);

CREATE INDEX idx_outbox_pending ON outbox(status) WHERE status = 'PENDING';
```

---

## Part 5: Implementation Guidelines

### Layer Responsibilities

| Layer | Transaction Role |
|-------|------------------|
| **Experience (BFF)** | Does NOT coordinate transactions |
| **Composable API** | Orchestrator - coordinates SAGA |
| **Domain API** | Participant - implements compensation |
| **System API** | No transaction awareness |

### Mapping to API Model (ADR-001)

```
┌─────────────────────────────────────────────────────────────────┐
│                        COMPOSABLE API                            │
│                     (SAGA Orchestrator)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Responsibilities:                                              │
│   - Define transaction flow                                      │
│   - Track transaction state                                      │
│   - Execute steps in sequence                                    │
│   - Execute compensations on failure                             │
│   - Handle timeouts                                              │
│                                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   DOMAIN API    │ │   DOMAIN API    │ │   DOMAIN API    │
│   (Participant) │ │   (Participant) │ │   (Participant) │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│                 │ │                 │ │                 │
│ Responsibilities│ │ Responsibilities│ │ Responsibilities│
│ - Execute       │ │ - Execute       │ │ - Execute       │
│   operation     │ │   operation     │ │   operation     │
│ - Expose        │ │ - Expose        │ │ - Expose        │
│   /compensate   │ │   /compensate   │ │   /compensate   │
│ - Be idempotent │ │ - Be idempotent │ │ - Be idempotent │
│                 │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Error Handling

| Scenario | Handling |
|----------|----------|
| Step fails | Compensate all previous steps in reverse order |
| Compensation fails | Retry with backoff; alert if exhausted |
| Timeout | Treat as failure; compensate |
| Network partition | Retry; use idempotency keys |
| Orchestrator crashes | Persist state; recover on restart |

### Idempotency Requirements

| Component | Idempotency Mechanism |
|-----------|----------------------|
| Composable API | Transaction ID stored; check before starting |
| Domain API operations | Idempotency key in request header |
| Compensation | Check if already compensated |
| Events | Event ID for deduplication |

---

## Rationale

### Why SAGA over 2PC

| Aspect | SAGA | 2PC |
|--------|------|-----|
| Availability | ✅ No blocking | ❌ Coordinator blocks |
| Scalability | ✅ Async possible | ❌ Synchronous |
| Cloud-native | ✅ Works across services | ❌ Requires coordinator |
| Complexity | ⚠️ Compensation logic | ⚠️ Coordinator setup |

### Why Orchestration over Choreography (by default)

| Aspect | Orchestration | Choreography |
|--------|---------------|--------------|
| Visibility | ✅ Central view | ❌ Distributed |
| Debugging | ✅ Single place | ❌ Trace events |
| Control | ✅ Explicit flow | ❌ Implicit |
| Coupling | ⚠️ Knows participants | ✅ Loosely coupled |

### Trade-offs Accepted

- **Eventual consistency**: Transactions are not ACID; temporary inconsistency is possible
- **Compensation complexity**: Each write operation needs compensation logic
- **State management**: Orchestrator must persist transaction state

---

## Consequences

### Positive

- ✅ Reliable distributed transactions without 2PC
- ✅ Clear responsibility model (Composable orchestrates, Domain compensates)
- ✅ Resilient to partial failures
- ✅ Auditable transaction flow
- ✅ Aligns with API layer model (ADR-001)

### Negative

- ⚠️ Eventual consistency (not immediate)
- ⚠️ Compensation logic adds complexity to Domain APIs
- ⚠️ Need to persist transaction state
- ⚠️ Debugging distributed flows requires tooling

### Mitigations

- Use distributed tracing (correlation IDs)
- Implement robust logging at each step
- Provide transaction status query endpoints
- Alert on compensation failures

---

## Implementation

### Reference Implementations

| Aspect | ERI | Status |
|--------|-----|--------|
| SAGA Orchestration | eri-code-015-distributed-transactions-java-spring | ⏳ Planned |
| Compensation Interface | (included in ERI-015) | ⏳ Planned |

### Modules

| Module | Purpose | Status |
|--------|---------|--------|
| mod-code-020-compensation-java-spring | Compensation interface and endpoint | ⏳ Planned |
| mod-code-021-saga-orchestration-java-spring | SAGA orchestrator patterns | ⏳ Planned |

---

## Validation

### Success Criteria

- [ ] All Domain APIs participating in sagas implement `/compensate`
- [ ] Compensation endpoints are idempotent
- [ ] Composable APIs track transaction state
- [ ] All transactions have correlation IDs
- [ ] Compensation failures trigger alerts

### Compliance Checks

- Automated validation of compensation endpoints
- Idempotency testing for compensation operations
- Transaction flow tracing validation

---

## References

### Related ADRs

- **ADR-001:** API Design - Model, Types & Standards
- **ADR-004:** Resilience Patterns

### External Resources

- [SAGA Pattern - Microservices.io](https://microservices.io/patterns/data/saga.html)
- [Compensating Transaction Pattern - Microsoft](https://docs.microsoft.com/en-us/azure/architecture/patterns/compensating-transaction)
- [Outbox Pattern - Microservices.io](https://microservices.io/patterns/data/transactional-outbox.html)
- [Life Beyond Distributed Transactions - Pat Helland](https://queue.acm.org/detail.cfm?id=3025012)

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 1.0 | Initial version | C4E Team |

---

**Decision Status:** ✅ Accepted and Active  
**Next Review:** Q2 2026
