---
id: mod-code-020-compensation-java-spring
title: "MOD-020: Compensation - Java/Spring Boot"
version: 1.0
date: 2025-12-01
status: Active
derived_from: eri-code-015-distributed-transactions-java-spring
domain: code
tags:
  - java
  - spring-boot
  - saga
  - compensation
  - distributed-transactions
used_by:
  - skill-code-020-generate-microservice-java-spring
---

# MOD-020: Compensation - Java/Spring Boot

**Module ID:** mod-code-020-compensation-java-spring  
**Version:** 1.0  
**Source ERI:** eri-code-015-distributed-transactions-java-spring  
**Framework:** Java 17+ / Spring Boot 3.2.x  
**Used by:** skill-code-020-generate-microservice-java-spring

---

## Purpose

Provides reusable code templates for implementing compensation capabilities in Domain APIs that participate in SAGA orchestration, following ADR-013 standards. This module **extends** the base hexagonal structure (mod-015) with distributed transaction support.

**Use when:**
- Domain API participates in multi-domain workflows
- Service write operations may need to be reversed
- APIs are orchestrated by Composable APIs using SAGA pattern

**Do NOT use when:**
- Read-only services (no write operations to compensate)
- Internal microservices not exposed for orchestration
- Simple CRUD without business transactions

**Composes with:**
- `mod-code-015-hexagonal-base-java-spring` (base structure)
- `mod-code-019-api-public-exposure-java-spring` (if public API)

---

## Template Structure

```
templates/
├── domain/
│   ├── Compensable.java.tpl           # Interface for compensable services
│   ├── CompensationRequest.java.tpl   # Base compensation request
│   ├── CompensationResult.java.tpl    # Compensation result
│   ├── CompensationStatus.java.tpl    # Status enum
│   └── TransactionLog.java.tpl        # Operation tracking entity
├── application/
│   └── CompensableServiceMixin.java.tpl  # Mixin with compensation logic
├── adapter/
│   └── CompensationEndpoint.java.tpl  # REST endpoint fragment
└── infrastructure/
    └── TransactionLogRepository.java.tpl  # Repository for tx logs
```

---

## Template Variables

### Required Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `{{basePackage}}` | string | Java base package | `com.bank.customer` |
| `{{basePackagePath}}` | string | Package as path | `com/bank/customer` |
| `{{entityName}}` | string | Entity name (PascalCase) | `Customer` |
| `{{entityNameLower}}` | string | Entity name (camelCase) | `customer` |

### Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `{{retentionDays}}` | int | 30 | Transaction log retention period |

---

## Template Catalog

| Template | Output Path | Description |
|----------|-------------|-------------|
| `domain/Compensable.java.tpl` | `src/main/java/{{basePackagePath}}/domain/transaction/Compensable.java` | Compensable interface |
| `domain/CompensationRequest.java.tpl` | `src/main/java/{{basePackagePath}}/domain/transaction/CompensationRequest.java` | Base request |
| `domain/CompensationResult.java.tpl` | `src/main/java/{{basePackagePath}}/domain/transaction/CompensationResult.java` | Result record |
| `domain/CompensationStatus.java.tpl` | `src/main/java/{{basePackagePath}}/domain/transaction/CompensationStatus.java` | Status enum |
| `domain/TransactionLog.java.tpl` | `src/main/java/{{basePackagePath}}/domain/transaction/TransactionLog.java` | Transaction log entity |
| `infrastructure/TransactionLogRepository.java.tpl` | `src/main/java/{{basePackagePath}}/infrastructure/persistence/TransactionLogRepository.java` | Repository |

---

## Templates

### 1. Compensable Interface

**File:** `templates/domain/Compensable.java.tpl`

```java
// Template: Compensable.java.tpl
// Output: {{basePackagePath}}/domain/transaction/Compensable.java
// Purpose: Interface for services participating in SAGA

package {{basePackage}}.domain.transaction;

/**
 * Interface for services that participate in distributed transactions.
 * 
 * Domain APIs that can be orchestrated in a SAGA MUST implement this interface
 * to provide compensation capabilities per ADR-013.
 * 
 * @param <T> Type of the compensation request specific to this domain
 */
public interface Compensable<T extends CompensationRequest> {
    
    /**
     * Execute compensation to reverse a previously completed operation.
     * 
     * This method MUST be:
     * - Idempotent: calling multiple times produces the same result
     * - Recorded: compensation attempt is logged for audit
     * - Safe: never throws exceptions that would break the SAGA
     * 
     * @param request the compensation request with operation details
     * @return the result of the compensation attempt
     */
    CompensationResult compensate(T request);
}
```

### 2. Compensation Request

**File:** `templates/domain/CompensationRequest.java.tpl`

```java
// Template: CompensationRequest.java.tpl
// Output: {{basePackagePath}}/domain/transaction/CompensationRequest.java
// Purpose: Base compensation request per ADR-013

package {{basePackage}}.domain.transaction;

import jakarta.validation.constraints.NotBlank;
import java.util.Map;

/**
 * Base compensation request following ADR-013 structure.
 */
public class CompensationRequest {
    
    @NotBlank(message = "Transaction ID is required")
    private final String transactionId;
    
    @NotBlank(message = "Correlation ID is required")
    private final String correlationId;
    
    private final String originalOperationId;
    
    private final String reason;
    
    private final Map<String, Object> context;
    
    public CompensationRequest(String transactionId, 
                                String correlationId,
                                String originalOperationId,
                                String reason,
                                Map<String, Object> context) {
        this.transactionId = transactionId;
        this.correlationId = correlationId;
        this.originalOperationId = originalOperationId;
        this.reason = reason;
        this.context = context != null ? Map.copyOf(context) : Map.of();
    }
    
    public String getTransactionId() { return transactionId; }
    public String getCorrelationId() { return correlationId; }
    public String getOriginalOperationId() { return originalOperationId; }
    public String getReason() { return reason; }
    public Map<String, Object> getContext() { return context; }
    
    @SuppressWarnings("unchecked")
    public <V> V getContextValue(String key, Class<V> type) {
        Object value = context.get(key);
        if (value == null) return null;
        if (type.isInstance(value)) return (V) value;
        throw new IllegalArgumentException(
            "Context value for key '" + key + "' is not of type " + type.getName());
    }
}
```

### 3. Compensation Result

**File:** `templates/domain/CompensationResult.java.tpl`

```java
// Template: CompensationResult.java.tpl
// Output: {{basePackagePath}}/domain/transaction/CompensationResult.java
// Purpose: Compensation result per ADR-013

package {{basePackage}}.domain.transaction;

import java.time.Instant;

/**
 * Result of a compensation attempt following ADR-013 structure.
 */
public record CompensationResult(
    CompensationStatus status,
    String transactionId,
    String originalOperationId,
    Instant compensatedAt,
    String message
) {
    
    public static CompensationResult compensated(String transactionId, 
                                                   String originalOperationId,
                                                   String message) {
        return new CompensationResult(
            CompensationStatus.COMPENSATED,
            transactionId,
            originalOperationId,
            Instant.now(),
            message
        );
    }
    
    public static CompensationResult alreadyCompensated(String transactionId,
                                                         String originalOperationId) {
        return new CompensationResult(
            CompensationStatus.ALREADY_COMPENSATED,
            transactionId,
            originalOperationId,
            Instant.now(),
            "Operation was already compensated"
        );
    }
    
    public static CompensationResult notFound(String transactionId,
                                               String originalOperationId) {
        return new CompensationResult(
            CompensationStatus.NOT_FOUND,
            transactionId,
            originalOperationId,
            null,
            "Original operation not found"
        );
    }
    
    public static CompensationResult failed(String transactionId,
                                             String originalOperationId,
                                             String errorMessage) {
        return new CompensationResult(
            CompensationStatus.FAILED,
            transactionId,
            originalOperationId,
            null,
            errorMessage
        );
    }
}
```

### 4. Compensation Status

**File:** `templates/domain/CompensationStatus.java.tpl`

```java
// Template: CompensationStatus.java.tpl
// Output: {{basePackagePath}}/domain/transaction/CompensationStatus.java
// Purpose: Status values per ADR-013

package {{basePackage}}.domain.transaction;

/**
 * Status values for compensation results per ADR-013.
 */
public enum CompensationStatus {
    
    /** Compensation executed successfully. */
    COMPENSATED,
    
    /** Idempotency: operation was already compensated. */
    ALREADY_COMPENSATED,
    
    /** Original operation not found (may already be rolled back). */
    NOT_FOUND,
    
    /** Compensation failed (requires manual intervention). */
    FAILED,
    
    /** Compensation queued for async processing. */
    PENDING
}
```

### 5. Transaction Log

**File:** `templates/domain/TransactionLog.java.tpl`

```java
// Template: TransactionLog.java.tpl
// Output: {{basePackagePath}}/domain/transaction/TransactionLog.java
// Purpose: Track operations for compensation lookup

package {{basePackage}}.domain.transaction;

import java.time.Instant;
import java.util.Map;

/**
 * Log entry for tracking operations that may need compensation.
 */
public class TransactionLog {
    
    private final String id;
    private final String transactionId;
    private final String operationType;
    private final String entityId;
    private final Map<String, Object> operationData;
    private final Instant createdAt;
    private TransactionLogStatus status;
    private Instant compensatedAt;
    private String compensationReason;
    
    public TransactionLog(String id, 
                          String transactionId,
                          String operationType,
                          String entityId,
                          Map<String, Object> operationData) {
        this.id = id;
        this.transactionId = transactionId;
        this.operationType = operationType;
        this.entityId = entityId;
        this.operationData = operationData;
        this.createdAt = Instant.now();
        this.status = TransactionLogStatus.COMPLETED;
    }
    
    public void markCompensated(String reason) {
        this.status = TransactionLogStatus.COMPENSATED;
        this.compensatedAt = Instant.now();
        this.compensationReason = reason;
    }
    
    public boolean isCompensated() {
        return status == TransactionLogStatus.COMPENSATED;
    }
    
    // Getters
    public String getId() { return id; }
    public String getTransactionId() { return transactionId; }
    public String getOperationType() { return operationType; }
    public String getEntityId() { return entityId; }
    public Map<String, Object> getOperationData() { return operationData; }
    public Instant getCreatedAt() { return createdAt; }
    public TransactionLogStatus getStatus() { return status; }
    public Instant getCompensatedAt() { return compensatedAt; }
    public String getCompensationReason() { return compensationReason; }
    
    public enum TransactionLogStatus {
        COMPLETED,
        COMPENSATED,
        FAILED
    }
}
```

### 6. Transaction Log Repository

**File:** `templates/infrastructure/TransactionLogRepository.java.tpl`

```java
// Template: TransactionLogRepository.java.tpl
// Output: {{basePackagePath}}/infrastructure/persistence/TransactionLogRepository.java
// Purpose: Repository for transaction log persistence

package {{basePackage}}.infrastructure.persistence;

import {{basePackage}}.domain.transaction.TransactionLog;

import java.util.Optional;

/**
 * Repository for persisting transaction logs.
 * Used for compensation lookup and idempotency checks.
 */
public interface TransactionLogRepository {
    
    /**
     * Save a transaction log entry.
     */
    TransactionLog save(TransactionLog log);
    
    /**
     * Find transaction log by transaction ID.
     */
    Optional<TransactionLog> findByTransactionId(String transactionId);
    
    /**
     * Find transaction log by entity ID and operation type.
     */
    Optional<TransactionLog> findByEntityIdAndOperationType(String entityId, String operationType);
}
```

---

## Controller Integration

This module provides the `/compensate` endpoint pattern. Add this to your controller:

```java
/**
 * Compensation endpoint for SAGA orchestration.
 * POST /api/v1/{{entityNamePlural}}/compensate
 */
@PostMapping("/compensate")
@Operation(summary = "Compensate {{entityName}} operation")
public ResponseEntity<CompensationResult> compensate(
        @Valid @RequestBody CompensationRequest request,
        @RequestHeader("X-Correlation-ID") String correlationId) {
    
    log.info("Compensation request - transaction: {}, correlation: {}",
             request.getTransactionId(), correlationId);
    
    CompensationResult result = {{entityNameLower}}Service.compensate(request);
    
    return switch (result.status()) {
        case COMPENSATED, ALREADY_COMPENSATED -> ResponseEntity.ok(result);
        case NOT_FOUND -> ResponseEntity.status(404).body(result);
        case FAILED -> ResponseEntity.status(500).body(result);
        case PENDING -> ResponseEntity.accepted().body(result);
    };
}
```

---

## Service Integration

Application services should implement Compensable and log operations:

```java
@Service
public class {{entityName}}ApplicationService implements Compensable<CompensationRequest> {
    
    private final TransactionLogRepository transactionLogRepository;
    
    @Transactional
    public {{entityName}} create(Create{{entityName}}Request request, String transactionId) {
        // 1. Execute business operation
        {{entityName}} entity = // ... create entity
        
        // 2. Log for potential compensation
        TransactionLog txLog = new TransactionLog(
            UUID.randomUUID().toString(),
            transactionId,
            "CREATE_{{entityName}}",
            entity.getId().toString(),
            Map.of("{{entityNameLower}}Id", entity.getId().toString())
        );
        transactionLogRepository.save(txLog);
        
        return entity;
    }
    
    @Override
    @Transactional
    public CompensationResult compensate(CompensationRequest request) {
        // See ERI-015 for full implementation
    }
}
```

---

## Validation (Tier 3)

### Scripts

| Script | Severity | Validates |
|--------|----------|-----------|
| `compensation-interface-check.sh` | ERROR | Compensable interface and implementation |
| `compensation-endpoint-check.sh` | ERROR | /compensate endpoint exists |
| `transaction-log-check.sh` | WARNING | TransactionLog entity and repository |

### ERI Constraint Mapping

| ERI Constraint | Severity | Script | Check |
|----------------|----------|--------|-------|
| compensable-interface-implemented | ERROR | compensation-interface-check.sh | Service implements Compensable |
| compensate-method-exists | ERROR | compensation-interface-check.sh | compensate() method exists |
| compensation-endpoint-exists | ERROR | compensation-endpoint-check.sh | @PostMapping("/compensate") exists |
| transaction-log-entity-exists | ERROR | transaction-log-check.sh | TransactionLog.java exists |
| idempotency-test | ERROR | compensation-interface-check.sh | Tests verify ALREADY_COMPENSATED |

---

## Usage by Skills

This module is used by:

- `skill-code-020-generate-microservice-java-spring` - When generating Domain APIs that participate in SAGAs

### Layer Selection

| API Layer | Include This Module? |
|-----------|---------------------|
| Experience (BFF) | ❌ No (doesn't participate in SAGAs) |
| Composable | ❌ No (orchestrates, doesn't compensate) |
| Domain | ✅ Yes (participant in SAGAs) |
| System | ❌ No (wrapped by Domain) |

---

## Related

- **ERI:** [eri-code-015-distributed-transactions-java-spring](../../ERIs/eri-code-015-distributed-transactions-java-spring/)
- **ADR:** [ADR-013: Distributed Transactions](../../ADRs/adr-013-distributed-transactions/)
- **Base Module:** [mod-code-015-hexagonal-base-java-spring](../mod-code-015-hexagonal-base-java-spring/)
- **Skills:** skill-code-020-generate-microservice-java-spring

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 1.0 | Initial version | C4E Team |
