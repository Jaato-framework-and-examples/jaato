---
id: eri-code-015-distributed-transactions-java-spring
title: "ERI-CODE-015: Distributed Transactions"
sidebar_label: Distributed Transactions
version: 1.0
date: 2025-12-19
status: Active
author: C4E Team
domain: code
pattern: distributed-transactions
framework: java-spring
library: spring-web
java_version: "17"
spring_boot_version: "3.2.x"
implements:
  - adr-013-distributed-transactions
tags:
  - transactions
  - saga
  - compensation
  - orchestration
  - java
  - spring
related:
  - eri-code-001-hexagonal-light-java-spring
  - eri-code-014-api-public-exposure-java-spring
automated_by:
  - skill-020-microservice-java-spring
---

# ERI-CODE-015: Distributed Transactions

## Overview

This ERI provides reference implementations for distributed transaction patterns defined in ADR-013. It covers the compensation interface for Domain APIs participating in SAGA orchestration, including the standard endpoint, request/response structures, and idempotency handling.

**Implements:** ADR-013 (Distributed Transactions)  
**Status:** Active

**When to use:**
- Domain APIs that participate in multi-domain workflows
- Services whose write operations may need to be reversed
- APIs orchestrated by Composable APIs using SAGA pattern

**When NOT to use:**
- Read-only services (no write operations to compensate)
- Internal microservices not exposed for orchestration
- Simple CRUD without business transactions

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Java | 17+ |
| **Framework** | Spring Boot | 3.2.x |
| **Web** | Spring Web MVC | 6.1.x |
| **Validation** | Jakarta Validation | 3.0.x |

---

## Project Structure

```
{service-name}/
├── src/main/java/com/{company}/{service}/
│   ├── domain/
│   │   └── transaction/
│   │       ├── Compensable.java
│   │       ├── CompensationRequest.java
│   │       ├── CompensationResult.java
│   │       └── CompensationStatus.java
│   ├── application/
│   │   └── service/
│   │       └── {Entity}ApplicationService.java  # implements Compensable
│   ├── adapter/
│   │   └── in/
│   │       └── rest/
│   │           └── {Entity}Controller.java  # includes /compensate endpoint
│   └── infrastructure/
│       └── persistence/
│           └── TransactionLogRepository.java
├── src/main/resources/
│   └── application.yml
└── pom.xml
```

---

## Code Reference

### 1. Compensable Interface

```java
// File: domain/transaction/Compensable.java
// Purpose: Interface that Domain APIs MUST implement to participate in SAGA

package com.bank.customer.domain.transaction;

/**
 * Interface for services that participate in distributed transactions.
 * 
 * Domain APIs that can be orchestrated in a SAGA MUST implement this interface
 * to provide compensation capabilities.
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

```java
// File: domain/transaction/CompensationRequest.java
// Purpose: Base request structure for compensation operations

package com.bank.customer.domain.transaction;

import jakarta.validation.constraints.NotBlank;
import java.util.Map;

/**
 * Base compensation request following ADR-013 structure.
 * 
 * Domain-specific compensation requests should extend this class
 * or use it directly if no additional context is needed.
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
    
    // Getters
    public String getTransactionId() { return transactionId; }
    public String getCorrelationId() { return correlationId; }
    public String getOriginalOperationId() { return originalOperationId; }
    public String getReason() { return reason; }
    public Map<String, Object> getContext() { return context; }
    
    /**
     * Get a typed value from context.
     */
    @SuppressWarnings("unchecked")
    public <T> T getContextValue(String key, Class<T> type) {
        Object value = context.get(key);
        if (value == null) {
            return null;
        }
        if (type.isInstance(value)) {
            return (T) value;
        }
        throw new IllegalArgumentException(
            "Context value for key '" + key + "' is not of type " + type.getName());
    }
}
```

### 3. Domain-Specific Compensation Request

```java
// File: domain/transaction/CustomerCompensationRequest.java
// Purpose: Customer-specific compensation request with typed context

package com.bank.customer.domain.transaction;

import java.util.Map;

/**
 * Customer-specific compensation request.
 * Provides typed access to customer-related compensation context.
 */
public class CustomerCompensationRequest extends CompensationRequest {
    
    public CustomerCompensationRequest(String transactionId,
                                        String correlationId,
                                        String originalOperationId,
                                        String reason,
                                        Map<String, Object> context) {
        super(transactionId, correlationId, originalOperationId, reason, context);
    }
    
    /**
     * Get the customer ID from context.
     */
    public String getCustomerId() {
        return getContextValue("customerId", String.class);
    }
    
    /**
     * Get the operation type from context.
     */
    public String getOperationType() {
        return getContextValue("operationType", String.class);
    }
}
```

### 4. Compensation Result

```java
// File: domain/transaction/CompensationResult.java
// Purpose: Standard result structure for compensation operations

package com.bank.customer.domain.transaction;

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
    
    /**
     * Create a successful compensation result.
     */
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
    
    /**
     * Create result for already compensated operation (idempotency).
     */
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
    
    /**
     * Create result when original operation not found.
     */
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
    
    /**
     * Create result for failed compensation.
     */
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

### 5. Compensation Status Enum

```java
// File: domain/transaction/CompensationStatus.java
// Purpose: Status values for compensation results

package com.bank.customer.domain.transaction;

/**
 * Status values for compensation results per ADR-013.
 */
public enum CompensationStatus {
    
    /**
     * Compensation executed successfully.
     */
    COMPENSATED,
    
    /**
     * Idempotency: operation was already compensated.
     */
    ALREADY_COMPENSATED,
    
    /**
     * Original operation not found (may already be rolled back).
     */
    NOT_FOUND,
    
    /**
     * Compensation failed (requires manual intervention).
     */
    FAILED,
    
    /**
     * Compensation queued for async processing.
     */
    PENDING
}
```

### 6. Transaction Log Entity

```java
// File: domain/transaction/TransactionLog.java
// Purpose: Record of operations for compensation lookup

package com.bank.customer.domain.transaction;

import java.time.Instant;
import java.util.Map;

/**
 * Log entry for tracking operations that may need compensation.
 * Stored in persistence layer for idempotency checks.
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

### 7. Application Service with Compensation

```java
// File: application/service/CustomerApplicationService.java
// Purpose: Application service implementing Compensable interface

package com.bank.customer.application.service;

import com.bank.customer.domain.model.Customer;
import com.bank.customer.domain.model.CustomerId;
import com.bank.customer.domain.repository.CustomerRepository;
import com.bank.customer.domain.transaction.*;
import com.bank.customer.infrastructure.persistence.TransactionLogRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/**
 * Customer application service with compensation support.
 * Implements Compensable to participate in SAGA orchestration.
 */
@Service
public class CustomerApplicationService implements Compensable<CustomerCompensationRequest> {
    
    private static final Logger log = LoggerFactory.getLogger(CustomerApplicationService.class);
    
    private final CustomerRepository customerRepository;
    private final TransactionLogRepository transactionLogRepository;
    
    public CustomerApplicationService(CustomerRepository customerRepository,
                                       TransactionLogRepository transactionLogRepository) {
        this.customerRepository = customerRepository;
        this.transactionLogRepository = transactionLogRepository;
    }
    
    /**
     * Create customer with transaction logging for compensation support.
     */
    @Transactional
    public Customer createCustomer(CreateCustomerRequest request, String transactionId) {
        log.info("Creating customer in transaction: {}", transactionId);
        
        // Create customer
        Customer customer = Customer.create(request.toRegistration());
        customer = customerRepository.save(customer);
        
        // Log operation for potential compensation
        TransactionLog txLog = new TransactionLog(
            UUID.randomUUID().toString(),
            transactionId,
            "CREATE_CUSTOMER",
            customer.getId().toString(),
            Map.of(
                "customerId", customer.getId().toString(),
                "email", request.email()
            )
        );
        transactionLogRepository.save(txLog);
        
        log.info("Customer created: {} in transaction: {}", customer.getId(), transactionId);
        return customer;
    }
    
    /**
     * Compensate a customer creation operation.
     * Implements the Compensable interface for SAGA participation.
     */
    @Override
    @Transactional
    public CompensationResult compensate(CustomerCompensationRequest request) {
        log.info("Compensation requested for transaction: {}, reason: {}", 
                 request.getTransactionId(), request.getReason());
        
        try {
            // Find the original operation
            Optional<TransactionLog> txLog = transactionLogRepository
                .findByTransactionId(request.getTransactionId());
            
            if (txLog.isEmpty()) {
                log.warn("Transaction not found for compensation: {}", 
                         request.getTransactionId());
                return CompensationResult.notFound(
                    request.getTransactionId(),
                    request.getOriginalOperationId()
                );
            }
            
            TransactionLog log = txLog.get();
            
            // Idempotency check
            if (log.isCompensated()) {
                return CompensationResult.alreadyCompensated(
                    request.getTransactionId(),
                    log.getId()
                );
            }
            
            // Execute compensation based on operation type
            switch (log.getOperationType()) {
                case "CREATE_CUSTOMER":
                    compensateCustomerCreation(log, request.getReason());
                    break;
                case "UPDATE_CUSTOMER":
                    compensateCustomerUpdate(log, request.getReason());
                    break;
                default:
                    return CompensationResult.failed(
                        request.getTransactionId(),
                        log.getId(),
                        "Unknown operation type: " + log.getOperationType()
                    );
            }
            
            // Mark as compensated
            log.markCompensated(request.getReason());
            transactionLogRepository.save(log);
            
            return CompensationResult.compensated(
                request.getTransactionId(),
                log.getId(),
                "Customer operation compensated successfully"
            );
            
        } catch (Exception e) {
            log.error("Compensation failed for transaction: {}", 
                      request.getTransactionId(), e);
            return CompensationResult.failed(
                request.getTransactionId(),
                request.getOriginalOperationId(),
                "Compensation failed: " + e.getMessage()
            );
        }
    }
    
    private void compensateCustomerCreation(TransactionLog txLog, String reason) {
        String customerId = (String) txLog.getOperationData().get("customerId");
        log.info("Compensating customer creation: {}", customerId);
        
        // Option 1: Hard delete (if no downstream dependencies)
        // customerRepository.deleteById(CustomerId.of(customerId));
        
        // Option 2: Soft delete / cancel (preferred)
        Optional<Customer> customer = customerRepository.findById(CustomerId.of(customerId));
        customer.ifPresent(c -> {
            c.cancel(reason);
            customerRepository.save(c);
        });
    }
    
    private void compensateCustomerUpdate(TransactionLog txLog, String reason) {
        // Restore previous state from operation data
        log.info("Compensating customer update from transaction log");
        // Implementation depends on what was stored in operationData
    }
}
```

### 8. Controller with Compensation Endpoint

```java
// File: adapter/in/rest/CustomerController.java
// Purpose: REST controller including standard compensation endpoint

package com.bank.customer.adapter.in.rest;

import com.bank.customer.application.service.CustomerApplicationService;
import com.bank.customer.domain.transaction.*;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * REST controller for Customer operations.
 * Includes compensation endpoint per ADR-013.
 */
@RestController
@RequestMapping("/api/v1/customers")
@Tag(name = "Customers", description = "Customer management with compensation support")
public class CustomerController {
    
    private static final Logger log = LoggerFactory.getLogger(CustomerController.class);
    
    private final CustomerApplicationService customerService;
    
    public CustomerController(CustomerApplicationService customerService) {
        this.customerService = customerService;
    }
    
    // ... other endpoints ...
    
    /**
     * Compensation endpoint for SAGA orchestration.
     * 
     * POST /api/v1/customers/compensate
     * 
     * This endpoint is called by the Composable API (SAGA orchestrator)
     * when a distributed transaction needs to be rolled back.
     */
    @PostMapping("/compensate")
    @Operation(
        summary = "Compensate customer operation",
        description = "Reverses a customer operation as part of SAGA rollback"
    )
    public ResponseEntity<CompensationResult> compensate(
            @Valid @RequestBody CustomerCompensationRequest request,
            @RequestHeader("X-Correlation-ID") String correlationId) {
        
        log.info("Compensation request received - transaction: {}, correlation: {}",
                 request.getTransactionId(), correlationId);
        
        CompensationResult result = customerService.compensate(request);
        
        log.info("Compensation result - transaction: {}, status: {}",
                 request.getTransactionId(), result.status());
        
        return switch (result.status()) {
            case COMPENSATED, ALREADY_COMPENSATED -> ResponseEntity.ok(result);
            case NOT_FOUND -> ResponseEntity.status(404).body(result);
            case FAILED -> ResponseEntity.status(500).body(result);
            case PENDING -> ResponseEntity.accepted().body(result);
        };
    }
}
```

### 9. Compensation Request DTO (REST)

```java
// File: adapter/in/rest/dto/CompensationRequestDto.java
// Purpose: REST DTO for compensation requests

package com.bank.customer.adapter.in.rest.dto;

import com.bank.customer.domain.transaction.CustomerCompensationRequest;
import jakarta.validation.constraints.NotBlank;

import java.util.Map;

/**
 * REST DTO for compensation requests.
 * Maps to domain CompensationRequest.
 */
public record CompensationRequestDto(
    @NotBlank(message = "Transaction ID is required")
    String transactionId,
    
    @NotBlank(message = "Correlation ID is required")  
    String correlationId,
    
    String originalOperationId,
    
    String reason,
    
    Map<String, Object> context
) {
    public CustomerCompensationRequest toDomain() {
        return new CustomerCompensationRequest(
            transactionId,
            correlationId,
            originalOperationId,
            reason,
            context
        );
    }
}
```

---

## Configuration

### application.yml

```yaml
# Transaction logging configuration
app:
  transaction:
    log:
      retention-days: 30  # How long to keep transaction logs
      
# Compensation endpoint configuration
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics
  endpoint:
    health:
      show-details: always
      probes:
        enabled: true
```

---

## Dependencies

### Required Dependencies

```xml
<!-- pom.xml -->
<dependencies>
    <!-- Spring Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- Validation -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
    
    <!-- Data JPA (for transaction log persistence) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
</dependencies>
```

---

## Testing

### Compensation Service Test

```java
// File: application/service/CustomerApplicationServiceCompensationTest.java

package com.bank.customer.application.service;

import com.bank.customer.domain.transaction.*;
import com.bank.customer.infrastructure.persistence.TransactionLogRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CustomerApplicationServiceCompensationTest {
    
    @Mock
    private CustomerRepository customerRepository;
    
    @Mock
    private TransactionLogRepository transactionLogRepository;
    
    @InjectMocks
    private CustomerApplicationService service;
    
    @Test
    void compensate_WhenTransactionExists_ShouldCompensate() {
        // Given
        String transactionId = "tx-123";
        TransactionLog txLog = new TransactionLog(
            "log-1", transactionId, "CREATE_CUSTOMER", "cust-1", 
            Map.of("customerId", "cust-1")
        );
        when(transactionLogRepository.findByTransactionId(transactionId))
            .thenReturn(Optional.of(txLog));
        
        CustomerCompensationRequest request = new CustomerCompensationRequest(
            transactionId, "corr-1", null, "Payment failed", Map.of()
        );
        
        // When
        CompensationResult result = service.compensate(request);
        
        // Then
        assertThat(result.status()).isEqualTo(CompensationStatus.COMPENSATED);
        verify(transactionLogRepository).save(any(TransactionLog.class));
    }
    
    @Test
    void compensate_WhenAlreadyCompensated_ShouldReturnIdempotent() {
        // Given
        String transactionId = "tx-123";
        TransactionLog txLog = new TransactionLog(
            "log-1", transactionId, "CREATE_CUSTOMER", "cust-1", Map.of()
        );
        txLog.markCompensated("Previous failure");
        when(transactionLogRepository.findByTransactionId(transactionId))
            .thenReturn(Optional.of(txLog));
        
        CustomerCompensationRequest request = new CustomerCompensationRequest(
            transactionId, "corr-1", null, "New failure", Map.of()
        );
        
        // When
        CompensationResult result = service.compensate(request);
        
        // Then
        assertThat(result.status()).isEqualTo(CompensationStatus.ALREADY_COMPENSATED);
    }
    
    @Test
    void compensate_WhenTransactionNotFound_ShouldReturnNotFound() {
        // Given
        when(transactionLogRepository.findByTransactionId(any()))
            .thenReturn(Optional.empty());
        
        CustomerCompensationRequest request = new CustomerCompensationRequest(
            "unknown-tx", "corr-1", null, "Failure", Map.of()
        );
        
        // When
        CompensationResult result = service.compensate(request);
        
        // Then
        assertThat(result.status()).isEqualTo(CompensationStatus.NOT_FOUND);
    }
}
```

### Controller Test

```java
// File: adapter/in/rest/CustomerControllerCompensationTest.java

package com.bank.customer.adapter.in.rest;

import com.bank.customer.application.service.CustomerApplicationService;
import com.bank.customer.domain.transaction.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CustomerController.class)
class CustomerControllerCompensationTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private CustomerApplicationService customerService;
    
    @Test
    void compensate_ShouldReturnResult() throws Exception {
        // Given
        when(customerService.compensate(any()))
            .thenReturn(CompensationResult.compensated("tx-123", "op-1", "Success"));
        
        // When/Then
        mockMvc.perform(post("/api/v1/customers/compensate")
                .contentType(MediaType.APPLICATION_JSON)
                .header("X-Correlation-ID", "corr-123")
                .content("""
                    {
                        "transactionId": "tx-123",
                        "correlationId": "corr-123",
                        "reason": "Payment failed"
                    }
                    """))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("COMPENSATED"))
            .andExpect(jsonPath("$.transactionId").value("tx-123"));
    }
    
    @Test
    void compensate_WhenNotFound_ShouldReturn404() throws Exception {
        // Given
        when(customerService.compensate(any()))
            .thenReturn(CompensationResult.notFound("tx-123", null));
        
        // When/Then
        mockMvc.perform(post("/api/v1/customers/compensate")
                .contentType(MediaType.APPLICATION_JSON)
                .header("X-Correlation-ID", "corr-123")
                .content("""
                    {
                        "transactionId": "tx-123",
                        "correlationId": "corr-123"
                    }
                    """))
            .andExpect(status().isNotFound())
            .andExpect(jsonPath("$.status").value("NOT_FOUND"));
    }
}
```

---

## Compliance Checklist

Requirements that implementations MUST satisfy:

### Compensation Interface (ERROR if not met)
- [ ] Application Service implements `Compensable<T>` interface
- [ ] `compensate()` method is idempotent
- [ ] `compensate()` returns proper `CompensationResult`
- [ ] Transaction logs are persisted for compensation lookup

### Compensation Endpoint (ERROR if not met)
- [ ] Endpoint exposed at `POST /api/v1/{resource}/compensate`
- [ ] Request includes `transactionId` and `correlationId`
- [ ] Response follows `CompensationResult` structure
- [ ] HTTP status codes: 200 (success), 404 (not found), 500 (failed)

### Idempotency (ERROR if not met)
- [ ] Duplicate compensation requests return `ALREADY_COMPENSATED`
- [ ] Compensation state is persisted

### Logging (WARNING if not met)
- [ ] All compensation attempts are logged
- [ ] Correlation ID is included in logs

---

## Related Documentation

- **ADR:** [ADR-013: Distributed Transactions](../../ADRs/adr-013-distributed-transactions/) - Pattern definition
- **Module:** mod-code-020-compensation-java-spring - Derived templates
- **Skill:** skill-020-microservice-java-spring - Generation automation
- **ERI:** [ERI-001: Hexagonal Light](../eri-code-001-hexagonal-light-java-spring/) - Base architecture

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 1.0 | Initial version | C4E Team |

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-015-distributed-transactions-constraints
  version: "1.0"
  eri_reference: eri-code-015-distributed-transactions-java-spring
  adr_reference: adr-013-distributed-transactions
  
  structural_constraints:
    - id: compensable-interface-implemented
      rule: "Application Service MUST implement Compensable<T> interface"
      validation: "Class implements Compensable interface"
      severity: ERROR
      layer: application
      
    - id: compensate-method-exists
      rule: "Application Service MUST have compensate() method"
      validation: "Method named 'compensate' exists with CompensationResult return type"
      severity: ERROR
      layer: application
      
    - id: compensation-endpoint-exists
      rule: "Controller MUST expose POST /compensate endpoint"
      validation: "@PostMapping with path '/compensate' exists"
      severity: ERROR
      layer: adapter
      
    - id: transaction-log-entity-exists
      rule: "TransactionLog entity MUST exist for compensation lookup"
      validation: "Class TransactionLog exists in domain.transaction package"
      severity: ERROR
      layer: domain
      
    - id: compensation-request-validated
      rule: "CompensationRequest MUST validate transactionId and correlationId"
      validation: "Fields have @NotBlank annotation"
      severity: ERROR
      
    - id: compensation-result-factory-methods
      rule: "CompensationResult SHOULD use factory methods"
      validation: "Static factory methods exist: compensated, alreadyCompensated, notFound, failed"
      severity: WARNING
      
  configuration_constraints:
    - id: correlation-id-header-required
      rule: "Compensation endpoint MUST require X-Correlation-ID header"
      validation: "@RequestHeader with X-Correlation-ID exists on compensate method"
      severity: ERROR
      
  dependency_constraints:
    required:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-web
        reason: "REST endpoint support"
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-validation
        reason: "Request validation"
        
  testing_constraints:
    - id: idempotency-test
      rule: "Tests MUST verify compensation idempotency"
      validation: "Test method verifies ALREADY_COMPENSATED status on duplicate"
      severity: ERROR
      
    - id: not-found-test
      rule: "Tests MUST verify NOT_FOUND handling"
      validation: "Test method verifies NOT_FOUND status for unknown transaction"
      severity: ERROR
      
    - id: endpoint-test
      rule: "Controller tests MUST verify compensation endpoint"
      validation: "MockMvc test for POST /compensate exists"
      severity: WARNING
```
