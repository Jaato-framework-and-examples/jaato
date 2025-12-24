// Template: TransactionLog.java.tpl
// Output: {{basePackagePath}}/domain/transaction/TransactionLog.java
// Purpose: Track operations for compensation lookup

package {{basePackage}}.domain.transaction;

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
        this.operationData = operationData != null ? Map.copyOf(operationData) : Map.of();
        this.createdAt = Instant.now();
        this.status = TransactionLogStatus.COMPLETED;
    }
    
    /**
     * Mark this transaction log as compensated.
     */
    public void markCompensated(String reason) {
        this.status = TransactionLogStatus.COMPENSATED;
        this.compensatedAt = Instant.now();
        this.compensationReason = reason;
    }
    
    /**
     * Check if already compensated (for idempotency).
     */
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
    
    /**
     * Status of a transaction log entry.
     */
    public enum TransactionLogStatus {
        /** Operation completed successfully. */
        COMPLETED,
        /** Operation was compensated (rolled back). */
        COMPENSATED,
        /** Operation or compensation failed. */
        FAILED
    }
}
