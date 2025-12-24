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
     * 
     * @param log the transaction log to save
     * @return the saved log
     */
    TransactionLog save(TransactionLog log);
    
    /**
     * Find transaction log by transaction ID.
     * 
     * @param transactionId the SAGA transaction ID
     * @return the log if found
     */
    Optional<TransactionLog> findByTransactionId(String transactionId);
    
    /**
     * Find transaction log by entity ID and operation type.
     * 
     * @param entityId the entity that was operated on
     * @param operationType the type of operation (CREATE, UPDATE, etc.)
     * @return the log if found
     */
    Optional<TransactionLog> findByEntityIdAndOperationType(String entityId, String operationType);
    
    /**
     * Delete transaction logs older than specified days.
     * Used for cleanup.
     * 
     * @param retentionDays number of days to retain logs
     * @return count of deleted logs
     */
    int deleteOlderThan(int retentionDays);
}
