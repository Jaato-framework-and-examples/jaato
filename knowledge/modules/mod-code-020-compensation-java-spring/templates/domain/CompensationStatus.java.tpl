// Template: CompensationStatus.java.tpl
// Output: {{basePackagePath}}/domain/transaction/CompensationStatus.java
// Purpose: Status values per ADR-013

package {{basePackage}}.domain.transaction;

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
