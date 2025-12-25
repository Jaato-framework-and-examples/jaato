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
