// Template: Compensable.java.tpl
// Output: {{basePackagePath}}/domain/transaction/Compensable.java
// Purpose: Interface for services participating in SAGA per ADR-013

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
