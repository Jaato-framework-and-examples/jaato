// Template: CompensationRequest.java.tpl
// Output: {{basePackagePath}}/domain/transaction/CompensationRequest.java
// Purpose: Base compensation request per ADR-013

package {{basePackage}}.domain.transaction;

import jakarta.validation.constraints.NotBlank;
import java.util.Map;

/**
 * Base compensation request following ADR-013 structure.
 * 
 * Domain-specific requests may extend this class to add typed context access.
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
    
    /**
     * Get a typed value from context.
     * 
     * @param key the context key
     * @param type the expected type
     * @return the value or null if not present
     * @throws IllegalArgumentException if value is not of expected type
     */
    @SuppressWarnings("unchecked")
    public <V> V getContextValue(String key, Class<V> type) {
        Object value = context.get(key);
        if (value == null) return null;
        if (type.isInstance(value)) return (V) value;
        throw new IllegalArgumentException(
            "Context value for key '" + key + "' is not of type " + type.getName());
    }
}
