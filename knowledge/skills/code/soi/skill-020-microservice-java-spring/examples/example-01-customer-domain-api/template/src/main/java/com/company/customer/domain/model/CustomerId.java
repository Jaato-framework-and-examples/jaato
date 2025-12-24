package com.company.customer.domain.model;

import java.util.Objects;
import java.util.UUID;

/**
 * Value Object representing a Customer identifier.
 * Immutable record - NO framework annotations.
 */
public record CustomerId(String value) {
    
    public CustomerId {
        Objects.requireNonNull(value, "Customer ID value must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("Customer ID value must not be blank");
        }
    }
    
    /**
     * Generates a new unique CustomerId.
     */
    public static CustomerId generate() {
        return new CustomerId(UUID.randomUUID().toString());
    }
    
    /**
     * Creates a CustomerId from an existing string value.
     */
    public static CustomerId of(String value) {
        return new CustomerId(value);
    }
    
    @Override
    public String toString() {
        return value;
    }
}
