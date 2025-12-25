package com.company.customer.domain.model;

import java.util.Objects;

/**
 * Command object for customer registration.
 * Immutable record - NO framework annotations.
 */
public record CustomerRegistration(
    String name,
    String email,
    int age
) {
    public CustomerRegistration {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(email, "email must not be null");
        
        if (name.length() < 2 || name.length() > 100) {
            throw new IllegalArgumentException("name must be between 2 and 100 characters");
        }
        if (age < 18 || age > 120) {
            throw new IllegalArgumentException("age must be between 18 and 120");
        }
    }
}
