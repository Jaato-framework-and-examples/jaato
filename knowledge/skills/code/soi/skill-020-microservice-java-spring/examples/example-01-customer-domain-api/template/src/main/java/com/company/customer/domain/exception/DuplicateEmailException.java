package com.company.customer.domain.exception;

/**
 * Exception thrown when attempting to use a duplicate email.
 * Domain exception - NO framework annotations.
 */
public class DuplicateEmailException extends RuntimeException {
    
    private final String email;
    
    public DuplicateEmailException(String email) {
        super("Email already in use: " + email);
        this.email = email;
    }
    
    public String getEmail() {
        return email;
    }
}
