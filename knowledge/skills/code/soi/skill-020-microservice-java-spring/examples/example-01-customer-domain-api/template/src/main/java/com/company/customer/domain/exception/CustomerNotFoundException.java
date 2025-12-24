package com.company.customer.domain.exception;

import com.company.customer.domain.model.CustomerId;

/**
 * Exception thrown when a customer is not found.
 * Domain exception - NO framework annotations.
 */
public class CustomerNotFoundException extends RuntimeException {
    
    private final CustomerId customerId;
    
    public CustomerNotFoundException(CustomerId customerId) {
        super("Customer not found with ID: " + customerId);
        this.customerId = customerId;
    }
    
    public CustomerId getCustomerId() {
        return customerId;
    }
}
