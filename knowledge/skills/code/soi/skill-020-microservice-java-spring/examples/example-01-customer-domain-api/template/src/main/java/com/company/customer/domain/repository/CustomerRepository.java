package com.company.customer.domain.repository;

import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface (port) for Customer aggregate.
 * 
 * This is a domain port - NO Spring annotations allowed.
 * Implementation (adapter) is in the persistence adapter layer.
 */
public interface CustomerRepository {
    
    /**
     * Saves a customer (create or update).
     * 
     * @param customer the customer to save
     * @return the saved customer
     */
    Customer save(Customer customer);
    
    /**
     * Finds a customer by their unique identifier.
     * 
     * @param id the customer ID
     * @return an Optional containing the customer if found
     */
    Optional<Customer> findById(CustomerId id);
    
    /**
     * Finds all customers.
     * 
     * @return list of all customers
     */
    List<Customer> findAll();
    
    /**
     * Deletes a customer by their identifier.
     * 
     * @param id the customer ID to delete
     */
    void deleteById(CustomerId id);
    
    /**
     * Checks if an email is already in use.
     * 
     * @param email the email to check
     * @return true if email exists
     */
    boolean existsByEmail(String email);
    
    /**
     * Checks if a customer exists.
     * 
     * @param id the customer ID
     * @return true if customer exists
     */
    boolean existsById(CustomerId id);
}
