package com.company.customer.domain.service;

import com.company.customer.domain.exception.CustomerNotFoundException;
import com.company.customer.domain.exception.DuplicateEmailException;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;
import com.company.customer.domain.model.CustomerRegistration;
import com.company.customer.domain.repository.CustomerRepository;

import java.util.List;
import java.util.Objects;

/**
 * Domain service containing business logic for Customer operations.
 * 
 * IMPORTANT: This is a pure POJO - NO Spring annotations (@Service, @Component, etc.)
 * It is instantiated via @Bean in ApplicationConfig.
 */
public class CustomerDomainService {
    
    private final CustomerRepository repository;
    
    /**
     * Constructor injection - no @Autowired needed.
     * Spring will inject via @Bean configuration.
     */
    public CustomerDomainService(CustomerRepository repository) {
        this.repository = Objects.requireNonNull(repository, "repository must not be null");
    }
    
    /**
     * Registers a new customer.
     * 
     * @param registration the registration data
     * @return the created customer
     * @throws DuplicateEmailException if email already exists
     */
    public Customer registerCustomer(CustomerRegistration registration) {
        Objects.requireNonNull(registration, "registration must not be null");
        
        if (repository.existsByEmail(registration.email())) {
            throw new DuplicateEmailException(registration.email());
        }
        
        Customer customer = Customer.create(registration);
        return repository.save(customer);
    }
    
    /**
     * Retrieves a customer by ID.
     * 
     * @param id the customer ID
     * @return the customer
     * @throws CustomerNotFoundException if not found
     */
    public Customer getCustomer(CustomerId id) {
        Objects.requireNonNull(id, "id must not be null");
        
        return repository.findById(id)
                .orElseThrow(() -> new CustomerNotFoundException(id));
    }
    
    /**
     * Retrieves all customers.
     * 
     * @return list of all customers
     */
    public List<Customer> getAllCustomers() {
        return repository.findAll();
    }
    
    /**
     * Updates a customer's profile.
     * 
     * @param id the customer ID
     * @param name new name
     * @param email new email
     * @param age new age
     * @return the updated customer
     * @throws CustomerNotFoundException if not found
     * @throws DuplicateEmailException if new email already exists
     */
    public Customer updateCustomer(CustomerId id, String name, String email, int age) {
        Objects.requireNonNull(id, "id must not be null");
        
        Customer customer = getCustomer(id);
        
        // Check email uniqueness if changed
        if (!customer.getEmail().equals(email) && repository.existsByEmail(email)) {
            throw new DuplicateEmailException(email);
        }
        
        customer.updateProfile(name, email, age);
        return repository.save(customer);
    }
    
    /**
     * Deletes a customer.
     * 
     * @param id the customer ID
     * @throws CustomerNotFoundException if not found
     */
    public void deleteCustomer(CustomerId id) {
        Objects.requireNonNull(id, "id must not be null");
        
        if (!repository.existsById(id)) {
            throw new CustomerNotFoundException(id);
        }
        
        repository.deleteById(id);
    }
}
