package com.company.customer.application.service;

import com.company.customer.adapter.rest.dto.CreateCustomerRequest;
import com.company.customer.adapter.rest.dto.CustomerDTO;
import com.company.customer.adapter.rest.dto.UpdateCustomerRequest;
import com.company.customer.adapter.rest.mapper.CustomerDtoMapper;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;
import com.company.customer.domain.model.CustomerRegistration;
import com.company.customer.domain.service.CustomerDomainService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * Application service for Customer operations.
 * 
 * This layer contains Spring annotations (@Service, @Transactional).
 * It orchestrates domain services and handles DTO mapping.
 */
@Service
@Transactional
public class CustomerApplicationService {
    
    private final CustomerDomainService domainService;
    private final CustomerDtoMapper mapper;
    
    public CustomerApplicationService(CustomerDomainService domainService, 
                                       CustomerDtoMapper mapper) {
        this.domainService = domainService;
        this.mapper = mapper;
    }
    
    /**
     * Creates a new customer.
     */
    public CustomerDTO createCustomer(CreateCustomerRequest request) {
        CustomerRegistration registration = new CustomerRegistration(
            request.name(),
            request.email(),
            request.age()
        );
        
        Customer customer = domainService.registerCustomer(registration);
        return mapper.toDto(customer);
    }
    
    /**
     * Retrieves a customer by ID.
     */
    @Transactional(readOnly = true)
    public CustomerDTO getCustomer(String id) {
        CustomerId customerId = CustomerId.of(id);
        Customer customer = domainService.getCustomer(customerId);
        return mapper.toDto(customer);
    }
    
    /**
     * Retrieves all customers.
     */
    @Transactional(readOnly = true)
    public List<CustomerDTO> getAllCustomers() {
        return domainService.getAllCustomers().stream()
                .map(mapper::toDto)
                .toList();
    }
    
    /**
     * Updates a customer.
     */
    public CustomerDTO updateCustomer(String id, UpdateCustomerRequest request) {
        CustomerId customerId = CustomerId.of(id);
        Customer customer = domainService.updateCustomer(
            customerId,
            request.name(),
            request.email(),
            request.age()
        );
        return mapper.toDto(customer);
    }
    
    /**
     * Deletes a customer.
     */
    public void deleteCustomer(String id) {
        CustomerId customerId = CustomerId.of(id);
        domainService.deleteCustomer(customerId);
    }
}
