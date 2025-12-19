package com.bank.customer.controller;

import com.bank.customer.model.CreateCustomerRequest;
import com.bank.customer.model.Customer;
import com.bank.customer.service.CustomerService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.net.URI;
import java.util.UUID;

@RestController
public class CustomerController implements CustomerApi {

    private final CustomerService customerService;

    public CustomerController(CustomerService customerService) {
        this.customerService = customerService;
    }

    @Override
    public ResponseEntity<Customer> createCustomer(CreateCustomerRequest createCustomerRequest) {
        var customer = customerService.createCustomer(createCustomerRequest);
        URI location = ServletUriComponentsBuilder.fromCurrentRequest()
                .path("/{id}")
                .buildAndExpand(customer.getId())
                .toUri();
        return ResponseEntity.created(location).body(customer);
    }

    @Override
    public ResponseEntity<Customer> getCustomerById(UUID customerId, UUID correlationId) {
        var customer = customerService.getCustomerById(customerId);
        return ResponseEntity.ok(customer);
    }
}
