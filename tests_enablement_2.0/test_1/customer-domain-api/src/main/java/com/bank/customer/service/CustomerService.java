package com.bank.customer.service;

import com.bank.customer.model.CreateCustomerRequest;
import com.bank.customer.model.Customer;

import java.util.UUID;

public interface CustomerService {

    Customer createCustomer(CreateCustomerRequest createCustomerRequest);

    Customer getCustomerById(UUID customerId);

}
