package com.bank.customer.service;

import com.bank.customer.model.CreateCustomerRequest;
import com.bank.customer.model.Customer;
import com.bank.customer.repository.PartiesRepository;
import com.bank.customer.service.mapper.CustomerMapper;
import org.springframework.stereotype.Service;

import java.util.UUID;

@Service
public class CustomerServiceImpl implements CustomerService {

    private final PartiesRepository partiesRepository;
    private final CustomerMapper customerMapper;

    public CustomerServiceImpl(PartiesRepository partiesRepository, CustomerMapper customerMapper) {
        this.partiesRepository = partiesRepository;
        this.customerMapper = customerMapper;
    }

    @Override
    public Customer createCustomer(CreateCustomerRequest createCustomerRequest) {
        var partyCreateRequest = customerMapper.toPartyCreateRequest(createCustomerRequest);
        var partyResponse = partiesRepository.createParty(partyCreateRequest);
        return customerMapper.toCustomer(partyResponse);
    }

    @Override
    public Customer getCustomerById(UUID customerId) {
        return partiesRepository.findPartyById(customerId)
                .map(customerMapper::toCustomer)
                .orElseThrow(() -> new RuntimeException("Customer not found")); // Should be a custom exception
    }
}
