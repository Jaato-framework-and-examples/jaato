package com.company.customer.infrastructure.config;

import com.company.customer.domain.repository.CustomerRepository;
import com.company.customer.domain.service.CustomerDomainService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Application configuration for wiring domain services.
 * 
 * Domain services are POJOs (no @Service annotation), so they must be
 * instantiated here as Spring beans.
 */
@Configuration
public class ApplicationConfig {
    
    /**
     * Creates the CustomerDomainService bean.
     * 
     * The domain service is a POJO that receives the repository via constructor.
     * This keeps the domain layer free of Spring annotations.
     */
    @Bean
    public CustomerDomainService customerDomainService(CustomerRepository repository) {
        return new CustomerDomainService(repository);
    }
}
