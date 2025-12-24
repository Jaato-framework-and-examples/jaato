---
id: eri-code-001-hexagonal-light-java-spring
title: "ERI-CODE-001: Hexagonal Light Architecture - Java/Spring Boot"
sidebar_label: Hexagonal Light (Java)
version: 1.1
date: 2025-11-24
updated: 2025-11-27
status: Active
author: Architecture Team
domain: code
pattern: hexagonal-light
framework: java
library: spring-boot
library_version: 3.2.x
java_version: "17"
implements:
  - adr-009-service-architecture-patterns
tags:
  - java
  - spring-boot
  - hexagonal
  - architecture
  - microservice
  - reference-implementation
related:
  - adr-001-api-design-standards
  - adr-004-resilience-patterns
  - eri-code-008-circuit-breaker-java-resilience4j
automated_by:
  - skill-code-020-generate-microservice-java-spring
cross_domain_usage: qa
---

# ERI-CODE-001: Hexagonal Light Architecture - Java/Spring Boot

## Overview

This ERI provides a **complete, production-ready reference implementation** of the Hexagonal Light architecture pattern for Java/Spring Boot microservices, as defined in ADR-009.

**Purpose:**
- Starting point for generating new microservices
- Reference for understanding Hexagonal Light in Spring Boot
- Validation target for compliance checking
- Training resource for developers

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Java | 17+ |
| **Framework** | Spring Boot | 3.2.x |
| **Build** | Maven | 3.9+ |
| **Persistence** | Spring Data JPA | 3.2.x |
| **Validation** | Jakarta Validation | 3.0 |
| **Mapping** | MapStruct | 1.5.x |
| **Testing** | JUnit 5 + Mockito | 5.x |
| **Integration Testing** | Testcontainers | 1.19.x |

---

## Project Structure

```
{service-name}/
├── pom.xml
├── README.md
├── Dockerfile
├── .gitignore
│
├── src/main/java/{basePackage}/
│   │
│   ├── Application.java                          # Spring Boot main
│   │
│   ├── domain/                                   # 🎯 DOMAIN LAYER (Pure POJOs)
│   │   ├── model/
│   │   │   ├── Customer.java                    # Domain entity
│   │   │   ├── CustomerId.java                  # Value object
│   │   │   ├── CustomerRegistration.java        # Domain command
│   │   │   └── CustomerTier.java                # Domain enum
│   │   │
│   │   ├── service/
│   │   │   └── CustomerDomainService.java       # Business logic (POJO)
│   │   │
│   │   ├── repository/
│   │   │   └── CustomerRepository.java          # Port interface
│   │   │
│   │   └── exception/
│   │       ├── CustomerNotFoundException.java
│   │       └── InvalidCustomerException.java
│   │
│   ├── application/                              # 🔄 APPLICATION LAYER
│   │   └── service/
│   │       └── CustomerApplicationService.java  # @Service orchestration
│   │
│   ├── adapter/                                  # 🔌 ADAPTER LAYER
│   │   ├── rest/
│   │   │   ├── controller/
│   │   │   │   └── CustomerController.java      # @RestController
│   │   │   ├── dto/
│   │   │   │   ├── CustomerDTO.java
│   │   │   │   ├── CreateCustomerRequest.java
│   │   │   │   └── UpdateCustomerRequest.java
│   │   │   └── mapper/
│   │   │       └── CustomerDtoMapper.java       # DTO ↔ Domain
│   │   │
│   │   └── persistence/
│   │       ├── entity/
│   │       │   └── CustomerEntity.java          # @Entity
│   │       ├── repository/
│   │       │   └── CustomerJpaRepository.java   # JpaRepository
│   │       ├── adapter/
│   │       │   └── CustomerRepositoryAdapter.java
│   │       └── mapper/
│   │           └── CustomerEntityMapper.java    # Entity ↔ Domain
│   │
│   └── infrastructure/                           # ⚙️ INFRASTRUCTURE
│       ├── config/
│       │   ├── ApplicationConfig.java           # Bean wiring
│       │   └── WebConfig.java
│       └── exception/
│           ├── GlobalExceptionHandler.java
│           └── ErrorResponse.java
│
├── src/main/resources/
│   ├── application.yml
│   ├── application-dev.yml
│   └── application-prod.yml
│
└── src/test/java/{basePackage}/
    ├── domain/service/
    │   └── CustomerDomainServiceTest.java       # Fast unit tests
    ├── application/service/
    │   └── CustomerApplicationServiceTest.java
    └── adapter/rest/controller/
        └── CustomerControllerIntegrationTest.java
```

---

## Code Reference

### Domain Layer (Pure POJOs)

#### Domain Entity

```java
// domain/model/Customer.java
package com.company.customer.domain.model;

import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Domain entity representing a Customer.
 * Pure POJO - NO framework annotations.
 */
public class Customer {
    
    private final CustomerId id;
    private String name;
    private String email;
    private int age;
    private CustomerTier tier;
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // Factory method for creation
    public static Customer create(CustomerRegistration registration) {
        return new Customer(
            CustomerId.generate(),
            registration.name(),
            registration.email(),
            registration.age(),
            CustomerTier.STANDARD,
            LocalDateTime.now()
        );
    }
    
    // Constructor
    public Customer(CustomerId id, String name, String email, int age, 
                    CustomerTier tier, LocalDateTime createdAt) {
        this.id = Objects.requireNonNull(id, "id must not be null");
        this.name = Objects.requireNonNull(name, "name must not be null");
        this.email = Objects.requireNonNull(email, "email must not be null");
        this.age = age;
        this.tier = tier;
        this.createdAt = createdAt;
        this.updatedAt = createdAt;
    }
    
    // Business logic methods
    public void assignDefaultTier() {
        this.tier = CustomerTier.STANDARD;
    }
    
    public void upgradeTier() {
        this.tier = switch (this.tier) {
            case STANDARD -> CustomerTier.PREMIUM;
            case PREMIUM -> CustomerTier.VIP;
            case VIP -> CustomerTier.VIP;
        };
        this.updatedAt = LocalDateTime.now();
    }
    
    public void updateDetails(String name, String email) {
        this.name = Objects.requireNonNull(name);
        this.email = Objects.requireNonNull(email);
        this.updatedAt = LocalDateTime.now();
    }
    
    // Getters (no setters - immutable where possible)
    public CustomerId getId() { return id; }
    public String getName() { return name; }
    public String getEmail() { return email; }
    public int getAge() { return age; }
    public CustomerTier getTier() { return tier; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Customer customer = (Customer) o;
        return Objects.equals(id, customer.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
```

#### Value Object

```java
// domain/model/CustomerId.java
package com.company.customer.domain.model;

import java.util.Objects;
import java.util.UUID;

/**
 * Value object for Customer ID.
 * Immutable and self-validating.
 */
public record CustomerId(String value) {
    
    public CustomerId {
        Objects.requireNonNull(value, "CustomerId value must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("CustomerId value must not be blank");
        }
    }
    
    public static CustomerId generate() {
        return new CustomerId(UUID.randomUUID().toString());
    }
    
    public static CustomerId of(String value) {
        return new CustomerId(value);
    }
    
    @Override
    public String toString() {
        return value;
    }
}
```

#### Domain Command

```java
// domain/model/CustomerRegistration.java
package com.company.customer.domain.model;

/**
 * Domain command for customer registration.
 * Represents intent to create a customer.
 */
public record CustomerRegistration(
    String name,
    String email,
    int age
) {
    public CustomerRegistration {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (email == null || email.isBlank()) {
            throw new IllegalArgumentException("email must not be blank");
        }
    }
}
```

#### Domain Enum

```java
// domain/model/CustomerTier.java
package com.company.customer.domain.model;

public enum CustomerTier {
    STANDARD,
    PREMIUM,
    VIP
}
```

#### Domain Service (Business Logic)

```java
// domain/service/CustomerDomainService.java
package com.company.customer.domain.service;

import com.company.customer.domain.model.*;
import com.company.customer.domain.repository.CustomerRepository;
import com.company.customer.domain.exception.*;

import java.util.Optional;

/**
 * Domain service containing business logic.
 * Pure POJO - NO Spring annotations (@Service, @Autowired, etc.)
 * Easily unit testable without Spring context.
 */
public class CustomerDomainService {
    
    private final CustomerRepository repository;
    
    // Constructor injection (no @Autowired)
    public CustomerDomainService(CustomerRepository repository) {
        this.repository = repository;
    }
    
    /**
     * Register a new customer with business rules.
     */
    public Customer registerCustomer(CustomerRegistration registration) {
        // Business rule: Age validation
        validateAge(registration.age());
        
        // Business rule: Check duplicate email
        if (repository.existsByEmail(registration.email())) {
            throw new InvalidCustomerException("Email already registered: " + registration.email());
        }
        
        // Create domain entity
        Customer customer = Customer.create(registration);
        
        // Business rule: Assign default tier
        customer.assignDefaultTier();
        
        // Persist and return
        return repository.save(customer);
    }
    
    /**
     * Get customer by ID.
     */
    public Customer getCustomer(CustomerId id) {
        return repository.findById(id)
            .orElseThrow(() -> new CustomerNotFoundException(id));
    }
    
    /**
     * Update customer details.
     */
    public Customer updateCustomer(CustomerId id, String name, String email) {
        Customer customer = getCustomer(id);
        
        // Business rule: Check email uniqueness if changed
        if (!customer.getEmail().equals(email) && repository.existsByEmail(email)) {
            throw new InvalidCustomerException("Email already in use: " + email);
        }
        
        customer.updateDetails(name, email);
        return repository.save(customer);
    }
    
    /**
     * Upgrade customer tier.
     */
    public Customer upgradeTier(CustomerId id) {
        Customer customer = getCustomer(id);
        customer.upgradeTier();
        return repository.save(customer);
    }
    
    // Private business rule validation
    private void validateAge(int age) {
        if (age < 18) {
            throw new InvalidCustomerException("Customer must be at least 18 years old");
        }
        if (age > 120) {
            throw new InvalidCustomerException("Invalid age: " + age);
        }
    }
}
```

#### Repository Interface (Port)

```java
// domain/repository/CustomerRepository.java
package com.company.customer.domain.repository;

import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;

import java.util.Optional;

/**
 * Repository interface (port) defined in domain layer.
 * Implementation provided by adapter layer.
 */
public interface CustomerRepository {
    
    Customer save(Customer customer);
    
    Optional<Customer> findById(CustomerId id);
    
    boolean existsByEmail(String email);
    
    void deleteById(CustomerId id);
}
```

#### Domain Exceptions

```java
// domain/exception/CustomerNotFoundException.java
package com.company.customer.domain.exception;

import com.company.customer.domain.model.CustomerId;

public class CustomerNotFoundException extends RuntimeException {
    
    private final CustomerId customerId;
    
    public CustomerNotFoundException(CustomerId customerId) {
        super("Customer not found: " + customerId.value());
        this.customerId = customerId;
    }
    
    public CustomerId getCustomerId() {
        return customerId;
    }
}

// domain/exception/InvalidCustomerException.java
package com.company.customer.domain.exception;

public class InvalidCustomerException extends RuntimeException {
    
    public InvalidCustomerException(String message) {
        super(message);
    }
}
```

---

### Application Layer

```java
// application/service/CustomerApplicationService.java
package com.company.customer.application.service;

import com.company.customer.adapter.rest.dto.*;
import com.company.customer.adapter.rest.mapper.CustomerDtoMapper;
import com.company.customer.domain.model.*;
import com.company.customer.domain.service.CustomerDomainService;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Application service - thin orchestration layer.
 * Spring annotations live HERE, not in domain.
 * Coordinates between adapters and domain.
 */
@Service
@Transactional
public class CustomerApplicationService {
    
    private final CustomerDomainService domainService;
    private final CustomerDtoMapper mapper;
    
    public CustomerApplicationService(
            CustomerDomainService domainService,
            CustomerDtoMapper mapper) {
        this.domainService = domainService;
        this.mapper = mapper;
    }
    
    public CustomerDTO createCustomer(CreateCustomerRequest request) {
        // Map DTO to domain command
        CustomerRegistration registration = mapper.toRegistration(request);
        
        // Execute domain logic
        Customer customer = domainService.registerCustomer(registration);
        
        // Map domain to DTO
        return mapper.toDTO(customer);
    }
    
    @Transactional(readOnly = true)
    public CustomerDTO getCustomer(String id) {
        Customer customer = domainService.getCustomer(CustomerId.of(id));
        return mapper.toDTO(customer);
    }
    
    public CustomerDTO updateCustomer(String id, UpdateCustomerRequest request) {
        Customer customer = domainService.updateCustomer(
            CustomerId.of(id),
            request.name(),
            request.email()
        );
        return mapper.toDTO(customer);
    }
    
    public CustomerDTO upgradeTier(String id) {
        Customer customer = domainService.upgradeTier(CustomerId.of(id));
        return mapper.toDTO(customer);
    }
    
    public void deleteCustomer(String id) {
        // Could add domain logic here if needed
        domainService.getCustomer(CustomerId.of(id)); // Verify exists
        // Delete via domain service or directly
    }
}
```

---

### Adapter Layer - REST

#### Controller

```java
// adapter/rest/controller/CustomerController.java
package com.company.customer.adapter.rest.controller;

import com.company.customer.adapter.rest.dto.*;
import com.company.customer.application.service.CustomerApplicationService;

import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/customers")
public class CustomerController {
    
    private final CustomerApplicationService applicationService;
    
    public CustomerController(CustomerApplicationService applicationService) {
        this.applicationService = applicationService;
    }
    
    @PostMapping
    public ResponseEntity<CustomerDTO> createCustomer(
            @Valid @RequestBody CreateCustomerRequest request) {
        CustomerDTO customer = applicationService.createCustomer(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(customer);
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<CustomerDTO> getCustomer(@PathVariable String id) {
        CustomerDTO customer = applicationService.getCustomer(id);
        return ResponseEntity.ok(customer);
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<CustomerDTO> updateCustomer(
            @PathVariable String id,
            @Valid @RequestBody UpdateCustomerRequest request) {
        CustomerDTO customer = applicationService.updateCustomer(id, request);
        return ResponseEntity.ok(customer);
    }
    
    @PostMapping("/{id}/upgrade-tier")
    public ResponseEntity<CustomerDTO> upgradeTier(@PathVariable String id) {
        CustomerDTO customer = applicationService.upgradeTier(id);
        return ResponseEntity.ok(customer);
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteCustomer(@PathVariable String id) {
        applicationService.deleteCustomer(id);
        return ResponseEntity.noContent().build();
    }
}
```

#### DTOs

```java
// adapter/rest/dto/CustomerDTO.java
package com.company.customer.adapter.rest.dto;

import java.time.LocalDateTime;

public record CustomerDTO(
    String id,
    String name,
    String email,
    int age,
    String tier,
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {}

// adapter/rest/dto/CreateCustomerRequest.java
package com.company.customer.adapter.rest.dto;

import jakarta.validation.constraints.*;

public record CreateCustomerRequest(
    @NotBlank(message = "Name is required")
    @Size(min = 2, max = 100, message = "Name must be between 2 and 100 characters")
    String name,
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email must be valid")
    String email,
    
    @Min(value = 18, message = "Must be at least 18 years old")
    @Max(value = 120, message = "Invalid age")
    int age
) {}

// adapter/rest/dto/UpdateCustomerRequest.java
package com.company.customer.adapter.rest.dto;

import jakarta.validation.constraints.*;

public record UpdateCustomerRequest(
    @NotBlank(message = "Name is required")
    @Size(min = 2, max = 100)
    String name,
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email must be valid")
    String email
) {}
```

---

### Adapter Layer - Persistence

#### JPA Entity

```java
// adapter/persistence/entity/CustomerEntity.java
package com.company.customer.adapter.persistence.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "customers")
public class CustomerEntity {
    
    @Id
    @Column(name = "id", length = 36)
    private String id;
    
    @Column(name = "name", nullable = false, length = 100)
    private String name;
    
    @Column(name = "email", nullable = false, unique = true)
    private String email;
    
    @Column(name = "age", nullable = false)
    private int age;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "tier", nullable = false)
    private CustomerTierEntity tier;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;
    
    // JPA requires no-arg constructor
    protected CustomerEntity() {}
    
    // Getters and setters for JPA
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    
    public int getAge() { return age; }
    public void setAge(int age) { this.age = age; }
    
    public CustomerTierEntity getTier() { return tier; }
    public void setTier(CustomerTierEntity tier) { this.tier = tier; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public enum CustomerTierEntity {
        STANDARD, PREMIUM, VIP
    }
}
```

#### Repository Adapter

```java
// adapter/persistence/adapter/CustomerRepositoryAdapter.java
package com.company.customer.adapter.persistence.adapter;

import com.company.customer.adapter.persistence.entity.CustomerEntity;
import com.company.customer.adapter.persistence.mapper.CustomerEntityMapper;
import com.company.customer.adapter.persistence.repository.CustomerJpaRepository;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;
import com.company.customer.domain.repository.CustomerRepository;

import org.springframework.stereotype.Component;

import java.util.Optional;

/**
 * Adapter implementing domain repository interface.
 * Bridges domain and JPA persistence.
 */
@Component
public class CustomerRepositoryAdapter implements CustomerRepository {
    
    private final CustomerJpaRepository jpaRepository;
    private final CustomerEntityMapper mapper;
    
    public CustomerRepositoryAdapter(
            CustomerJpaRepository jpaRepository,
            CustomerEntityMapper mapper) {
        this.jpaRepository = jpaRepository;
        this.mapper = mapper;
    }
    
    @Override
    public Customer save(Customer customer) {
        CustomerEntity entity = mapper.toEntity(customer);
        CustomerEntity saved = jpaRepository.save(entity);
        return mapper.toDomain(saved);
    }
    
    @Override
    public Optional<Customer> findById(CustomerId id) {
        return jpaRepository.findById(id.value())
            .map(mapper::toDomain);
    }
    
    @Override
    public boolean existsByEmail(String email) {
        return jpaRepository.existsByEmail(email);
    }
    
    @Override
    public void deleteById(CustomerId id) {
        jpaRepository.deleteById(id.value());
    }
}
```

---

### Infrastructure - Configuration

```java
// infrastructure/config/ApplicationConfig.java
package com.company.customer.infrastructure.config;

import com.company.customer.domain.repository.CustomerRepository;
import com.company.customer.domain.service.CustomerDomainService;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration for wiring domain layer beans.
 * Domain services are POJOs, instantiated here.
 */
@Configuration
public class ApplicationConfig {
    
    @Bean
    public CustomerDomainService customerDomainService(CustomerRepository repository) {
        // Domain service is a POJO - we instantiate it manually
        return new CustomerDomainService(repository);
    }
}
```

---

## Unit Testing Domain Layer

```java
// test/domain/service/CustomerDomainServiceTest.java
package com.company.customer.domain.service;

import com.company.customer.domain.model.*;
import com.company.customer.domain.repository.CustomerRepository;
import com.company.customer.domain.exception.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Unit tests for domain service.
 * NO Spring context needed - pure POJO testing.
 * Runs in milliseconds.
 */
@ExtendWith(MockitoExtension.class)
class CustomerDomainServiceTest {
    
    @Mock
    private CustomerRepository repository;
    
    private CustomerDomainService domainService;
    
    @BeforeEach
    void setUp() {
        domainService = new CustomerDomainService(repository);
    }
    
    @Test
    void registerCustomer_WithValidData_CreatesCustomer() {
        // Given
        var registration = new CustomerRegistration("John Doe", "john@example.com", 25);
        when(repository.existsByEmail("john@example.com")).thenReturn(false);
        when(repository.save(any(Customer.class))).thenAnswer(inv -> inv.getArgument(0));
        
        // When
        Customer result = domainService.registerCustomer(registration);
        
        // Then
        assertThat(result.getName()).isEqualTo("John Doe");
        assertThat(result.getEmail()).isEqualTo("john@example.com");
        assertThat(result.getTier()).isEqualTo(CustomerTier.STANDARD);
        verify(repository).save(any(Customer.class));
    }
    
    @Test
    void registerCustomer_WithAgeUnder18_ThrowsException() {
        // Given
        var registration = new CustomerRegistration("Minor User", "minor@example.com", 16);
        
        // When/Then
        assertThatThrownBy(() -> domainService.registerCustomer(registration))
            .isInstanceOf(InvalidCustomerException.class)
            .hasMessageContaining("at least 18");
    }
    
    @Test
    void registerCustomer_WithDuplicateEmail_ThrowsException() {
        // Given
        var registration = new CustomerRegistration("John Doe", "existing@example.com", 25);
        when(repository.existsByEmail("existing@example.com")).thenReturn(true);
        
        // When/Then
        assertThatThrownBy(() -> domainService.registerCustomer(registration))
            .isInstanceOf(InvalidCustomerException.class)
            .hasMessageContaining("already registered");
    }
    
    @Test
    void getCustomer_WhenNotFound_ThrowsException() {
        // Given
        var id = CustomerId.of("non-existent");
        when(repository.findById(id)).thenReturn(Optional.empty());
        
        // When/Then
        assertThatThrownBy(() -> domainService.getCustomer(id))
            .isInstanceOf(CustomerNotFoundException.class);
    }
    
    @Test
    void upgradeTier_FromStandard_UpgradesToPremium() {
        // Given
        var id = CustomerId.of("cust-123");
        var customer = Customer.create(new CustomerRegistration("John", "john@example.com", 30));
        when(repository.findById(id)).thenReturn(Optional.of(customer));
        when(repository.save(any(Customer.class))).thenAnswer(inv -> inv.getArgument(0));
        
        // When
        Customer result = domainService.upgradeTier(id);
        
        // Then
        assertThat(result.getTier()).isEqualTo(CustomerTier.PREMIUM);
    }
}
```

---

## Dependencies (pom.xml excerpt)

```xml
<dependencies>
    <!-- Spring Boot Starters -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    
    <!-- MapStruct -->
    <dependency>
        <groupId>org.mapstruct</groupId>
        <artifactId>mapstruct</artifactId>
        <version>1.5.5.Final</version>
    </dependency>
    
    <!-- Database -->
    <dependency>
        <groupId>org.postgresql</groupId>
        <artifactId>postgresql</artifactId>
        <scope>runtime</scope>
    </dependency>
    <dependency>
        <groupId>com.h2database</groupId>
        <artifactId>h2</artifactId>
        <scope>test</scope>
    </dependency>
    
    <!-- Testing -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
    <dependency>
        <groupId>org.testcontainers</groupId>
        <artifactId>junit-jupiter</artifactId>
        <scope>test</scope>
    </dependency>
    <dependency>
        <groupId>org.testcontainers</groupId>
        <artifactId>postgresql</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

---

## Compliance Checklist

| Rule | Check |
|------|-------|
| Domain layer has NO framework annotations | ✅ |
| Domain entities are POJOs | ✅ |
| Repository interface in domain layer | ✅ |
| Repository implementation in adapter layer | ✅ |
| @Service only in application layer | ✅ |
| @RestController only in adapter layer | ✅ |
| Domain tests run without Spring | ✅ |
| Dependency direction: Adapters → Application → Domain | ✅ |

---

## Related Documentation

- **ADR-009:** Service Architecture Patterns
- **ADR-001:** API Design Standards
- **skill-020:** Generate Microservice (uses this ERI)
- **mod-code-015:** Hexagonal Base Module (templates extracted from this ERI)

---

## Changelog

### v1.1 (2025-11-27)
- Added domain prefix to ID (eri-code-001)
- Added cross_domain_usage metadata
- Updated front matter format

### v1.0 (2025-11-24)
- Initial version
- Complete Hexagonal Light implementation for Java/Spring Boot
- Domain, Application, Adapter layers with full code examples
- Unit test examples for domain layer

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-001-hexagonal-constraints
  version: "1.0"
  eri_reference: eri-code-001-hexagonal-light-java-spring
  adr_reference: adr-009-service-architecture-patterns
  
  structural_constraints:
    # Domain Layer Purity
    - id: domain-no-framework-annotations
      rule: "Domain layer classes MUST NOT have Spring/JPA annotations (@Service, @Repository, @Entity, @Component)"
      validation: "grep -rE '@(Service|Repository|Entity|Component|Autowired)' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: domain
      
    - id: domain-entities-are-pojos
      rule: "Domain entities MUST be pure POJOs with business logic methods"
      validation: "Domain model classes have no framework imports"
      severity: ERROR
      layer: domain
      
    - id: repository-interface-in-domain
      rule: "Repository interfaces MUST be defined in domain layer"
      validation: "Interface files exist in domain/repository/"
      severity: ERROR
      layer: domain
      
    # Application Layer
    - id: service-annotation-in-application
      rule: "@Service annotation MUST be used only in application layer"
      validation: "grep -r '@Service' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: application
      
    - id: application-orchestrates-domain
      rule: "Application services MUST delegate business logic to domain services"
      validation: "Application services inject and call domain services"
      severity: WARNING
      layer: application
      
    # Adapter Layer
    - id: controller-in-adapter
      rule: "@RestController MUST be in adapter/rest/ package only"
      validation: "grep -r '@RestController' src/*/java/**/domain/ returns empty AND grep -r '@RestController' src/*/java/**/application/ returns empty"
      severity: ERROR
      layer: adapter
      
    - id: repository-impl-in-adapter
      rule: "Repository implementations MUST be in adapter/persistence/ package"
      validation: "Classes implementing domain repository interfaces are in adapter/persistence/"
      severity: ERROR
      layer: adapter
      
    - id: entity-annotation-in-adapter
      rule: "@Entity annotation MUST be in adapter/persistence/entity/ only"
      validation: "grep -r '@Entity' src/*/java/**/domain/ returns empty"
      severity: ERROR
      layer: adapter
      
    # Dependency Direction
    - id: dependency-direction
      rule: "Dependencies MUST flow inward: Adapter → Application → Domain"
      validation: "Domain layer has no imports from application or adapter packages"
      severity: ERROR
      
    # Mappers
    - id: dto-mapper-in-adapter
      rule: "DTO mappers MUST be in adapter layer"
      validation: "Mapper classes/interfaces are in adapter/**/mapper/"
      severity: ERROR
      layer: adapter
      
    - id: entity-mapper-in-adapter
      rule: "Entity mappers MUST be in adapter/persistence/mapper/"
      validation: "Entity mapper classes are in adapter/persistence/mapper/"
      severity: ERROR
      layer: adapter

  configuration_constraints:
    - id: spring-boot-main-class
      rule: "Application.java with @SpringBootApplication MUST exist at package root"
      validation: "File exists matching **/Application.java with @SpringBootApplication"
      severity: ERROR
      
    - id: actuator-enabled
      rule: "Spring Boot Actuator SHOULD be configured for health endpoints"
      validation: "application.yml contains management.endpoints configuration"
      severity: WARNING
      
    - id: application-config-exists
      rule: "ApplicationConfig.java MUST exist to wire domain beans"
      validation: "File exists in infrastructure/config/ApplicationConfig.java"
      severity: ERROR

  dependency_constraints:
    required:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-web
        reason: "REST API support"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-data-jpa
        reason: "JPA persistence support"
        
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-validation
        reason: "Jakarta validation support"
        
      - groupId: org.mapstruct
        artifactId: mapstruct
        minVersion: "1.5.0"
        reason: "Type-safe DTO/Entity mapping"
        
    optional:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-actuator
        reason: "Health and metrics endpoints"
        
      - groupId: org.testcontainers
        artifactId: postgresql
        reason: "Integration testing with real database"

  testing_constraints:
    - id: domain-tests-no-spring
      rule: "Domain layer unit tests MUST run without Spring context (@SpringBootTest)"
      validation: "Test classes in domain/ package do not use @SpringBootTest"
      severity: ERROR
      
    - id: domain-tests-use-mockito
      rule: "Domain layer tests SHOULD use Mockito for repository mocking"
      validation: "Domain test classes use @ExtendWith(MockitoExtension.class)"
      severity: WARNING
      
    - id: controller-integration-tests
      rule: "Controllers SHOULD have integration tests with @SpringBootTest"
      validation: "Test classes exist for controllers using @SpringBootTest or @WebMvcTest"
      severity: WARNING
      
    - id: test-naming-convention
      rule: "Test methods SHOULD follow naming: methodName_condition_expectedBehavior"
      validation: "Test method names contain underscores separating parts"
      severity: WARNING
```

---

**ERI Status:** ✅ Active  
**Last Reviewed:** 2025-11-28
