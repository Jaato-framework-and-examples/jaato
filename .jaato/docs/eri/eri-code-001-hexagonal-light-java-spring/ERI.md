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
automated_by:
  - skill-code-020-generate-microservice-java-spring
---

# ERI-CODE-001: Hexagonal Light Architecture - Java/Spring Boot

## Overview

This ERI provides a **complete, production-ready reference implementation** of the Hexagonal Light architecture pattern for Java/Spring Boot microservices, as defined in ADR-009.

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Java | 17+ |
| **Framework** | Spring Boot | 3.2.x |
| **Build** | Maven | 3.9+ |
| **Persistence** | Spring Data JPA | 3.2.x |
| **Mapping** | MapStruct | 1.5.x |
| **Testing** | JUnit 5 + Mockito | 5.x |

---

## Project Structure

```
{service-name}/
├── src/main/java/{basePackage}/
│   ├── Application.java                          # Spring Boot main
│   │
│   ├── domain/                                   # 🎯 DOMAIN LAYER (Pure POJOs)
│   │   ├── model/
│   │   │   ├── Customer.java                    # Domain entity
│   │   │   ├── CustomerId.java                  # Value object
│   │   │   └── CustomerTier.java                # Domain enum
│   │   ├── service/
│   │   │   └── CustomerDomainService.java       # Business logic (POJO)
│   │   ├── repository/
│   │   │   └── CustomerRepository.java          # Port interface
│   │   └── exception/
│   │       └── CustomerNotFoundException.java
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
│   │   │   │   └── CreateCustomerRequest.java
│   │   │   └── mapper/
│   │   │       └── CustomerDtoMapper.java
│   │   └── persistence/
│   │       ├── entity/
│   │       │   └── CustomerEntity.java          # @Entity
│   │       ├── repository/
│   │       │   └── CustomerJpaRepository.java
│   │       └── adapter/
│   │           └── CustomerRepositoryAdapter.java
│   │
│   └── infrastructure/
│       └── config/
│           └── ApplicationConfig.java           # Bean wiring
```

---

## Code Reference

### Domain Entity (Pure POJO)

```java
// domain/model/Customer.java
public class Customer {

    private final CustomerId id;
    private String name;
    private String email;
    private CustomerTier tier;

    public static Customer create(CustomerRegistration registration) {
        return new Customer(
            CustomerId.generate(),
            registration.name(),
            registration.email(),
            CustomerTier.STANDARD
        );
    }

    // Business logic methods
    public void upgradeTier() {
        this.tier = switch (this.tier) {
            case STANDARD -> CustomerTier.PREMIUM;
            case PREMIUM -> CustomerTier.VIP;
            case VIP -> CustomerTier.VIP;
        };
    }
}
```

### Domain Service (Business Logic - POJO)

```java
// domain/service/CustomerDomainService.java
public class CustomerDomainService {

    private final CustomerRepository repository;

    public CustomerDomainService(CustomerRepository repository) {
        this.repository = repository;
    }

    public Customer registerCustomer(CustomerRegistration registration) {
        // Business rule: Age validation
        if (registration.age() < 18) {
            throw new InvalidCustomerException("Must be at least 18");
        }

        // Business rule: Check duplicate email
        if (repository.existsByEmail(registration.email())) {
            throw new InvalidCustomerException("Email already registered");
        }

        Customer customer = Customer.create(registration);
        return repository.save(customer);
    }
}
```

### Application Service (Spring Integration)

```java
// application/service/CustomerApplicationService.java
@Service
@Transactional
public class CustomerApplicationService {

    private final CustomerDomainService domainService;
    private final CustomerDtoMapper mapper;

    public CustomerDTO createCustomer(CreateCustomerRequest request) {
        CustomerRegistration registration = mapper.toRegistration(request);
        Customer customer = domainService.registerCustomer(registration);
        return mapper.toDTO(customer);
    }
}
```

### Repository Adapter

```java
// adapter/persistence/adapter/CustomerRepositoryAdapter.java
@Component
public class CustomerRepositoryAdapter implements CustomerRepository {

    private final CustomerJpaRepository jpaRepository;
    private final CustomerEntityMapper mapper;

    @Override
    public Customer save(Customer customer) {
        CustomerEntity entity = mapper.toEntity(customer);
        CustomerEntity saved = jpaRepository.save(entity);
        return mapper.toDomain(saved);
    }
}
```

---

## Unit Testing Domain Layer

```java
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
        var registration = new CustomerRegistration("John", "john@example.com", 25);
        when(repository.existsByEmail("john@example.com")).thenReturn(false);
        when(repository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        Customer result = domainService.registerCustomer(registration);

        assertThat(result.getName()).isEqualTo("John");
        assertThat(result.getTier()).isEqualTo(CustomerTier.STANDARD);
    }
}
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
| Domain tests run without Spring | ✅ |

---

**ERI Status:** ✅ Active
**Last Reviewed:** 2025-11-28
