---
id: eri-code-012-persistence-patterns-java-spring
title: "ERI-CODE-012: Persistence Patterns - Java/Spring Boot"
sidebar_label: "Persistence Patterns (Java)"
version: 1.1
date: 2025-12-01
updated: 2025-12-22
status: Active
author: "Architecture Team"
domain: code
pattern: persistence
framework: java
spring_boot_version: 3.2.x
implements:
  - adr-011-persistence-patterns
  - adr-009-service-architecture-patterns
tags:
  - java
  - spring-boot
  - persistence
  - jpa
  - system-api
  - hexagonal
  - repository
related:
  - eri-code-001-hexagonal-light-java-spring
  - eri-code-008-circuit-breaker-java-resilience4j
  - eri-code-009-retry-java-resilience4j
  - eri-code-010-timeout-java-resilience4j
modules:
  - mod-code-016-persistence-jpa-spring
  - mod-code-017-persistence-systemapi
cross_domain_usage: qa
---

## Overview

This Enterprise Reference Implementation provides the standard patterns for implementing persistence in Java/Spring Boot microservices following Hexagonal Architecture.

**The Core Principle:** The domain layer defines a Repository interface (port). The adapter layer provides the implementation, which can be either JPA (local database) or System API (delegation to mainframe).

```
┌─────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                           │
│  ┌─────────────────┐    ┌────────────────────────────────┐ │
│  │  Domain Entity  │    │    Repository Interface        │ │
│  │   (Customer)    │    │  (CustomerRepository - port)   │ │
│  └─────────────────┘    └────────────────────────────────┘ │
│                                      ▲                      │
└──────────────────────────────────────┼──────────────────────┘
                                       │ implements
┌──────────────────────────────────────┼──────────────────────┐
│                      ADAPTER LAYER   │                      │
│         ┌────────────────────────────┴───────────────┐      │
│         │                                            │      │
│    ┌────┴─────┐                            ┌────────┴───┐  │
│    │   JPA    │                            │ System API │  │
│    │ Adapter  │                            │  Adapter   │  │
│    └────┬─────┘                            └─────┬──────┘  │
│         │                                        │         │
└─────────┼────────────────────────────────────────┼─────────┘
          │                                        │
          ▼                                        ▼
   ┌──────────────┐                      ┌─────────────────┐
   │   Database   │                      │   System API    │
   │ (PostgreSQL) │                      │   (Mainframe)   │
   └──────────────┘                      └─────────────────┘
```

**Two implementation options:**

| Option | Use When | Module |
|--------|----------|--------|
| **JPA Adapter** | Service owns its data (is System of Record) | mod-code-016 |
| **System API Adapter** | Service delegates to mainframe via REST APIs | mod-code-017 |

---

## Decision Criteria

### When to Use JPA

- Service **is the System of Record** for this data
- Data lifecycle is **managed locally**
- Need complex queries, joins, aggregations
- Examples: Audit logs, configuration data, local caches

### When to Use System API

- Data resides in **mainframe** (Z/OS, DB2, CICS)
- Service is a **Domain API** consuming System APIs
- Need **transactional integrity** with legacy systems
- Examples: Customer data, Account data, Transactions

**In this organization, Domain APIs typically use System API Adapter** because the System of Record is the mainframe.

---

## Common Elements

### Domain Layer (Same for Both Options)

#### Domain Entity

```java
// File: domain/model/Customer.java
package com.company.customer.domain.model;

import lombok.Builder;
import lombok.Getter;
import java.time.LocalDate;

/**
 * Domain entity - NO framework annotations.
 * Pure business object.
 */
@Getter
@Builder
public class Customer {
    private final String id;
    private final String firstName;
    private final String lastName;
    private final String email;
    private final LocalDate birthDate;
    private final CustomerStatus status;
    
    public String getFullName() {
        return firstName + " " + lastName;
    }
    
    public boolean isActive() {
        return status == CustomerStatus.ACTIVE;
    }
}
```

#### Repository Interface (Port)

```java
// File: domain/repository/CustomerRepository.java
package com.company.customer.domain.repository;

import com.company.customer.domain.model.Customer;
import java.util.Optional;
import java.util.List;

/**
 * Repository port - defined in domain layer.
 * Implementation provided by adapter layer.
 */
public interface CustomerRepository {
    
    Optional<Customer> findById(String id);
    
    List<Customer> findByEmail(String email);
    
    Customer save(Customer customer);
    
    void deleteById(String id);
}
```

---

## Option A: JPA Adapter

For services that **own their data** and persist to a local database.

### Project Structure

```
src/main/java/com/company/{service}/
├── domain/
│   ├── model/
│   │   └── Customer.java              # Domain entity (NO JPA)
│   └── repository/
│       └── CustomerRepository.java    # Port interface
│
└── adapter/
    └── persistence/
        ├── entity/
        │   └── CustomerJpaEntity.java # JPA entity (adapter)
        ├── repository/
        │   └── CustomerJpaRepository.java  # Spring Data JPA
        ├── mapper/
        │   └── CustomerPersistenceMapper.java
        └── CustomerPersistenceAdapter.java # Implements port
```

### JPA Entity (Adapter Layer)

```java
// File: adapter/persistence/entity/CustomerJpaEntity.java
package com.company.customer.adapter.persistence.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDate;

@Entity
@Table(name = "customers")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CustomerJpaEntity {
    
    @Id
    @Column(name = "customer_id", length = 36)
    private String id;
    
    @Column(name = "first_name", nullable = false, length = 100)
    private String firstName;
    
    @Column(name = "last_name", nullable = false, length = 100)
    private String lastName;
    
    @Column(name = "email", nullable = false, unique = true, length = 255)
    private String email;
    
    @Column(name = "birth_date")
    private LocalDate birthDate;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    private CustomerStatus status;
    
    @Version
    private Long version;
}
```

### Spring Data Repository

```java
// File: adapter/persistence/repository/CustomerJpaRepository.java
package com.company.customer.adapter.persistence.repository;

import com.company.customer.adapter.persistence.entity.CustomerJpaEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface CustomerJpaRepository extends JpaRepository<CustomerJpaEntity, String> {
    
    List<CustomerJpaEntity> findByEmail(String email);
}
```

### Persistence Mapper

```java
// File: adapter/persistence/mapper/CustomerPersistenceMapper.java
package com.company.customer.adapter.persistence.mapper;

import com.company.customer.adapter.persistence.entity.CustomerJpaEntity;
import com.company.customer.domain.model.Customer;
import org.springframework.stereotype.Component;

@Component
public class CustomerPersistenceMapper {
    
    public Customer toDomain(CustomerJpaEntity entity) {
        if (entity == null) return null;
        
        return Customer.builder()
            .id(entity.getId())
            .firstName(entity.getFirstName())
            .lastName(entity.getLastName())
            .email(entity.getEmail())
            .birthDate(entity.getBirthDate())
            .status(entity.getStatus())
            .build();
    }
    
    public CustomerJpaEntity toEntity(Customer domain) {
        if (domain == null) return null;
        
        return CustomerJpaEntity.builder()
            .id(domain.getId())
            .firstName(domain.getFirstName())
            .lastName(domain.getLastName())
            .email(domain.getEmail())
            .birthDate(domain.getBirthDate())
            .status(domain.getStatus())
            .build();
    }
}
```

### Persistence Adapter (Implements Port)

```java
// File: adapter/persistence/CustomerPersistenceAdapter.java
package com.company.customer.adapter.persistence;

import com.company.customer.adapter.persistence.entity.CustomerJpaEntity;
import com.company.customer.adapter.persistence.mapper.CustomerPersistenceMapper;
import com.company.customer.adapter.persistence.repository.CustomerJpaRepository;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.repository.CustomerRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

@Component
@RequiredArgsConstructor
public class CustomerPersistenceAdapter implements CustomerRepository {
    
    private final CustomerJpaRepository jpaRepository;
    private final CustomerPersistenceMapper mapper;
    
    @Override
    public Optional<Customer> findById(String id) {
        return jpaRepository.findById(id)
            .map(mapper::toDomain);
    }
    
    @Override
    public List<Customer> findByEmail(String email) {
        return jpaRepository.findByEmail(email).stream()
            .map(mapper::toDomain)
            .toList();
    }
    
    @Override
    public Customer save(Customer customer) {
        CustomerJpaEntity entity = mapper.toEntity(customer);
        CustomerJpaEntity saved = jpaRepository.save(entity);
        return mapper.toDomain(saved);
    }
    
    @Override
    public void deleteById(String id) {
        jpaRepository.deleteById(id);
    }
}
```

### JPA Configuration

```yaml
# application.yml
spring:
  datasource:
    url: jdbc:postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME:customerdb}
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
    hikari:
      maximum-pool-size: ${DB_POOL_SIZE:10}
      minimum-idle: 5
      connection-timeout: 30000
      
  jpa:
    hibernate:
      ddl-auto: validate  # MUST be 'validate' in production
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        format_sql: true
        jdbc:
          batch_size: 50
    show-sql: false
    open-in-view: false  # Disable OSIV for better performance
```

### JPA Dependencies

```xml
<!-- pom.xml -->
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    
    <dependency>
        <groupId>org.postgresql</groupId>
        <artifactId>postgresql</artifactId>
        <scope>runtime</scope>
    </dependency>
    
    <!-- For testing -->
    <dependency>
        <groupId>com.h2database</groupId>
        <artifactId>h2</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

---

## Option B: System API Adapter

For services that **delegate persistence** to System APIs wrapping mainframe transactions.

### Project Structure

```
src/main/java/com/company/{service}/
├── domain/
│   ├── model/
│   │   └── Customer.java              # Domain entity
│   └── repository/
│       └── CustomerRepository.java    # Port interface (same as JPA)
│
└── adapter/
    └── systemapi/
        ├── client/
        │   └── CustomerSystemApiClient.java  # REST client
        ├── dto/
        │   └── CustomerDto.java       # API contract DTO
        ├── mapper/
        │   └── CustomerSystemApiMapper.java
        ├── config/
        │   └── SystemApiConfig.java   # Client configuration
        └── CustomerSystemApiAdapter.java # Implements port
```

### System API DTO

```java
// File: adapter/systemapi/dto/CustomerDto.java
package com.company.customer.adapter.systemapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import java.time.LocalDate;

/**
 * DTO matching System API contract.
 * Field names may differ from domain model.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CustomerDto {
    
    @JsonProperty("customer_id")
    private String customerId;
    
    @JsonProperty("first_name")
    private String firstName;
    
    @JsonProperty("last_name")
    private String lastName;
    
    @JsonProperty("email_address")
    private String emailAddress;
    
    @JsonProperty("date_of_birth")
    private LocalDate dateOfBirth;
    
    @JsonProperty("customer_status")
    private String customerStatus;
}
```

### REST Client Options

The System API adapter can use different REST clients. Choose based on project requirements:

#### Option B.1: Feign Client (Recommended)

```java
// File: adapter/systemapi/client/CustomerSystemApiClient.java
package com.company.customer.adapter.systemapi.client;

import com.company.customer.adapter.systemapi.dto.CustomerDto;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@FeignClient(
    name = "customer-system-api",
    url = "${system-api.customer.base-url}",
    configuration = SystemApiFeignConfig.class
)
public interface CustomerSystemApiClient {
    
    @GetMapping("/api/v1/customers/{id}")
    CustomerDto findById(@PathVariable("id") String id);
    
    @GetMapping("/api/v1/customers")
    List<CustomerDto> findByEmail(@RequestParam("email") String email);
    
    @PostMapping("/api/v1/customers")
    CustomerDto save(@RequestBody CustomerDto customer);
    
    @DeleteMapping("/api/v1/customers/{id}")
    void deleteById(@PathVariable("id") String id);
}
```

**Feign Configuration:**

```java
// File: adapter/systemapi/config/SystemApiFeignConfig.java
package com.company.customer.adapter.systemapi.config;

import feign.Logger;
import feign.RequestInterceptor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SystemApiFeignConfig {
    
    @Bean
    public Logger.Level feignLoggerLevel() {
        return Logger.Level.BASIC;
    }
    
    @Bean
    public RequestInterceptor systemApiRequestInterceptor() {
        return requestTemplate -> {
            requestTemplate.header("X-Source-System", "customer-domain-api");
            requestTemplate.header("X-Correlation-Id", getCorrelationId());
            // Add other System API specific headers
        };
    }
    
    private String getCorrelationId() {
        // Get from MDC or generate
        return java.util.UUID.randomUUID().toString();
    }
}
```

**Feign Dependencies:**

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

#### Option B.2: RestTemplate

```java
// File: adapter/systemapi/client/CustomerSystemApiClient.java
package com.company.customer.adapter.systemapi.client;

import com.company.customer.adapter.systemapi.dto.CustomerDto;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import java.util.List;

@Component
@RequiredArgsConstructor
public class CustomerSystemApiClient {
    
    private final RestTemplate restTemplate;
    
    @Value("${system-api.customer.base-url}")
    private String baseUrl;
    
    public CustomerDto findById(String id) {
        HttpHeaders headers = createHeaders();
        HttpEntity<Void> entity = new HttpEntity<>(headers);
        
        ResponseEntity<CustomerDto> response = restTemplate.exchange(
            baseUrl + "/api/v1/customers/{id}",
            HttpMethod.GET,
            entity,
            CustomerDto.class,
            id
        );
        return response.getBody();
    }
    
    public List<CustomerDto> findByEmail(String email) {
        HttpHeaders headers = createHeaders();
        HttpEntity<Void> entity = new HttpEntity<>(headers);
        
        ResponseEntity<List<CustomerDto>> response = restTemplate.exchange(
            baseUrl + "/api/v1/customers?email={email}",
            HttpMethod.GET,
            entity,
            new ParameterizedTypeReference<>() {},
            email
        );
        return response.getBody();
    }
    
    public CustomerDto save(CustomerDto customer) {
        HttpHeaders headers = createHeaders();
        HttpEntity<CustomerDto> entity = new HttpEntity<>(customer, headers);
        
        ResponseEntity<CustomerDto> response = restTemplate.exchange(
            baseUrl + "/api/v1/customers",
            HttpMethod.POST,
            entity,
            CustomerDto.class
        );
        return response.getBody();
    }
    
    public void deleteById(String id) {
        HttpHeaders headers = createHeaders();
        HttpEntity<Void> entity = new HttpEntity<>(headers);
        
        restTemplate.exchange(
            baseUrl + "/api/v1/customers/{id}",
            HttpMethod.DELETE,
            entity,
            Void.class,
            id
        );
    }
    
    private HttpHeaders createHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("X-Source-System", "customer-domain-api");
        headers.set("X-Correlation-Id", java.util.UUID.randomUUID().toString());
        return headers;
    }
}
```

**RestTemplate Configuration:**

```java
// File: adapter/systemapi/config/RestTemplateConfig.java
package com.company.customer.adapter.systemapi.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import java.time.Duration;

@Configuration
public class RestTemplateConfig {
    
    @Value("${system-api.customer.connect-timeout:5s}")
    private Duration connectTimeout;
    
    @Value("${system-api.customer.read-timeout:10s}")
    private Duration readTimeout;
    
    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
            .connectTimeout(connectTimeout)
            .readTimeout(readTimeout)
            .build();
    }
}
```

#### Option B.3: RestClient (Spring 6.1+)

```java
// File: adapter/systemapi/client/CustomerSystemApiClient.java
package com.company.customer.adapter.systemapi.client;

import com.company.customer.adapter.systemapi.dto.CustomerDto;
import lombok.RequiredArgsConstructor;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import java.util.List;

@Component
@RequiredArgsConstructor
public class CustomerSystemApiClient {
    
    private final RestClient restClient;
    
    public CustomerDto findById(String id) {
        return restClient.get()
            .uri("/api/v1/customers/{id}", id)
            .header("X-Source-System", "customer-domain-api")
            .retrieve()
            .body(CustomerDto.class);
    }
    
    public List<CustomerDto> findByEmail(String email) {
        return restClient.get()
            .uri("/api/v1/customers?email={email}", email)
            .header("X-Source-System", "customer-domain-api")
            .retrieve()
            .body(new ParameterizedTypeReference<>() {});
    }
    
    public CustomerDto save(CustomerDto customer) {
        return restClient.post()
            .uri("/api/v1/customers")
            .header("X-Source-System", "customer-domain-api")
            .body(customer)
            .retrieve()
            .body(CustomerDto.class);
    }
    
    public void deleteById(String id) {
        restClient.delete()
            .uri("/api/v1/customers/{id}", id)
            .header("X-Source-System", "customer-domain-api")
            .retrieve()
            .toBodilessEntity();
    }
}
```

**RestClient Configuration:**

```java
// File: adapter/systemapi/config/RestClientConfig.java
package com.company.customer.adapter.systemapi.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {
    
    @Value("${system-api.customer.base-url}")
    private String baseUrl;
    
    @Value("${system-api.customer.connect-timeout-ms:5000}")
    private int connectTimeout;
    
    @Value("${system-api.customer.read-timeout-ms:10000}")
    private int readTimeout;
    
    @Bean
    public RestClient restClient() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(connectTimeout);
        factory.setReadTimeout(readTimeout);
        
        return RestClient.builder()
            .baseUrl(baseUrl)
            .requestFactory(factory)
            .defaultHeader("Content-Type", "application/json")
            .build();
    }
}
```

### Client Selection Criteria

| Criteria | Feign | RestTemplate | RestClient |
|----------|-------|--------------|------------|
| **Spring Boot** | Any | Any | 3.2+ |
| **Style** | Declarative | Imperative | Fluent |
| **Boilerplate** | Minimal | Verbose | Low |
| **Resilience4j** | Built-in support | Manual | Manual |
| **Testing** | Mock interface | Mock RestTemplate | Mock RestClient |
| **Recommendation** | ✅ New projects | Legacy only | Spring 3.2+ |

### System API Mapper

```java
// File: adapter/systemapi/mapper/CustomerSystemApiMapper.java
package com.company.customer.adapter.systemapi.mapper;

import com.company.customer.adapter.systemapi.dto.CustomerDto;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerStatus;
import org.springframework.stereotype.Component;

@Component
public class CustomerSystemApiMapper {
    
    public Customer toDomain(CustomerDto dto) {
        if (dto == null) return null;
        
        return Customer.builder()
            .id(dto.getCustomerId())
            .firstName(dto.getFirstName())
            .lastName(dto.getLastName())
            .email(dto.getEmailAddress())
            .birthDate(dto.getDateOfBirth())
            .status(CustomerStatus.valueOf(dto.getCustomerStatus()))
            .build();
    }
    
    public CustomerDto toDto(Customer domain) {
        if (domain == null) return null;
        
        return CustomerDto.builder()
            .customerId(domain.getId())
            .firstName(domain.getFirstName())
            .lastName(domain.getLastName())
            .emailAddress(domain.getEmail())
            .dateOfBirth(domain.getBirthDate())
            .customerStatus(domain.getStatus().name())
            .build();
    }
}
```

### System API Adapter with Resilience

```java
// File: adapter/systemapi/CustomerSystemApiAdapter.java
package com.company.customer.adapter.systemapi;

import com.company.customer.adapter.systemapi.client.CustomerSystemApiClient;
import com.company.customer.adapter.systemapi.dto.CustomerDto;
import com.company.customer.adapter.systemapi.mapper.CustomerSystemApiMapper;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.repository.CustomerRepository;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

@Component
@RequiredArgsConstructor
@Slf4j
public class CustomerSystemApiAdapter implements CustomerRepository {
    
    private final CustomerSystemApiClient client;
    private final CustomerSystemApiMapper mapper;
    
    private static final String SYSTEM_API = "customerSystemApi";
    
    @Override
    @CircuitBreaker(name = SYSTEM_API, fallbackMethod = "findByIdFallback")
    @Retry(name = SYSTEM_API)
    public Optional<Customer> findById(String id) {
        log.debug("Fetching customer from System API: {}", id);
        CustomerDto dto = client.findById(id);
        return Optional.ofNullable(mapper.toDomain(dto));
    }
    
    private Optional<Customer> findByIdFallback(String id, Exception ex) {
        log.warn("System API unavailable for customer: {}. Error: {}", id, ex.getMessage());
        // Strategy: return empty (or could check cache)
        return Optional.empty();
    }
    
    @Override
    @CircuitBreaker(name = SYSTEM_API, fallbackMethod = "findByEmailFallback")
    @Retry(name = SYSTEM_API)
    public List<Customer> findByEmail(String email) {
        log.debug("Searching customers by email via System API: {}", email);
        List<CustomerDto> dtos = client.findByEmail(email);
        return dtos.stream()
            .map(mapper::toDomain)
            .toList();
    }
    
    private List<Customer> findByEmailFallback(String email, Exception ex) {
        log.warn("System API unavailable for email search: {}. Error: {}", email, ex.getMessage());
        return Collections.emptyList();
    }
    
    @Override
    @CircuitBreaker(name = SYSTEM_API)
    @Retry(name = SYSTEM_API)
    public Customer save(Customer customer) {
        log.debug("Saving customer via System API: {}", customer.getId());
        CustomerDto dto = mapper.toDto(customer);
        CustomerDto saved = client.save(dto);
        return mapper.toDomain(saved);
    }
    
    @Override
    @CircuitBreaker(name = SYSTEM_API)
    @Retry(name = SYSTEM_API)
    public void deleteById(String id) {
        log.debug("Deleting customer via System API: {}", id);
        client.deleteById(id);
    }
}
```

### System API Configuration

```yaml
# application.yml
system-api:
  customer:
    base-url: ${SYSTEM_API_CUSTOMER_URL:http://localhost:8081}
    connect-timeout: 5s
    read-timeout: 10s

# Feign configuration (if using Feign)
feign:
  client:
    config:
      customer-system-api:
        connectTimeout: 5000
        readTimeout: 10000
        loggerLevel: basic

# Resilience4j configuration
resilience4j:
  circuitbreaker:
    instances:
      customerSystemApi:
        slidingWindowSize: 10
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        permittedNumberOfCallsInHalfOpenState: 3
        
  retry:
    instances:
      customerSystemApi:
        maxAttempts: 3
        waitDuration: 500ms
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2
        retryExceptions:
          - java.net.ConnectException
          - java.net.SocketTimeoutException
          - org.springframework.web.client.ResourceAccessException
        ignoreExceptions:
          - com.company.customer.exception.CustomerNotFoundException
          
  timelimiter:
    instances:
      customerSystemApi:
        timeoutDuration: 5s
        cancelRunningFuture: true

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,circuitbreakers,retries
  health:
    circuitbreakers:
      enabled: true
```

---

## Error Handling

### System API Exception Handling

```java
// File: adapter/systemapi/exception/SystemApiExceptionHandler.java
package com.company.customer.adapter.systemapi.exception;

import com.company.customer.domain.exception.CustomerNotFoundException;
import com.company.customer.domain.exception.PersistenceException;
import feign.FeignException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;

@Component
@Slf4j
public class SystemApiExceptionHandler {
    
    public RuntimeException handle(Exception ex, String operation, String entityId) {
        if (ex instanceof FeignException.NotFound || 
            ex instanceof HttpClientErrorException.NotFound) {
            return new CustomerNotFoundException("Customer not found: " + entityId);
        }
        
        if (ex instanceof FeignException.BadRequest ||
            ex instanceof HttpClientErrorException.BadRequest) {
            log.error("Bad request to System API: {}", ex.getMessage());
            return new PersistenceException("Invalid request: " + ex.getMessage(), ex);
        }
        
        if (ex instanceof HttpServerErrorException ||
            (ex instanceof FeignException && ((FeignException) ex).status() >= 500)) {
            log.error("System API server error during {}: {}", operation, ex.getMessage());
            return new PersistenceException("System API unavailable", ex);
        }
        
        log.error("Unexpected error during {}: {}", operation, ex.getMessage(), ex);
        return new PersistenceException("Persistence operation failed", ex);
    }
}
```

---

## Testing

### JPA Adapter Test

```java
@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@Testcontainers
class CustomerPersistenceAdapterTest {
    
    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15");
    
    @Autowired
    private CustomerJpaRepository jpaRepository;
    
    @Autowired
    private CustomerPersistenceMapper mapper;
    
    private CustomerPersistenceAdapter adapter;
    
    @BeforeEach
    void setUp() {
        adapter = new CustomerPersistenceAdapter(jpaRepository, mapper);
    }
    
    @Test
    void findById_existingCustomer_returnsCustomer() {
        // Arrange
        CustomerJpaEntity entity = createTestEntity();
        jpaRepository.save(entity);
        
        // Act
        Optional<Customer> result = adapter.findById(entity.getId());
        
        // Assert
        assertThat(result).isPresent();
        assertThat(result.get().getEmail()).isEqualTo(entity.getEmail());
    }
    
    @Test
    void save_newCustomer_persistsAndReturns() {
        // Arrange
        Customer customer = createTestCustomer();
        
        // Act
        Customer saved = adapter.save(customer);
        
        // Assert
        assertThat(saved.getId()).isEqualTo(customer.getId());
        assertThat(jpaRepository.findById(customer.getId())).isPresent();
    }
}
```

### System API Adapter Test

```java
@ExtendWith(MockitoExtension.class)
class CustomerSystemApiAdapterTest {
    
    @Mock
    private CustomerSystemApiClient client;
    
    @Mock
    private CustomerSystemApiMapper mapper;
    
    @InjectMocks
    private CustomerSystemApiAdapter adapter;
    
    @Test
    void findById_existingCustomer_returnsCustomer() {
        // Arrange
        String customerId = "cust-123";
        CustomerDto dto = createTestDto();
        Customer expected = createTestCustomer();
        
        when(client.findById(customerId)).thenReturn(dto);
        when(mapper.toDomain(dto)).thenReturn(expected);
        
        // Act
        Optional<Customer> result = adapter.findById(customerId);
        
        // Assert
        assertThat(result).isPresent();
        assertThat(result.get()).isEqualTo(expected);
        verify(client).findById(customerId);
    }
    
    @Test
    void findById_clientThrowsException_fallbackReturnsEmpty() {
        // Arrange
        String customerId = "cust-123";
        when(client.findById(customerId)).thenThrow(new RuntimeException("Connection refused"));
        
        // Act - Note: Fallback requires Spring context, this tests the fallback method directly
        Optional<Customer> result = adapter.findByIdFallback(customerId, new RuntimeException());
        
        // Assert
        assertThat(result).isEmpty();
    }
}
```

---

## Best Practices

### General

1. **Domain entities are pure** - No JPA or framework annotations
2. **Repository interface in domain** - Implementation in adapter
3. **Mapper per adapter** - Don't share mappers between JPA and System API
4. **Externalize configuration** - URLs, timeouts via environment variables

### JPA Specific

1. **Use `ddl-auto: validate`** in production
2. **Disable OSIV** (`open-in-view: false`)
3. **Use batch operations** for bulk inserts
4. **Version fields** for optimistic locking

### System API Specific

1. **Always use resilience patterns** - Circuit Breaker + Retry minimum
2. **Include correlation headers** - For distributed tracing
3. **Handle all HTTP status codes** - Map to domain exceptions
4. **Log at appropriate levels** - DEBUG for success, WARN for fallbacks, ERROR for failures

---

## Common Pitfalls

### Pitfall 1: JPA Annotations in Domain

```java
// ❌ WRONG - JPA in domain
package com.company.customer.domain.model;

@Entity  // NO!
public class Customer {
    @Id    // NO!
    private String id;
}

// ✅ CORRECT - Pure domain
package com.company.customer.domain.model;

public class Customer {
    private String id;
}
```

### Pitfall 2: Missing Resilience on System API

```java
// ❌ WRONG - No resilience
@Override
public Optional<Customer> findById(String id) {
    return Optional.ofNullable(mapper.toDomain(client.findById(id)));
}

// ✅ CORRECT - With resilience
@CircuitBreaker(name = SYSTEM_API, fallbackMethod = "findByIdFallback")
@Retry(name = SYSTEM_API)
@Override
public Optional<Customer> findById(String id) {
    return Optional.ofNullable(mapper.toDomain(client.findById(id)));
}
```

### Pitfall 3: Hardcoded System API URL

```java
// ❌ WRONG
@FeignClient(url = "http://system-api.internal:8080")

// ✅ CORRECT
@FeignClient(url = "${system-api.customer.base-url}")
```

---

## References

- **Implements:** ADR-011 (Persistence Patterns)
- **Related:** ADR-009 (Hexagonal Architecture)
- **Modules:** mod-code-016-persistence-jpa-spring, mod-code-017-persistence-systemapi
- **Resilience:** ERI-CODE-008 (Circuit Breaker), ERI-CODE-009 (Retry)

---

## Annex: Implementation Constraints

```yaml
eri_constraints:
  id: eri-code-012-persistence-constraints
  version: "1.1"
  eri_reference: eri-code-012-persistence-patterns-java-spring
  adr_reference: adr-011-persistence-patterns

  # Implementation options - DISPARATE (result in separate modules, not variants)
  implementation_options:
    type: disparate  # Options are NOT interchangeable - separate modules
    note: "JPA and System API are architecturally different - use separate modules"
    options:
      - id: jpa
        name: "JPA Persistence"
        description: "Local database with Spring Data JPA"
        module: mod-code-016-persistence-jpa-spring
        use_when:
          - "Service owns its data"
          - "Local PostgreSQL/MySQL database"
          - "CRUD operations without external delegation"
          
      - id: systemapi
        name: "System API Persistence"
        description: "Delegation to mainframe via REST API"
        module: mod-code-017-persistence-systemapi
        use_when:
          - "Data owned by mainframe"
          - "Service is consumer of external system"
          - "Integration with legacy systems"

  common_constraints:
    - id: repository-interface-in-domain
      rule: "Repository interface MUST be in domain/repository/"
      validation: "Repository interfaces exist in domain/repository/"
      severity: ERROR
      
    - id: domain-entity-no-framework
      rule: "Domain entities MUST NOT have JPA or framework annotations"
      validation: "No @Entity, @Table, @Id in domain/model/"
      severity: ERROR
      
    - id: adapter-implements-port
      rule: "Adapter MUST implement domain repository interface"
      validation: "Adapter class implements CustomerRepository"
      severity: ERROR

  jpa_constraints:
    - id: jpa-entity-in-adapter
      rule: "JPA entities MUST be in adapter/persistence/entity/"
      validation: "@Entity classes only in adapter/persistence/entity/"
      severity: ERROR
      
    - id: ddl-auto-validate
      rule: "ddl-auto MUST be 'validate' in production profiles"
      validation: "spring.jpa.hibernate.ddl-auto=validate in prod"
      severity: ERROR
      
    - id: osiv-disabled
      rule: "Open Session In View SHOULD be disabled"
      validation: "spring.jpa.open-in-view=false"
      severity: WARNING

  systemapi_constraints:
    - id: resilience-required
      rule: "System API calls MUST have @CircuitBreaker and @Retry"
      validation: "Adapter methods have resilience annotations"
      severity: ERROR
      
    - id: base-url-externalized
      rule: "Base URL MUST be externalized via environment variable"
      validation: "URL uses ${...} placeholder"
      severity: ERROR
      
    - id: correlation-headers
      rule: "Requests SHOULD include correlation headers"
      validation: "X-Correlation-Id header set on requests"
      severity: WARNING
      
    - id: timeout-configured
      rule: "Connection and read timeouts MUST be configured"
      validation: "Timeout configuration present"
      severity: ERROR

  dependency_constraints:
    jpa:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-data-jpa
        reason: "JPA support"
        
    systemapi_feign:
      - groupId: org.springframework.cloud
        artifactId: spring-cloud-starter-openfeign
        reason: "Feign client"
        
    systemapi_common:
      - groupId: io.github.resilience4j
        artifactId: resilience4j-spring-boot3
        reason: "Resilience patterns"

  testing_constraints:
    - id: adapter-unit-tested
      rule: "Adapter MUST have unit tests"
      validation: "Test class exists for adapter"
      severity: ERROR
      
    - id: mapper-tested
      rule: "Mapper SHOULD have unit tests"
      validation: "Test class exists for mapper"
      severity: WARNING
```

---

**Status:** ✅ Production-Ready  
**Framework:** Java/Spring Boot 3.2.x  
**Options:** JPA | System API (Feign/RestTemplate/RestClient)
