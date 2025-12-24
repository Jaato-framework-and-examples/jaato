---
id: eri-code-014-api-public-exposure-java-spring
title: "ERI-CODE-014: API Public Exposure"
sidebar_label: API Public Exposure
version: 1.0
date: 2025-12-19
status: Active
author: C4E Team
domain: code
pattern: api-public-exposure
framework: java-spring
library: spring-web
java_version: "17"
spring_boot_version: "3.2.x"
implements:
  - adr-001-api-design
tags:
  - api
  - rest
  - pagination
  - hateoas
  - filtering
  - java
  - spring
related:
  - eri-code-001-hexagonal-light-java-spring
automated_by:
  - skill-020-microservice-java-spring
---

# ERI-CODE-014: API Public Exposure

## Overview

This ERI provides reference implementations for exposing REST APIs that follow the standards defined in ADR-001. It covers pagination, HATEOAS (hypermedia), filtering, sorting, and other patterns required for APIs consumed externally.

**Implements:** ADR-001 (API Design - Model, Types & Standards)  
**Status:** Active

**When to use:**
- Domain APIs exposed to Composable or Experience layers
- Public APIs exposed through API Gateway
- Partner APIs for B2B integrations

**When NOT to use:**
- Internal microservices within a bounded context
- System APIs (simpler contracts acceptable)
- gRPC or AsyncAPI (different standards apply)

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Java | 17+ |
| **Framework** | Spring Boot | 3.2.x |
| **Web** | Spring Web MVC | 6.1.x |
| **HATEOAS** | Spring HATEOAS | 2.2.x |
| **Validation** | Jakarta Validation | 3.0.x |
| **Documentation** | SpringDoc OpenAPI | 2.3.x |

---

## Project Structure

```
{service-name}/
├── src/main/java/com/{company}/{service}/
│   ├── adapter/
│   │   └── in/
│   │       └── rest/
│   │           ├── {Entity}Controller.java
│   │           ├── dto/
│   │           │   ├── {Entity}Request.java
│   │           │   ├── {Entity}Response.java
│   │           │   └── PageResponse.java
│   │           └── assembler/
│   │               └── {Entity}ModelAssembler.java
│   └── infrastructure/
│       └── web/
│           ├── PageableConfig.java
│           └── HateoasConfig.java
├── src/main/resources/
│   └── application.yml
└── pom.xml
```

---

## Code Reference

### 1. Page Response DTO

```java
// File: adapter/in/rest/dto/PageResponse.java
// Purpose: Standard paginated response structure per ADR-001

package com.bank.customer.adapter.in.rest.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.hateoas.Links;

import java.util.List;

/**
 * Standard page response structure following ADR-001 pagination standards.
 * 
 * @param <T> Type of content items
 */
public record PageResponse<T>(
    List<T> content,
    PageMetadata page,
    @JsonProperty("_links") Links links
) {
    
    public record PageMetadata(
        int number,
        int size,
        long totalElements,
        int totalPages
    ) {}
    
    /**
     * Factory method to create PageResponse from Spring Page.
     */
    public static <T> PageResponse<T> of(
            List<T> content,
            int number,
            int size,
            long totalElements,
            int totalPages,
            Links links) {
        return new PageResponse<>(
            content,
            new PageMetadata(number, size, totalElements, totalPages),
            links
        );
    }
}
```

### 2. Entity Response with HATEOAS

```java
// File: adapter/in/rest/dto/CustomerResponse.java
// Purpose: Response DTO with HATEOAS support

package com.bank.customer.adapter.in.rest.dto;

import org.springframework.hateoas.RepresentationModel;
import org.springframework.hateoas.server.core.Relation;

import java.time.Instant;
import java.time.LocalDate;

/**
 * Customer response with HATEOAS links.
 * Extends RepresentationModel for automatic _links serialization.
 */
@Relation(collectionRelation = "customers", itemRelation = "customer")
public class CustomerResponse extends RepresentationModel<CustomerResponse> {
    
    private final String id;
    private final String firstName;
    private final String lastName;
    private final String email;
    private final LocalDate dateOfBirth;
    private final String status;
    private final Instant createdAt;
    private final Instant updatedAt;
    
    public CustomerResponse(String id, String firstName, String lastName,
                            String email, LocalDate dateOfBirth, String status,
                            Instant createdAt, Instant updatedAt) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.dateOfBirth = dateOfBirth;
        this.status = status;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }
    
    // Getters
    public String getId() { return id; }
    public String getFirstName() { return firstName; }
    public String getLastName() { return lastName; }
    public String getEmail() { return email; }
    public LocalDate getDateOfBirth() { return dateOfBirth; }
    public String getStatus() { return status; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
```

### 3. HATEOAS Model Assembler

```java
// File: adapter/in/rest/assembler/CustomerModelAssembler.java
// Purpose: Converts domain entities to HATEOAS-enabled responses

package com.bank.customer.adapter.in.rest.assembler;

import com.bank.customer.adapter.in.rest.CustomerController;
import com.bank.customer.adapter.in.rest.dto.CustomerResponse;
import com.bank.customer.domain.model.Customer;
import org.springframework.hateoas.server.mvc.RepresentationModelAssemblerSupport;
import org.springframework.stereotype.Component;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

/**
 * Assembler that converts Customer domain entities to CustomerResponse DTOs
 * with HATEOAS links.
 */
@Component
public class CustomerModelAssembler 
        extends RepresentationModelAssemblerSupport<Customer, CustomerResponse> {
    
    public CustomerModelAssembler() {
        super(CustomerController.class, CustomerResponse.class);
    }
    
    @Override
    public CustomerResponse toModel(Customer customer) {
        CustomerResponse response = new CustomerResponse(
            customer.getId().toString(),
            customer.getFirstName(),
            customer.getLastName(),
            customer.getEmail(),
            customer.getDateOfBirth(),
            customer.getStatus().name(),
            customer.getCreatedAt(),
            customer.getUpdatedAt()
        );
        
        // Add self link
        response.add(linkTo(methodOn(CustomerController.class)
            .getCustomerById(customer.getId().toString(), null))
            .withSelfRel());
        
        // Add related resources
        response.add(linkTo(methodOn(CustomerController.class)
            .getCustomerOrders(customer.getId().toString(), null, null))
            .withRel("orders"));
        
        // Add available actions based on state
        if ("ACTIVE".equals(customer.getStatus().name())) {
            response.add(linkTo(methodOn(CustomerController.class)
                .deactivateCustomer(customer.getId().toString(), null))
                .withRel("deactivate"));
        }
        
        return response;
    }
}
```

### 4. Controller with Pagination and Filtering

```java
// File: adapter/in/rest/CustomerController.java
// Purpose: REST controller implementing ADR-001 standards

package com.bank.customer.adapter.in.rest;

import com.bank.customer.adapter.in.rest.assembler.CustomerModelAssembler;
import com.bank.customer.adapter.in.rest.dto.*;
import com.bank.customer.application.service.CustomerApplicationService;
import com.bank.customer.domain.model.Customer;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.hateoas.IanaLinkRelations;
import org.springframework.hateoas.Link;
import org.springframework.hateoas.Links;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

/**
 * REST controller for Customer operations.
 * Implements pagination, HATEOAS, and filtering per ADR-001.
 */
@RestController
@RequestMapping("/api/v1/customers")
@Tag(name = "Customers", description = "Customer management operations")
public class CustomerController {
    
    private final CustomerApplicationService customerService;
    private final CustomerModelAssembler assembler;
    
    public CustomerController(CustomerApplicationService customerService,
                              CustomerModelAssembler assembler) {
        this.customerService = customerService;
        this.assembler = assembler;
    }
    
    /**
     * List customers with pagination and filtering.
     * 
     * GET /api/v1/customers?page=0&size=20&sort=lastName,asc&status=ACTIVE
     */
    @GetMapping
    @Operation(summary = "List customers", description = "Returns paginated list of customers with optional filtering")
    public ResponseEntity<PageResponse<CustomerResponse>> listCustomers(
            @Parameter(description = "Filter by status")
            @RequestParam(required = false) String status,
            
            @Parameter(description = "Filter by country")
            @RequestParam(required = false) String country,
            
            @PageableDefault(size = 20, sort = "lastName")
            Pageable pageable,
            
            @RequestHeader(value = "X-Correlation-ID", required = false) 
            String correlationId) {
        
        // Build filter criteria
        CustomerFilter filter = new CustomerFilter(status, country);
        
        // Get paginated results
        Page<Customer> customerPage = customerService.findAll(filter, pageable);
        
        // Convert to responses with HATEOAS
        List<CustomerResponse> content = customerPage.getContent().stream()
            .map(assembler::toModel)
            .toList();
        
        // Build pagination links
        Links links = buildPaginationLinks(customerPage, pageable, status, country);
        
        PageResponse<CustomerResponse> response = PageResponse.of(
            content,
            customerPage.getNumber(),
            customerPage.getSize(),
            customerPage.getTotalElements(),
            customerPage.getTotalPages(),
            links
        );
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get customer by ID.
     * 
     * GET /api/v1/customers/{customerId}
     */
    @GetMapping("/{customerId}")
    @Operation(summary = "Get customer by ID")
    public ResponseEntity<CustomerResponse> getCustomerById(
            @PathVariable String customerId,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
        
        Customer customer = customerService.getById(customerId);
        CustomerResponse response = assembler.toModel(customer);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Create a new customer.
     * 
     * POST /api/v1/customers
     */
    @PostMapping
    @Operation(summary = "Create a new customer")
    public ResponseEntity<CustomerResponse> createCustomer(
            @Valid @RequestBody CreateCustomerRequest request,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
        
        Customer customer = customerService.create(request);
        CustomerResponse response = assembler.toModel(customer);
        
        return ResponseEntity
            .created(response.getRequiredLink(IanaLinkRelations.SELF).toUri())
            .body(response);
    }
    
    /**
     * Get customer orders (sub-resource).
     * 
     * GET /api/v1/customers/{customerId}/orders
     */
    @GetMapping("/{customerId}/orders")
    @Operation(summary = "Get customer orders")
    public ResponseEntity<PageResponse<OrderResponse>> getCustomerOrders(
            @PathVariable String customerId,
            @PageableDefault(size = 20) Pageable pageable,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
        
        // Implementation would delegate to order service
        // Placeholder for structure demonstration
        return ResponseEntity.ok().build();
    }
    
    /**
     * Deactivate customer (action endpoint).
     * 
     * POST /api/v1/customers/{customerId}/deactivate
     */
    @PostMapping("/{customerId}/deactivate")
    @Operation(summary = "Deactivate a customer")
    public ResponseEntity<CustomerResponse> deactivateCustomer(
            @PathVariable String customerId,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
        
        Customer customer = customerService.deactivate(customerId);
        CustomerResponse response = assembler.toModel(customer);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Build pagination links following ADR-001 structure.
     */
    private Links buildPaginationLinks(Page<?> page, Pageable pageable, 
                                        String status, String country) {
        List<Link> links = new ArrayList<>();
        
        // Self link
        links.add(linkTo(methodOn(CustomerController.class)
            .listCustomers(status, country, pageable, null))
            .withSelfRel());
        
        // First page
        links.add(linkTo(methodOn(CustomerController.class)
            .listCustomers(status, country, pageable.first(), null))
            .withRel(IanaLinkRelations.FIRST));
        
        // Last page
        if (page.getTotalPages() > 0) {
            Pageable lastPage = pageable.withPage(page.getTotalPages() - 1);
            links.add(linkTo(methodOn(CustomerController.class)
                .listCustomers(status, country, lastPage, null))
                .withRel(IanaLinkRelations.LAST));
        }
        
        // Previous page
        if (page.hasPrevious()) {
            links.add(linkTo(methodOn(CustomerController.class)
                .listCustomers(status, country, pageable.previousOrFirst(), null))
                .withRel(IanaLinkRelations.PREV));
        }
        
        // Next page
        if (page.hasNext()) {
            links.add(linkTo(methodOn(CustomerController.class)
                .listCustomers(status, country, pageable.next(), null))
                .withRel(IanaLinkRelations.NEXT));
        }
        
        return Links.of(links);
    }
}
```

### 5. Filter DTO

```java
// File: adapter/in/rest/dto/CustomerFilter.java
// Purpose: Filter criteria for list operations

package com.bank.customer.adapter.in.rest.dto;

/**
 * Filter criteria for customer queries.
 */
public record CustomerFilter(
    String status,
    String country
) {
    public boolean hasStatus() {
        return status != null && !status.isBlank();
    }
    
    public boolean hasCountry() {
        return country != null && !country.isBlank();
    }
}
```

### 6. Pageable Configuration

```java
// File: infrastructure/web/PageableConfig.java
// Purpose: Configure pagination defaults per ADR-001

package com.bank.customer.infrastructure.web;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.web.config.EnableSpringDataWebSupport;

/**
 * Configuration for pagination support.
 * Uses Spring Data's Pageable resolver.
 */
@Configuration
@EnableSpringDataWebSupport(pageSerializationMode = 
    EnableSpringDataWebSupport.PageSerializationMode.VIA_DTO)
public class PageableConfig {
    // Default configuration from application.yml
}
```

---

## Configuration

### application.yml

```yaml
# Pagination defaults per ADR-001
spring:
  data:
    web:
      pageable:
        default-page-size: 20
        max-page-size: 100
        one-indexed-parameters: false  # Zero-based pagination
        page-parameter: page
        size-parameter: size
      sort:
        sort-parameter: sort

# HATEOAS configuration
spring:
  hateoas:
    use-hal-as-default-json-media-type: true

# OpenAPI documentation
springdoc:
  api-docs:
    path: /api-docs
  swagger-ui:
    path: /swagger-ui.html
    operationsSorter: method
```

---

## Dependencies

### Required Dependencies

```xml
<!-- pom.xml -->
<dependencies>
    <!-- Spring Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- Spring HATEOAS -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-hateoas</artifactId>
    </dependency>
    
    <!-- Spring Data (for Pageable support) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    
    <!-- Validation -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
    
    <!-- OpenAPI Documentation -->
    <dependency>
        <groupId>org.springdoc</groupId>
        <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
        <version>2.3.0</version>
    </dependency>
</dependencies>
```

---

## Testing

### Controller Test Example

```java
// File: adapter/in/rest/CustomerControllerTest.java

package com.bank.customer.adapter.in.rest;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CustomerController.class)
class CustomerControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private CustomerApplicationService customerService;
    
    @MockBean
    private CustomerModelAssembler assembler;
    
    @Test
    void listCustomers_ShouldReturnPagedResponse() throws Exception {
        // Given
        when(customerService.findAll(any(), any(Pageable.class)))
            .thenReturn(new PageImpl<>(List.of()));
        
        // When/Then
        mockMvc.perform(get("/api/v1/customers")
                .param("page", "0")
                .param("size", "20"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.content").isArray())
            .andExpect(jsonPath("$.page.number").value(0))
            .andExpect(jsonPath("$.page.size").value(20))
            .andExpect(jsonPath("$._links.self").exists());
    }
    
    @Test
    void listCustomers_WithFilter_ShouldApplyFilter() throws Exception {
        // Given
        when(customerService.findAll(any(), any(Pageable.class)))
            .thenReturn(new PageImpl<>(List.of()));
        
        // When/Then
        mockMvc.perform(get("/api/v1/customers")
                .param("status", "ACTIVE"))
            .andExpect(status().isOk());
    }
    
    @Test
    void getCustomerById_ShouldIncludeHateoasLinks() throws Exception {
        // Given customer exists
        // When/Then
        mockMvc.perform(get("/api/v1/customers/{id}", "123"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$._links.self").exists())
            .andExpect(jsonPath("$._links.orders").exists());
    }
}
```

---

## Compliance Checklist

Requirements that implementations MUST satisfy:

### Pagination (ERROR if not met)
- [ ] Collection endpoints return `PageResponse` structure
- [ ] Response includes `page.number`, `page.size`, `page.totalElements`, `page.totalPages`
- [ ] Default page size is 20
- [ ] Maximum page size is 100
- [ ] Zero-based page numbering

### HATEOAS (ERROR for public APIs, WARNING for internal)
- [ ] Responses extend `RepresentationModel` or include `_links`
- [ ] Self link present on all resources
- [ ] Collection link present when applicable
- [ ] Pagination links (first, last, prev, next) present

### Filtering (WARNING if not met)
- [ ] Filter parameters use camelCase
- [ ] Empty filters are ignored (not errors)

### General (ERROR if not met)
- [ ] OpenAPI documentation generated
- [ ] Correlation ID propagated via `X-Correlation-ID`
- [ ] Proper HTTP status codes per ADR-001

---

## Related Documentation

- **ADR:** [ADR-001: API Design](../../ADRs/adr-001-api-design/) - Standards definition
- **Module:** mod-code-019-api-public-exposure-java-spring - Derived templates
- **Skill:** skill-020-microservice-java-spring - Generation automation
- **ERI:** [ERI-001: Hexagonal Light](../eri-code-001-hexagonal-light-java-spring/) - Base architecture

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 1.0 | Initial version | C4E Team |

---

## Annex: Implementation Constraints

> This annex defines rules that MUST be respected when creating Modules or Skills
> based on this ERI. Compliance is mandatory.

```yaml
eri_constraints:
  id: eri-code-014-api-public-exposure-constraints
  version: "1.0"
  eri_reference: eri-code-014-api-public-exposure-java-spring
  adr_reference: adr-001-api-design
  
  structural_constraints:
    - id: page-response-structure
      rule: "Collection endpoints MUST return PageResponse with content, page, and _links"
      validation: "Response JSON contains $.content, $.page, $._links"
      severity: ERROR
      
    - id: page-metadata-fields
      rule: "Page metadata MUST include number, size, totalElements, totalPages"
      validation: "Response JSON contains $.page.number, $.page.size, $.page.totalElements, $.page.totalPages"
      severity: ERROR
      
    - id: hateoas-self-link
      rule: "All resources MUST include self link in _links"
      validation: "Response JSON contains $._links.self"
      severity: ERROR
      
    - id: response-extends-representation-model
      rule: "Response DTOs for public APIs SHOULD extend RepresentationModel"
      validation: "Response DTO class extends RepresentationModel or includes Links field"
      severity: WARNING
      
    - id: model-assembler-exists
      rule: "Each entity exposed via HATEOAS MUST have a ModelAssembler"
      validation: "Class named {Entity}ModelAssembler exists in assembler package"
      severity: ERROR
      
  configuration_constraints:
    - id: default-page-size
      rule: "Default page size MUST be 20"
      validation: "application.yml contains spring.data.web.pageable.default-page-size: 20"
      severity: ERROR
      
    - id: max-page-size
      rule: "Maximum page size MUST be 100"
      validation: "application.yml contains spring.data.web.pageable.max-page-size: 100"
      severity: ERROR
      
    - id: zero-indexed-pagination
      rule: "Pagination MUST be zero-indexed"
      validation: "application.yml contains spring.data.web.pageable.one-indexed-parameters: false"
      severity: ERROR
      
  dependency_constraints:
    required:
      - groupId: org.springframework.boot
        artifactId: spring-boot-starter-hateoas
        reason: "HATEOAS support for hypermedia links"
      - groupId: org.springdoc
        artifactId: springdoc-openapi-starter-webmvc-ui
        minVersion: "2.0.0"
        reason: "OpenAPI documentation generation"
        
  testing_constraints:
    - id: pagination-response-test
      rule: "Controller tests MUST verify pagination response structure"
      validation: "Test class contains assertion for $.page structure"
      severity: WARNING
      
    - id: hateoas-links-test
      rule: "Controller tests MUST verify HATEOAS links presence"
      validation: "Test class contains assertion for $._links"
      severity: WARNING
```
