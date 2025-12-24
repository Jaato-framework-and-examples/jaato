---
id: mod-code-019-api-public-exposure-java-spring
title: "MOD-019: API Public Exposure - Java/Spring Boot"
version: 1.0
date: 2025-12-01
status: Active
derived_from: eri-code-014-api-public-exposure-java-spring
domain: code
tags:
  - java
  - spring-boot
  - rest-api
  - hateoas
  - pagination
used_by:
  - skill-code-020-generate-microservice-java-spring
---

# MOD-019: API Public Exposure - Java/Spring Boot

**Module ID:** mod-code-019-api-public-exposure-java-spring  
**Version:** 1.0  
**Source ERI:** eri-code-014-api-public-exposure-java-spring  
**Framework:** Java 17+ / Spring Boot 3.2.x  
**Used by:** skill-code-020-generate-microservice-java-spring

---

## Purpose

Provides reusable code templates for exposing REST APIs with pagination, HATEOAS, and filtering support following ADR-001 standards. This module **extends** the base hexagonal structure (mod-015) with public API exposure patterns.

**Use when:**
- Domain APIs need pagination for collection endpoints
- APIs require HATEOAS links for discoverability
- Public or Partner APIs need consistent response structures

**Composes with:**
- `mod-code-015-hexagonal-base-java-spring` (base structure)
- `mod-code-001-circuit-breaker-java-resilience4j` (if calling external services)

---

## Template Structure

```
templates/
├── dto/
│   ├── PageResponse.java.tpl         # Standard pagination wrapper
│   └── FilterRequest.java.tpl        # Base filter criteria
├── assembler/
│   └── EntityModelAssembler.java.tpl # HATEOAS link builder
└── config/
    ├── PageableConfig.java.tpl       # Pagination defaults
    └── application-pagination.yml.tpl # Pagination config
```

---

## Template Variables

### Required Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `{{basePackage}}` | string | Java base package | `com.bank.customer` |
| `{{basePackagePath}}` | string | Package as path | `com/bank/customer` |
| `{{entityName}}` | string | Entity name (PascalCase) | `Customer` |
| `{{entityNameLower}}` | string | Entity name (camelCase) | `customer` |
| `{{entityNamePlural}}` | string | Entity plural (lowercase) | `customers` |

### Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `{{defaultPageSize}}` | int | 20 | Default page size |
| `{{maxPageSize}}` | int | 100 | Maximum page size |
| `{{includeHateoas}}` | boolean | true | Generate HATEOAS support |

---

## Template Catalog

| Template | Output Path | Description |
|----------|-------------|-------------|
| `dto/PageResponse.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/in/rest/dto/PageResponse.java` | Standard paginated response |
| `dto/FilterRequest.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/in/rest/dto/{{entityName}}Filter.java` | Entity-specific filter |
| `assembler/EntityModelAssembler.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/in/rest/assembler/{{entityName}}ModelAssembler.java` | HATEOAS assembler |
| `config/PageableConfig.java.tpl` | `src/main/java/{{basePackagePath}}/infrastructure/web/PageableConfig.java` | Pagination config |
| `config/application-pagination.yml.tpl` | `src/main/resources/application-pagination.yml` | Pagination properties |

---

## Templates

### 1. PageResponse (Generic Pagination Wrapper)

**File:** `templates/dto/PageResponse.java.tpl`

```java
// Template: PageResponse.java.tpl
// Output: {{basePackagePath}}/adapter/in/rest/dto/PageResponse.java
// Purpose: Standard pagination response per ADR-001

package {{basePackage}}.adapter.in.rest.dto;

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

### 2. Entity Filter

**File:** `templates/dto/FilterRequest.java.tpl`

```java
// Template: FilterRequest.java.tpl
// Output: {{basePackagePath}}/adapter/in/rest/dto/{{entityName}}Filter.java
// Purpose: Filter criteria for {{entityName}} queries

package {{basePackage}}.adapter.in.rest.dto;

/**
 * Filter criteria for {{entityName}} queries.
 * Add domain-specific filter fields as needed.
 */
public record {{entityName}}Filter(
    String status,
    String country
) {
    public boolean hasStatus() {
        return status != null && !status.isBlank();
    }
    
    public boolean hasCountry() {
        return country != null && !country.isBlank();
    }
    
    /**
     * Returns true if any filter is active.
     */
    public boolean hasAnyFilter() {
        return hasStatus() || hasCountry();
    }
}
```

### 3. HATEOAS Model Assembler

**File:** `templates/assembler/EntityModelAssembler.java.tpl`

```java
// Template: EntityModelAssembler.java.tpl
// Output: {{basePackagePath}}/adapter/in/rest/assembler/{{entityName}}ModelAssembler.java
// Purpose: Converts {{entityName}} to HATEOAS-enabled responses

package {{basePackage}}.adapter.in.rest.assembler;

import {{basePackage}}.adapter.in.rest.{{entityName}}Controller;
import {{basePackage}}.adapter.in.rest.dto.{{entityName}}Response;
import {{basePackage}}.domain.model.{{entityName}};
import org.springframework.hateoas.server.mvc.RepresentationModelAssemblerSupport;
import org.springframework.stereotype.Component;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

/**
 * Assembler that converts {{entityName}} domain entities to {{entityName}}Response DTOs
 * with HATEOAS links per ADR-001.
 */
@Component
public class {{entityName}}ModelAssembler 
        extends RepresentationModelAssemblerSupport<{{entityName}}, {{entityName}}Response> {
    
    public {{entityName}}ModelAssembler() {
        super({{entityName}}Controller.class, {{entityName}}Response.class);
    }
    
    @Override
    public {{entityName}}Response toModel({{entityName}} entity) {
        {{entityName}}Response response = mapToResponse(entity);
        
        // Self link
        response.add(linkTo(methodOn({{entityName}}Controller.class)
            .getById(entity.getId().toString(), null))
            .withSelfRel());
        
        // Collection link
        response.add(linkTo(methodOn({{entityName}}Controller.class)
            .list(null, null, null, null))
            .withRel("collection"));
        
        // Add state-specific action links
        addActionLinks(response, entity);
        
        return response;
    }
    
    private {{entityName}}Response mapToResponse({{entityName}} entity) {
        // TODO: Map entity fields to response
        // This should be replaced with actual mapping logic
        return new {{entityName}}Response(
            entity.getId().toString()
            // Add other fields
        );
    }
    
    private void addActionLinks({{entityName}}Response response, {{entityName}} entity) {
        // Add conditional action links based on entity state
        // Example: if entity is ACTIVE, add deactivate link
        // if ("ACTIVE".equals(entity.getStatus().name())) {
        //     response.add(linkTo(methodOn({{entityName}}Controller.class)
        //         .deactivate(entity.getId().toString(), null))
        //         .withRel("deactivate"));
        // }
    }
}
```

### 4. Pageable Configuration

**File:** `templates/config/PageableConfig.java.tpl`

```java
// Template: PageableConfig.java.tpl
// Output: {{basePackagePath}}/infrastructure/web/PageableConfig.java
// Purpose: Configure pagination defaults per ADR-001

package {{basePackage}}.infrastructure.web;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.web.config.EnableSpringDataWebSupport;

/**
 * Configuration for pagination support per ADR-001.
 * Uses Spring Data's Pageable resolver with DTO serialization.
 */
@Configuration
@EnableSpringDataWebSupport(pageSerializationMode = 
    EnableSpringDataWebSupport.PageSerializationMode.VIA_DTO)
public class PageableConfig {
    // Configuration values from application.yml
}
```

### 5. Pagination Properties

**File:** `templates/config/application-pagination.yml.tpl`

```yaml
# Template: application-pagination.yml.tpl
# Output: src/main/resources/application-pagination.yml
# Purpose: Pagination configuration per ADR-001

# Pagination defaults per ADR-001
spring:
  data:
    web:
      pageable:
        default-page-size: {{defaultPageSize}}
        max-page-size: {{maxPageSize}}
        one-indexed-parameters: false  # Zero-based pagination
        page-parameter: page
        size-parameter: size
      sort:
        sort-parameter: sort

# HATEOAS configuration
  hateoas:
    use-hal-as-default-json-media-type: true
```

---

## Controller Integration

This module provides supporting classes. The actual controller implementation should follow this pattern (from ERI-014):

```java
@GetMapping
public ResponseEntity<PageResponse<{{entityName}}Response>> list(
        @RequestParam(required = false) String status,
        @RequestParam(required = false) String country,
        @PageableDefault(size = 20) Pageable pageable,
        @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
    
    {{entityName}}Filter filter = new {{entityName}}Filter(status, country);
    Page<{{entityName}}> page = service.findAll(filter, pageable);
    
    List<{{entityName}}Response> content = page.getContent().stream()
        .map(assembler::toModel)
        .toList();
    
    Links links = buildPaginationLinks(page, pageable, status, country);
    
    return ResponseEntity.ok(PageResponse.of(
        content,
        page.getNumber(),
        page.getSize(),
        page.getTotalElements(),
        page.getTotalPages(),
        links
    ));
}
```

---

## Validation (Tier 3)

### Scripts

| Script | Severity | Validates |
|--------|----------|-----------|
| `pagination-check.sh` | ERROR | PageResponse structure exists |
| `hateoas-check.sh` | ERROR | ModelAssembler exists for entities |
| `config-check.sh` | WARNING | Pagination config values |

### ERI Constraint Mapping

| ERI Constraint | Severity | Script | Check |
|----------------|----------|--------|-------|
| page-response-structure | ERROR | pagination-check.sh | PageResponse class exists with required fields |
| page-metadata-fields | ERROR | pagination-check.sh | PageMetadata has number, size, totalElements, totalPages |
| hateoas-self-link | ERROR | hateoas-check.sh | ModelAssembler adds self link |
| default-page-size | ERROR | config-check.sh | application.yml has default-page-size: 20 |
| max-page-size | ERROR | config-check.sh | application.yml has max-page-size: 100 |

---

## Usage by Skills

This module is used by:

- `skill-code-020-generate-microservice-java-spring` - When generating Domain APIs

### Layer Selection

| API Layer | Include This Module? |
|-----------|---------------------|
| Experience (BFF) | ✅ Yes (pagination for aggregated responses) |
| Composable | ⚠️ Optional (internal orchestration may not need HATEOAS) |
| Domain | ✅ Yes (external contract) |
| System | ❌ No (simpler contracts) |

---

## Dependencies Added

```xml
<!-- Spring HATEOAS -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-hateoas</artifactId>
</dependency>

<!-- OpenAPI Documentation -->
<dependency>
    <groupId>org.springdoc</groupId>
    <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
    <version>2.3.0</version>
</dependency>
```

---

## Related

- **ERI:** [eri-code-014-api-public-exposure-java-spring](../../ERIs/eri-code-014-api-public-exposure-java-spring/)
- **ADR:** [ADR-001: API Design](../../ADRs/adr-001-api-design/)
- **Base Module:** [mod-code-015-hexagonal-base-java-spring](../mod-code-015-hexagonal-base-java-spring/)
- **Skills:** skill-code-020-generate-microservice-java-spring

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 1.0 | Initial version | C4E Team |
