---
id: mod-code-019-api-public-exposure-java-spring
title: "MOD-019: API Public Exposure - Java/Spring Boot"
version: 1.2
date: 2026-01-19
updated: 2026-02-03
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

# ═══════════════════════════════════════════════════════════════════
# MODEL v2.0 - Capability Implementation
# ═══════════════════════════════════════════════════════════════════
implements:
  stack: java-spring
  capability: api-architecture
  feature: domain-api

# ═══════════════════════════════════════════════════════════════════
# DEC-035: Config Flags Published by this Module
# ═══════════════════════════════════════════════════════════════════
publishes_flags:
  hateoas: true      # Enables HATEOAS support (affects mod-015 Response.java.tpl)
  pagination: true   # Enables pagination support
---

# MOD-019: API Public Exposure - Java/Spring Boot

**Module ID:** mod-code-019-api-public-exposure-java-spring
**Version:** 1.2
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

## Config Flags Published (DEC-035)

This module **publishes** config flags that affect code generation in other modules:

| Flag | Value | Subscribers | Effect |
|------|-------|-------------|--------|
| `hateoas` | `true` | mod-015 (Response.java.tpl) | mod-015 skips Response; mod-019 generates HATEOAS version |
| `pagination` | `true` | mod-015 (Controller.java.tpl) | Controller includes pagination parameters |

These flags are collected by the Context Agent and propagate to `generation-context.json`:

```json
{
  "config_flags": {
    "hateoas": true,
    "pagination": true
  }
}
```

See [DEC-035](../../DECISION-LOG.md#dec-035) for the Config Flags Pub/Sub pattern.

---

## CRITICAL: NO Spring Data Pageable

**This module does NOT use Spring Data's `Pageable` interface.**

When persistence is via System API (HTTP), Spring Data is NOT available. Use manual pagination with `@RequestParam` instead.

```java
// WRONG - Requires spring-data-commons (NOT AVAILABLE with System API)
@GetMapping
public ResponseEntity<PageResponse<CustomerResponse>> list(
    @PageableDefault(size = 20) Pageable pageable) { }

// CORRECT - Manual pagination parameters
@GetMapping
public ResponseEntity<PageResponse<CustomerResponse>> list(
    @RequestParam(defaultValue = "0") int page,
    @RequestParam(defaultValue = "20") int size) { }
```

---

## Template Structure

```
templates/
├── dto/
│   ├── PageResponse.java.tpl         # Standard pagination wrapper
│   └── FilterRequest.java.tpl        # Base filter criteria
├── assembler/
│   └── EntityModelAssembler.java.tpl # HATEOAS link builder
├── config/
│   └── WebConfig.java.tpl            # Web configuration (NO Spring Data)
└── test/
    └── AssemblerTest.java.tpl        # HATEOAS assembler tests
```

---

## Tests Generated

This module generates the following unit tests:

| Test File | Layer | Purpose | Spring Context |
|-----------|-------|---------|----------------|
| `{{Entity}}ModelAssemblerTest.java` | Adapter IN | HATEOAS link generation | None (pure POJO) |

**Test Patterns:**
- Assembler tests: Verify HATEOAS links (self, collection)
- No Spring context needed - test link building logic directly
- Uses AssertJ for assertions

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
| `dto/PageResponse.java.tpl` | `adapter/in/rest/dto/PageResponse.java` | Standard paginated response |
| `dto/FilterRequest.java.tpl` | `adapter/in/rest/dto/{{entityName}}Filter.java` | Entity-specific filter |
| `assembler/EntityModelAssembler.java.tpl` | `adapter/in/rest/assembler/{{entityName}}ModelAssembler.java` | HATEOAS assembler |
| `config/WebConfig.java.tpl` | `infrastructure/config/WebConfig.java` | Web configuration |

---

## Templates

### 1. PageResponse (Generic Pagination Wrapper)

**File:** `templates/dto/PageResponse.java.tpl`

```java
package {{basePackage}}.adapter.in.rest.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.hateoas.Links;

import java.util.List;

/**
 * Standard page response structure following ADR-001 pagination standards.
 * Does NOT depend on Spring Data - uses manual pagination.
 *
 * @param <T> Type of content items
 * @generated mod-code-019-api-public-exposure-java-spring
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
    ) {
        public static PageMetadata of(int page, int size, long totalElements) {
            int totalPages = size > 0 ? (int) Math.ceil((double) totalElements / size) : 0;
            return new PageMetadata(page, size, totalElements, totalPages);
        }
    }

    /**
     * Factory method to create PageResponse with manual pagination.
     */
    public static <T> PageResponse<T> of(
            List<T> content,
            int page,
            int size,
            long totalElements,
            Links links) {
        return new PageResponse<>(
            content,
            PageMetadata.of(page, size, totalElements),
            links
        );
    }
}
```

### 2. Entity Filter

**File:** `templates/dto/FilterRequest.java.tpl`

```java
package {{basePackage}}.adapter.in.rest.dto;

/**
 * Filter criteria for {{entityName}} queries.
 * Add domain-specific filter fields as needed.
 *
 * @generated mod-code-019-api-public-exposure-java-spring
 */
public record {{entityName}}Filter(
    String status,
    String searchTerm
) {
    public boolean hasStatus() {
        return status != null && !status.isBlank();
    }

    public boolean hasSearchTerm() {
        return searchTerm != null && !searchTerm.isBlank();
    }

    public boolean hasAnyFilter() {
        return hasStatus() || hasSearchTerm();
    }
}
```

### 3. HATEOAS Model Assembler

**File:** `templates/assembler/EntityModelAssembler.java.tpl`

```java
package {{basePackage}}.adapter.in.rest.assembler;

import {{basePackage}}.adapter.in.rest.controller.{{entityName}}Controller;
import {{basePackage}}.adapter.in.rest.dto.{{entityName}}Response;
import {{basePackage}}.domain.model.{{entityName}};
import org.springframework.hateoas.server.mvc.RepresentationModelAssemblerSupport;
import org.springframework.stereotype.Component;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

/**
 * Assembler that converts {{entityName}} domain entities to {{entityName}}Response DTOs
 * with HATEOAS links per ADR-001.
 *
 * @generated mod-code-019-api-public-exposure-java-spring
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
            .getById(entity.getId().value().toString(), null))
            .withSelfRel());

        // Collection link
        response.add(linkTo(methodOn({{entityName}}Controller.class)
            .getAll(0, 20, null, null))
            .withRel("collection"));

        return response;
    }

    private {{entityName}}Response mapToResponse({{entityName}} entity) {
        // Mapping logic - customize based on entity fields
        return {{entityName}}Response.from(entity);
    }
}
```

### 4. Web Configuration (NO Spring Data)

**File:** `templates/config/WebConfig.java.tpl`

```java
package {{basePackage}}.infrastructure.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Web configuration for REST API.
 * Does NOT use Spring Data - pagination is manual via @RequestParam.
 *
 * @generated mod-code-019-api-public-exposure-java-spring
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    // Default pagination values (used in @RequestParam defaults)
    public static final int DEFAULT_PAGE_SIZE = 20;
    public static final int MAX_PAGE_SIZE = 100;

    // Add custom argument resolvers or converters if needed
}
```

---

## Controller Integration Pattern

**IMPORTANT:** Use `@RequestParam` for pagination, NOT `Pageable`.

```java
package {{basePackage}}.adapter.in.rest.controller;

import {{basePackage}}.adapter.in.rest.dto.*;
import {{basePackage}}.adapter.in.rest.assembler.{{entityName}}ModelAssembler;
import {{basePackage}}.application.service.{{entityName}}ApplicationService;
import org.springframework.hateoas.Links;
import org.springframework.hateoas.Link;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

@RestController
@RequestMapping("/api/v1/{{entityNamePlural}}")
public class {{entityName}}Controller {

    private final {{entityName}}ApplicationService service;
    private final {{entityName}}ModelAssembler assembler;

    // Constructor injection...

    /**
     * List with manual pagination - NO Pageable.
     */
    @GetMapping
    public ResponseEntity<PageResponse<{{entityName}}Response>> getAll(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String status,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {

        // Validate pagination params
        if (size > 100) size = 100;
        if (page < 0) page = 0;

        {{entityName}}Filter filter = new {{entityName}}Filter(status, null);

        // Get paginated results from service
        List<{{entityName}}> results = service.findAll(filter, page, size);
        long totalElements = service.count(filter);

        // Convert to response DTOs
        List<{{entityName}}Response> content = results.stream()
            .map(assembler::toModel)
            .toList();

        // Build HATEOAS links
        Links links = buildPaginationLinks(page, size, totalElements, status);

        return ResponseEntity.ok(PageResponse.of(content, page, size, totalElements, links));
    }

    private Links buildPaginationLinks(int page, int size, long totalElements, String status) {
        int totalPages = size > 0 ? (int) Math.ceil((double) totalElements / size) : 0;

        Link selfLink = linkTo(methodOn({{entityName}}Controller.class)
            .getAll(page, size, status, null)).withSelfRel();

        Link firstLink = linkTo(methodOn({{entityName}}Controller.class)
            .getAll(0, size, status, null)).withRel("first");

        Link lastLink = linkTo(methodOn({{entityName}}Controller.class)
            .getAll(Math.max(0, totalPages - 1), size, status, null)).withRel("last");

        if (page > 0) {
            Link prevLink = linkTo(methodOn({{entityName}}Controller.class)
                .getAll(page - 1, size, status, null)).withRel("prev");

            if (page < totalPages - 1) {
                Link nextLink = linkTo(methodOn({{entityName}}Controller.class)
                    .getAll(page + 1, size, status, null)).withRel("next");
                return Links.of(selfLink, firstLink, prevLink, nextLink, lastLink);
            }
            return Links.of(selfLink, firstLink, prevLink, lastLink);
        }

        if (page < totalPages - 1) {
            Link nextLink = linkTo(methodOn({{entityName}}Controller.class)
                .getAll(page + 1, size, status, null)).withRel("next");
            return Links.of(selfLink, firstLink, nextLink, lastLink);
        }

        return Links.of(selfLink, firstLink, lastLink);
    }
}
```

---

## Validation (Tier 3)

### Scripts

| Script | Severity | Validates |
|--------|----------|-----------|
| `pagination-check.sh` | ERROR | PageResponse structure exists |
| `hateoas-check.sh` | ERROR | ModelAssembler exists for entities |
| `no-spring-data-check.sh` | ERROR | No imports from org.springframework.data |

### Validation Rules

| Rule | Severity | Check |
|------|----------|-------|
| no-spring-data-pageable | ERROR | No `import org.springframework.data.domain.Pageable` |
| no-pageable-default | ERROR | No `@PageableDefault` annotation |
| manual-pagination | ERROR | Controllers use `@RequestParam` for page/size |
| page-response-structure | ERROR | PageResponse has content, page, links |

---

## Dependencies Added

```xml
<!-- Spring HATEOAS (does NOT require Spring Data) -->
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

<!-- DO NOT ADD spring-data-commons or spring-boot-starter-data-jpa -->
<!-- Pagination is manual via @RequestParam, not Pageable -->
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
| 2026-01-19 | 1.1 | Removed Spring Data Pageable dependency - use manual @RequestParam pagination for System API compatibility | C4E Team |
