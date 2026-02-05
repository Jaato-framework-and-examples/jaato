---
id: mod-code-017-persistence-systemapi
title: "MOD-017: System API Persistence"
version: 1.3
date: 2026-02-04
status: Active
derived_from: eri-code-012-persistence-patterns-java-spring
depends_on:
  - mod-code-018-api-integration-rest-java-spring
domain: code
tags:
  - java
  - spring-boot
  - system-api
  - persistence
  - resilience
used_by:
  - skill-code-020-generate-microservice-java-spring

# ═══════════════════════════════════════════════════════════════════
# MODEL v2.0 - Capability Implementation
# ═══════════════════════════════════════════════════════════════════
implements:
  stack: java-spring
  capability: persistence
  feature: systemapi

# ═══════════════════════════════════════════════════════════════════
# DEC-041: Module Variants
# ═══════════════════════════════════════════════════════════════════
# Variants allow user selection of implementation alternatives.
# Discovery Agent detects keywords from prompt, or uses default.
# CodeGen filters templates by // Variant: header.
variants:
  http_client:
    description: "HTTP client implementation for System API calls"
    default: restclient
    options:
      restclient:
        description: "Spring 6.1+ RestClient (recommended for Spring Boot 3.2+)"
        templates:
          - client/restclient.java.tpl
        keywords:
          - restclient
          - rest client
          - webclient
      feign:
        description: "OpenFeign declarative client (requires spring-cloud-starter-openfeign)"
        templates:
          - client/feign.java.tpl
          - config/feign-config.java.tpl
        dependencies:
          - groupId: org.springframework.cloud
            artifactId: spring-cloud-starter-openfeign
        keywords:
          - feign
          - openfeign
          - declarative client
      resttemplate:
        description: "Legacy RestTemplate (deprecated, use only for compatibility)"
        templates:
          - client/resttemplate.java.tpl
        keywords:
          - resttemplate
          - rest template
          - legacy client
---

# MOD-017: System API Persistence

## Overview

Reusable templates for implementing persistence via System API delegation. The Domain API's repository delegates to a REST client that calls System APIs wrapping mainframe transactions.

**Source ERI:** [ERI-CODE-012](../../../ERIs/eri-code-012-persistence-patterns-java-spring/ERI.md)

**Depends on:** [mod-code-018-api-integration-rest-java-spring](../mod-code-018-api-integration-rest-java-spring/MODULE.md) for REST client

**Use when:** Service delegates persistence to mainframe via System APIs

> **IMPORTANT:** This module provides the adapter layer only (DTO, Mapper, Adapter).
> REST client templates are in **mod-018**. Use both modules together.

---

## CRITICAL: NO @Transactional with System API

**When persistence is via System API (HTTP), DO NOT use `@Transactional`.**

```java
// WRONG - System API uses HTTP, not database transactions
@Service
@Transactional  // <-- REMOVE THIS
public class CustomerApplicationService {

// CORRECT - No @Transactional with System API persistence
@Service
public class CustomerApplicationService {
```

**Why?**
- `@Transactional` manages **database transactions** (JPA/JDBC)
- System API persistence uses **HTTP calls**, not database
- There is no local transaction to manage
- Requires `spring-boot-starter-data-jpa` which is NOT in dependencies

**This rule OVERRIDES mod-code-015 examples** which show `@Transactional` for JPA persistence.

---

## Structure

```
mod-code-017-persistence-systemapi/
├── MODULE.md
├── templates/
│   ├── dto/
│   │   └── Dto.java.tpl
│   ├── mapper/
│   │   └── SystemApiMapper.java.tpl
│   ├── adapter/
│   │   └── SystemApiAdapter.java.tpl     # Uses client from mod-018
│   ├── config/
│   │   └── application-systemapi.yml.tpl
│   ├── exception/
│   │   └── SystemApiUnavailableException.java.tpl
│   └── test/
│       └── SystemApiAdapterTest.java.tpl
└── validation/
    ├── README.md
    └── systemapi-check.sh
```

---

## Tests Generated

This module generates the following unit tests:

| Test File | Layer | Purpose | Spring Context |
|-----------|-------|---------|----------------|
| `{{Entity}}SystemApiAdapterTest.java` | Adapter OUT | Adapter logic with mocked client and mapper | None (Mockito only) |

**Test Patterns:**
- Adapter tests: Mock the client and mapper, verify mapping and delegation
- Test findById, findAll, save, deleteById operations
- No Spring context needed - pure unit testing with Mockito
- Resilience behavior (circuit breaker, retry) should be tested separately with integration tests

> **Note:** REST client templates have been moved to
> [mod-code-018-api-integration-rest-java-spring](../mod-code-018-api-integration-rest-java-spring/MODULE.md).
> This module focuses on the persistence adapter that wraps the client with resilience patterns.

---

## Client Examples (Reference Only)

The following examples show how to use clients from mod-018. These are for documentation
purposes - actual client templates are in mod-018.

---

## Template: DTO

```java
// File: {basePackage}/adapter/systemapi/dto/{Entity}Dto.java

package {basePackage}.adapter.systemapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class {Entity}Dto {

    @JsonProperty("{idJsonField}")
    private String id;

    // Add fields matching System API contract
}
```

---

## Template: Client (RestClient - DEFAULT)

> **Client selection:** This module uses REST clients from [mod-code-018](../mod-code-018-api-integration-rest-java-spring/MODULE.md).
> **Default:** RestClient (no extra dependencies needed)
> If user explicitly requests Feign or RestTemplate, see mod-code-018 for those templates.

```java
// File: {basePackage}/adapter/out/systemapi/client/{Entity}SystemApiClient.java

package {basePackage}.adapter.out.systemapi.client;

import {basePackage}.adapter.out.systemapi.dto.{Entity}Response;
import {basePackage}.adapter.out.systemapi.dto.{Entity}Request;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.slf4j.MDC;

/**
 * REST client for System API integration.
 * Uses Spring RestClient (Spring 6.1+ / Boot 3.2+).
 *
 * @generated mod-code-017-persistence-systemapi
 */
@Component
public class {Entity}SystemApiClient {

    private final RestClient restClient;

    public {Entity}SystemApiClient(
            RestClient.Builder restClientBuilder,
            @Value("${system-api.{service-name}.base-url}") String baseUrl) {
        this.restClient = restClientBuilder
            .baseUrl(baseUrl)
            .build();
    }

    public {Entity}Response findById(String id) {
        return restClient.get()
            .uri("/{resource}/{id}", id)
            .headers(this::addCorrelationHeaders)
            .retrieve()
            .body({Entity}Response.class);
    }

    public {Entity}Response create({Entity}Request request) {
        return restClient.post()
            .uri("/{resource}")
            .headers(this::addCorrelationHeaders)
            .body(request)
            .retrieve()
            .body({Entity}Response.class);
    }

    public {Entity}Response update(String id, {Entity}Request request) {
        return restClient.put()
            .uri("/{resource}/{id}", id)
            .headers(this::addCorrelationHeaders)
            .body(request)
            .retrieve()
            .body({Entity}Response.class);
    }

    public void deleteById(String id) {
        restClient.delete()
            .uri("/{resource}/{id}", id)
            .headers(this::addCorrelationHeaders)
            .retrieve()
            .toBodilessEntity();
    }

    private void addCorrelationHeaders(org.springframework.http.HttpHeaders headers) {
        String correlationId = MDC.get("X-Correlation-ID");
        if (correlationId != null) {
            headers.set("X-Correlation-ID", correlationId);
        }
    }
}
```

### Alternative: Feign (Only if explicitly requested)

If user requests Feign, see [mod-code-018 Template 2](../mod-code-018-api-integration-rest-java-spring/MODULE.md#template-2-feign-only-if-explicitly-requested).

**Remember:** Feign requires adding `spring-cloud-starter-openfeign` to pom.xml.

---

## Template: RestClient Configuration

```java
// File: {basePackage}/infrastructure/config/RestClientConfig.java

package {basePackage}.infrastructure.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

/**
 * RestClient configuration.
 *
 * @generated mod-code-017-persistence-systemapi
 */
@Configuration
public class RestClientConfig {

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }
}
```

---

## Template: Mapper

```java
// File: {basePackage}/adapter/systemapi/mapper/{Entity}SystemApiMapper.java

package {basePackage}.adapter.systemapi.mapper;

import {basePackage}.adapter.systemapi.dto.{Entity}Dto;
import {basePackage}.domain.model.{Entity};
import org.springframework.stereotype.Component;

@Component
public class {Entity}SystemApiMapper {

    public {Entity} toDomain({Entity}Dto dto) {
        if (dto == null) return null;

        return {Entity}.builder()
            .id(dto.getId())
            // Map other fields
            .build();
    }

    public {Entity}Dto toDto({Entity} domain) {
        if (domain == null) return null;

        return {Entity}Dto.builder()
            .id(domain.getId())
            // Map other fields
            .build();
    }
}
```

---

## Field Transformations (mapping.json)

System APIs often use different data formats than domain models. The `mapping.json` file defines field transformations between domain and System API representations.

### Transformation Types

| Type | Domain Format | System API Format | Example |
|------|---------------|-------------------|---------|
| `uuid_format` | UUID with hyphens | 36 chars uppercase no hyphens | `550e8400-e29b-41d4-a716-446655440000` to `550E8400E29B41D4A716446655440000` |
| `case_conversion` | Mixed case | UPPERCASE | `John` to `JOHN` |
| `enum_to_code` | Enum value | Single char code | `ACTIVE` to `A` |
| `date_format` | LocalDate | ISO string | `2024-01-15` to `"2024-01-15"` |
| `timestamp_format` | Instant | DB2 timestamp | `2024-01-15T10:30:00Z` to `2024-01-15-10.30.00.000000` |
| `direct` | Any | Same | No transformation |

### mapping.json Structure

```json
{
  "entity": "Customer",
  "systemApi": "parties-system-api",
  "fields": [
    {
      "domain": "id",
      "domainType": "CustomerId",
      "systemApi": "CUST_ID",
      "systemApiType": "String",
      "transformation": "uuid_format"
    },
    {
      "domain": "firstName",
      "domainType": "String",
      "systemApi": "CUST_FNAME",
      "systemApiType": "String",
      "transformation": "case_conversion",
      "toDomainRule": "capitalize",
      "toSystemApiRule": "uppercase"
    },
    {
      "domain": "status",
      "domainType": "CustomerStatus",
      "systemApi": "CUST_STAT_CD",
      "systemApiType": "String",
      "transformation": "enum_to_code",
      "mappings": {
        "ACTIVE": "A",
        "INACTIVE": "I",
        "BLOCKED": "B",
        "PENDING_VERIFICATION": "P"
      }
    },
    {
      "domain": "updatedAt",
      "domainType": "Instant",
      "systemApi": "LST_UPDT_TS",
      "systemApiType": "String",
      "transformation": "timestamp_format",
      "systemApiPattern": "yyyy-MM-dd-HH.mm.ss.SSSSSS"
    }
  ]
}
```

### Generated Mapper with Transformations

When `mapping.json` is provided, the generator produces a mapper with transformation methods:

```java
@Component
public class CustomerSystemApiMapper {

    // UUID transformation
    public CustomerId toCustomerId(String systemApiId) {
        if (systemApiId == null) return null;
        // 550E8400E29B41D4A716446655440000 -> 550e8400-e29b-41d4-a716-446655440000
        String formatted = systemApiId.substring(0, 8) + "-" +
                          systemApiId.substring(8, 12) + "-" +
                          systemApiId.substring(12, 16) + "-" +
                          systemApiId.substring(16, 20) + "-" +
                          systemApiId.substring(20);
        return CustomerId.of(UUID.fromString(formatted.toLowerCase()));
    }

    public String toSystemApiId(CustomerId id) {
        if (id == null) return null;
        return id.value().toString().replace("-", "").toUpperCase();
    }

    // Case conversion
    public String toDomainName(String systemApiName) {
        if (systemApiName == null) return null;
        // JOHN -> John
        return systemApiName.substring(0, 1).toUpperCase() +
               systemApiName.substring(1).toLowerCase();
    }

    public String toSystemApiName(String domainName) {
        if (domainName == null) return null;
        return domainName.toUpperCase();
    }

    // Enum mapping
    public CustomerStatus toDomainStatus(String code) {
        return switch (code) {
            case "A" -> CustomerStatus.ACTIVE;
            case "I" -> CustomerStatus.INACTIVE;
            case "B" -> CustomerStatus.BLOCKED;
            case "P" -> CustomerStatus.PENDING_VERIFICATION;
            default -> throw new IllegalArgumentException("Unknown status: " + code);
        };
    }

    public String toSystemApiStatus(CustomerStatus status) {
        return switch (status) {
            case ACTIVE -> "A";
            case INACTIVE -> "I";
            case BLOCKED -> "B";
            case PENDING_VERIFICATION -> "P";
        };
    }

    // Timestamp transformation
    private static final DateTimeFormatter DB2_FORMAT =
        DateTimeFormatter.ofPattern("yyyy-MM-dd-HH.mm.ss.SSSSSS");

    public Instant toDomainTimestamp(String db2Timestamp) {
        if (db2Timestamp == null) return null;
        return LocalDateTime.parse(db2Timestamp, DB2_FORMAT)
            .atZone(ZoneId.of("UTC"))
            .toInstant();
    }

    public String toSystemApiTimestamp(Instant instant) {
        if (instant == null) return null;
        return DB2_FORMAT.format(instant.atZone(ZoneId.of("UTC")));
    }
}
```

### Providing mapping.json

The `mapping.json` file can be:
1. **Manually created** by the developer based on System API contract
2. **Generated as draft** by a future skill (skill-021-generate-mapping) analyzing OpenAPI specs
3. **Reviewed and validated** by the developer before code generation

**Input location:** `./inputs/mapping.json` (referenced in generation-request.json)

```json
{
  "features": {
    "integration": {
      "apis": [
        {
          "name": "parties-system-api",
          "mapping": "./inputs/mapping.json"
        }
      ]
    }
  }
}
```

---

## Template: Adapter (with Resilience)

```java
// File: {basePackage}/adapter/systemapi/{Entity}SystemApiAdapter.java

package {basePackage}.adapter.systemapi;

import {basePackage}.adapter.systemapi.client.{Entity}SystemApiClient;
import {basePackage}.adapter.systemapi.dto.{Entity}Dto;
import {basePackage}.adapter.systemapi.mapper.{Entity}SystemApiMapper;
import {basePackage}.domain.model.{Entity};
import {basePackage}.domain.repository.{Entity}Repository;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
@RequiredArgsConstructor
@Slf4j
public class {Entity}SystemApiAdapter implements {Entity}Repository {

    private final {Entity}SystemApiClient client;
    private final {Entity}SystemApiMapper mapper;

    private static final String SYSTEM_API = "{serviceName}SystemApi";

    @Override
    @CircuitBreaker(name = SYSTEM_API, fallbackMethod = "findByIdFallback")
    @Retry(name = SYSTEM_API)
    public Optional<{Entity}> findById(String id) {
        log.debug("Fetching {entity} from System API: {}", id);
        {Entity}Dto dto = client.findById(id);
        return Optional.ofNullable(mapper.toDomain(dto));
    }

    private Optional<{Entity}> findByIdFallback(String id, Exception ex) {
        log.warn("System API unavailable for {entity}: {}. Error: {}", id, ex.getMessage());
        return Optional.empty();
    }

    @Override
    @CircuitBreaker(name = SYSTEM_API)
    @Retry(name = SYSTEM_API)
    public {Entity} save({Entity} entity) {
        log.debug("Saving {entity} via System API: {}", entity.getId());
        {Entity}Dto dto = mapper.toDto(entity);
        {Entity}Dto saved = client.save(dto);
        return mapper.toDomain(saved);
    }

    @Override
    @CircuitBreaker(name = SYSTEM_API)
    @Retry(name = SYSTEM_API)
    public void deleteById(String id) {
        log.debug("Deleting {entity} via System API: {}", id);
        client.deleteById(id);
    }
}
```

---

## Template: Configuration

### application.yml

```yaml
system-api:
  {serviceName}:
    base-url: ${SYSTEM_API_{SERVICE_NAME}_URL:http://localhost:8081}
    connect-timeout: 5s
    read-timeout: 10s

# Resilience4j
resilience4j:
  circuitbreaker:
    instances:
      {serviceName}SystemApi:
        slidingWindowSize: 10
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        permittedNumberOfCallsInHalfOpenState: 3

  retry:
    instances:
      {serviceName}SystemApi:
        maxAttempts: 3
        waitDuration: 500ms
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2
        retryExceptions:
          - java.net.ConnectException
          - java.net.SocketTimeoutException
        ignoreExceptions:
          - {basePackage}.domain.exception.{Entity}NotFoundException

management:
  health:
    circuitbreakers:
      enabled: true
```

---

## Template: Dependencies

### pom.xml (RestClient - DEFAULT)

```xml
<!-- Web (includes RestClient) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<!-- Resilience4j - version 2.2.0 -->
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>${resilience4j.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
```

### pom.xml (Feign - Only if explicitly requested)

If user requests Feign, **add this dependency:**

```xml
<!-- REQUIRED for Feign - only add if using Feign -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

---

## Template: Unit Test

```java
// File: {basePackage}/adapter/systemapi/{Entity}SystemApiAdapterTest.java

package {basePackage}.adapter.systemapi;

import {basePackage}.adapter.systemapi.client.{Entity}SystemApiClient;
import {basePackage}.adapter.systemapi.dto.{Entity}Dto;
import {basePackage}.adapter.systemapi.mapper.{Entity}SystemApiMapper;
import {basePackage}.domain.model.{Entity};
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {Entity}SystemApiAdapterTest {

    @Mock
    private {Entity}SystemApiClient client;

    @Mock
    private {Entity}SystemApiMapper mapper;

    @InjectMocks
    private {Entity}SystemApiAdapter adapter;

    @Test
    void findById_existingEntity_returnsEntity() {
        // Arrange
        String id = "test-id";
        {Entity}Dto dto = new {Entity}Dto();
        {Entity} expected = {Entity}.builder().id(id).build();

        when(client.findById(id)).thenReturn(dto);
        when(mapper.toDomain(dto)).thenReturn(expected);

        // Act
        Optional<{Entity}> result = adapter.findById(id);

        // Assert
        assertThat(result).isPresent();
        assertThat(result.get().getId()).isEqualTo(id);
        verify(client).findById(id);
    }

    @Test
    void save_validEntity_returnsPersistedEntity() {
        // Arrange
        {Entity} entity = {Entity}.builder().id("test-id").build();
        {Entity}Dto dto = new {Entity}Dto();

        when(mapper.toDto(entity)).thenReturn(dto);
        when(client.save(dto)).thenReturn(dto);
        when(mapper.toDomain(dto)).thenReturn(entity);

        // Act
        {Entity} result = adapter.save(entity);

        // Assert
        assertThat(result).isNotNull();
        verify(client).save(dto);
    }
}
```

---

## Parameter Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{basePackage}` | Base Java package | `com.company.customer` |
| `{Entity}` | Entity name (PascalCase) | `Customer` |
| `{entity}` | Entity name (camelCase) | `customer` |
| `{serviceName}` | Service name (lowercase) | `customer` |
| `{SERVICE_NAME}` | Service name (UPPERCASE) | `CUSTOMER` |
| `{resourcePath}` | REST resource path | `customers` |
| `{idJsonField}` | JSON field for ID | `customer_id` |

---

## Client Selection

| Client | Status | Dependencies | When to Use |
|--------|--------|--------------|-------------|
| **RestClient** | DEFAULT | None (in spring-boot-starter-web) | Always, unless user explicitly requests another |
| **Feign** | Optional | Must add `spring-cloud-starter-openfeign` | Only if user explicitly requests |
| **RestTemplate** | Deprecated | None (in spring-boot-starter-web) | Legacy compatibility only |

**Default Rule:** Use RestClient unless user explicitly requests Feign or RestTemplate.

See [mod-code-018](../mod-code-018-api-integration-rest-java-spring/MODULE.md) for complete client templates.

---

## Determinism (v1.1)

This section defines mandatory patterns for consistent code generation.

### Mandatory Patterns

| Element | Required Pattern | Rationale |
|---------|-----------------|-----------|
| Response DTO | `record` | Immutability |
| Request DTO | `record` | Immutability |
| Mapper | `@Component` class | Single responsibility |
| Code mapping | In Mapper, NOT in Enum | Separation of concerns |

### Code Mapping Rule

**CRITICAL:** External code mapping (e.g., mainframe status codes) MUST be in the Mapper class:

```java
// CORRECT - In Mapper
private CustomerStatus toStatus(String code) {
    return switch (code) {
        case "A" -> CustomerStatus.ACTIVE;
        case "I" -> CustomerStatus.INACTIVE;
        default -> throw new IllegalArgumentException("Unknown: " + code);
    };
}

// WRONG - In Enum
public enum CustomerStatus {
    ACTIVE("A"),  // NO! Don't put codes here
    ...
}
```

### Required Annotations

```java
/**
 * @generated {skill-id} v{version}
 * @module mod-code-017-persistence-systemapi
 */
```

### Forbidden Patterns

| Pattern | Reason | Alternative |
|---------|--------|-------------|
| Lombok `@Data` on DTOs | Records cleaner | Java `record` |
| Lombok `@Builder` | Records have constructor | Record constructor |
| Code in Enum | Couples domain | Mapper class |

---

## Validation

See [validation/README.md](validation/README.md) for validation script details.

---

## Related

- **Source ERI:** [ERI-CODE-012](../../../ERIs/eri-code-012-persistence-patterns-java-spring/ERI.md)
- **Alternative:** mod-code-016-persistence-jpa-spring (for local database)
- **Resilience:** mod-001 (Circuit Breaker), mod-002 (Retry)

---

**Module Version:** 1.1
**Last Updated:** 2025-12-22
