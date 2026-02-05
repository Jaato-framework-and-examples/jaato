---
id: mod-code-018-api-integration-rest-java-spring
title: "MOD-018: API Integration REST - Java/Spring"
version: 1.3
date: 2026-01-19
status: Active
derived_from: eri-code-013-api-integration-rest-java-spring
domain: code
tags:
  - java
  - spring-boot
  - rest-client
  - feign
  - resttemplate
  - integration
used_by:
  - skill-code-020-generate-microservice-java-spring
  - mod-code-017-persistence-systemapi

# Variant Configuration (v1.3)
variants:
  enabled: true
  selection_mode: explicit  # User must explicitly request non-default

  default:
    id: restclient
    name: "RestClient (Spring 6.1+)"
    description: "Modern REST client, DEFAULT for Spring Boot 3.2+"
    dependencies: []  # Included in spring-boot-starter-web
    templates:
      - client/restclient.java.tpl
      - config/restclient-config.java.tpl

  alternatives:
    - id: feign
      name: "OpenFeign (Declarative)"
      description: "Declarative REST client - REQUIRES additional dependency"
      dependencies:
        - groupId: org.springframework.cloud
          artifactId: spring-cloud-starter-openfeign
      templates:
        - client/feign.java.tpl
        - config/feign-config.java.tpl
      use_when: "User explicitly requests Feign"

    - id: resttemplate
      name: "RestTemplate (Legacy)"
      description: "Traditional REST client - DEPRECATED"
      dependencies: []  # Included in spring-boot-starter-web
      templates:
        - client/resttemplate.java.tpl
        - config/resttemplate-config.java.tpl
      deprecated: true
      deprecation_reason: "RestClient is the modern replacement"

# ═══════════════════════════════════════════════════════════════════
# MODEL v2.0 - Capability Implementation
# ═══════════════════════════════════════════════════════════════════
implements:
  stack: java-spring
  capability: integration
  feature: api-rest
---

# MOD-018: API Integration REST - Java/Spring

## Overview

Reusable templates for implementing REST API integration in Java/Spring Boot applications.

**Source ERI:** [ERI-CODE-013](../../../ERIs/eri-code-013-api-integration-rest-java-spring/ERI.md)

**Use when:** Service needs to call external REST APIs (internal services, System APIs, third-party)

---

## CRITICAL: Client Selection Rules

| Client | Status | Extra Dependencies | When to Use |
|--------|--------|-------------------|-------------|
| **RestClient** | DEFAULT | None | **ALWAYS use unless user explicitly requests another** |
| **Feign** | Optional | `spring-cloud-starter-openfeign` | Only if user explicitly requests Feign |
| **RestTemplate** | Deprecated | None | Only for legacy compatibility |

### MANDATORY RULE

**If the user does NOT explicitly request Feign or RestTemplate, ALWAYS use RestClient.**

### Dependencies by Client

| Client | Required in pom.xml |
|--------|---------------------|
| RestClient | Nothing extra (in `spring-boot-starter-web`) |
| Feign | **MUST ADD:** `spring-cloud-starter-openfeign` |
| RestTemplate | Nothing extra (in `spring-boot-starter-web`) |

---

## Template 1: RestClient (DEFAULT)

**Use this by default.** No additional dependencies needed.

```java
package {{basePackage}}.adapter.out.systemapi.client;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.slf4j.MDC;

/**
 * REST client for {{ApiName}} System API.
 * Uses Spring RestClient (Spring 6.1+ / Boot 3.2+).
 *
 * @generated mod-code-018-api-integration-rest-java-spring (restclient)
 */
@Component
public class {{ApiName}}SystemApiClient {

    private final RestClient restClient;

    public {{ApiName}}SystemApiClient(
            RestClient.Builder restClientBuilder,
            @Value("${system-api.{{api-name}}.base-url}") String baseUrl) {
        this.restClient = restClientBuilder
            .baseUrl(baseUrl)
            .build();
    }

    public {{ResponseDto}} findById(String id) {
        return restClient.get()
            .uri("/{{resource}}/{id}", id)
            .headers(this::addCorrelationHeaders)
            .retrieve()
            .body({{ResponseDto}}.class);
    }

    public {{ResponseDto}} create({{RequestDto}} request) {
        return restClient.post()
            .uri("/{{resource}}")
            .headers(this::addCorrelationHeaders)
            .body(request)
            .retrieve()
            .body({{ResponseDto}}.class);
    }

    public {{ResponseDto}} update(String id, {{RequestDto}} request) {
        return restClient.put()
            .uri("/{{resource}}/{id}", id)
            .headers(this::addCorrelationHeaders)
            .body(request)
            .retrieve()
            .body({{ResponseDto}}.class);
    }

    public void delete(String id) {
        restClient.delete()
            .uri("/{{resource}}/{id}", id)
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

### RestClient Configuration

```java
package {{basePackage}}.infrastructure.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }
}
```

---

## Template 2: Feign (Only if explicitly requested)

> **REQUIRES:** Add `spring-cloud-starter-openfeign` to pom.xml

**Only use if user explicitly requests Feign.**

### Required Dependency (MUST ADD to pom.xml)

```xml
<!-- REQUIRED for Feign -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

### Main Application (MUST add @EnableFeignClients)

```java
@SpringBootApplication
@EnableFeignClients  // REQUIRED for Feign
public class {{ApplicationName}} {
    public static void main(String[] args) {
        SpringApplication.run({{ApplicationName}}.class, args);
    }
}
```

### Feign Client Interface

```java
package {{basePackage}}.adapter.out.systemapi.client;

import {{basePackage}}.adapter.out.systemapi.dto.{{ResponseDto}};
import {{basePackage}}.adapter.out.systemapi.dto.{{RequestDto}};
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.*;

/**
 * Feign client for {{ApiName}} System API.
 *
 * @generated mod-code-018-api-integration-rest-java-spring (feign)
 */
@FeignClient(
    name = "{{api-name}}-system-api",
    url = "${system-api.{{api-name}}.base-url}",
    configuration = {{ApiName}}FeignConfig.class
)
public interface {{ApiName}}SystemApiClient {

    @GetMapping("/{{resource}}/{id}")
    {{ResponseDto}} findById(@PathVariable("id") String id);

    @PostMapping("/{{resource}}")
    {{ResponseDto}} create(@RequestBody {{RequestDto}} request);

    @PutMapping("/{{resource}}/{id}")
    {{ResponseDto}} update(@PathVariable("id") String id, @RequestBody {{RequestDto}} request);

    @DeleteMapping("/{{resource}}/{id}")
    void delete(@PathVariable("id") String id);
}
```

### Feign Configuration

```java
package {{basePackage}}.adapter.out.systemapi.config;

import feign.Logger;
import feign.RequestInterceptor;
import org.slf4j.MDC;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class {{ApiName}}FeignConfig {

    @Bean
    public Logger.Level feignLoggerLevel() {
        return Logger.Level.BASIC;
    }

    @Bean
    public RequestInterceptor correlationIdInterceptor() {
        return requestTemplate -> {
            String correlationId = MDC.get("X-Correlation-ID");
            if (correlationId != null) {
                requestTemplate.header("X-Correlation-ID", correlationId);
            }
        };
    }
}
```

---

## Template 3: RestTemplate (Deprecated - Legacy only)

> **DEPRECATED:** Use RestClient instead for new code.

**Only use for legacy compatibility when explicitly requested.**

```java
package {{basePackage}}.adapter.out.systemapi.client;

import {{basePackage}}.adapter.out.systemapi.dto.{{ResponseDto}};
import {{basePackage}}.adapter.out.systemapi.dto.{{RequestDto}};
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.slf4j.MDC;

/**
 * REST client for {{ApiName}} System API.
 * Uses RestTemplate (legacy).
 *
 * @deprecated Use RestClient instead
 * @generated mod-code-018-api-integration-rest-java-spring (resttemplate)
 */
@Component
public class {{ApiName}}SystemApiClient {

    private final RestTemplate restTemplate;
    private final String baseUrl;

    public {{ApiName}}SystemApiClient(
            RestTemplate restTemplate,
            @Value("${system-api.{{api-name}}.base-url}") String baseUrl) {
        this.restTemplate = restTemplate;
        this.baseUrl = baseUrl;
    }

    public {{ResponseDto}} findById(String id) {
        HttpEntity<Void> entity = new HttpEntity<>(createHeaders());
        ResponseEntity<{{ResponseDto}}> response = restTemplate.exchange(
            baseUrl + "/{{resource}}/{id}",
            HttpMethod.GET,
            entity,
            {{ResponseDto}}.class,
            id
        );
        return response.getBody();
    }

    public {{ResponseDto}} create({{RequestDto}} request) {
        HttpEntity<{{RequestDto}}> entity = new HttpEntity<>(request, createHeaders());
        ResponseEntity<{{ResponseDto}}> response = restTemplate.exchange(
            baseUrl + "/{{resource}}",
            HttpMethod.POST,
            entity,
            {{ResponseDto}}.class
        );
        return response.getBody();
    }

    public {{ResponseDto}} update(String id, {{RequestDto}} request) {
        HttpEntity<{{RequestDto}}> entity = new HttpEntity<>(request, createHeaders());
        ResponseEntity<{{ResponseDto}}> response = restTemplate.exchange(
            baseUrl + "/{{resource}}/{id}",
            HttpMethod.PUT,
            entity,
            {{ResponseDto}}.class,
            id
        );
        return response.getBody();
    }

    public void delete(String id) {
        HttpEntity<Void> entity = new HttpEntity<>(createHeaders());
        restTemplate.exchange(
            baseUrl + "/{{resource}}/{id}",
            HttpMethod.DELETE,
            entity,
            Void.class,
            id
        );
    }

    private HttpHeaders createHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        String correlationId = MDC.get("X-Correlation-ID");
        if (correlationId != null) {
            headers.set("X-Correlation-ID", correlationId);
        }
        return headers;
    }
}
```

### RestTemplate Configuration

```java
package {{basePackage}}.infrastructure.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

---

## Configuration (application.yml)

```yaml
system-api:
  {{api-name}}:
    base-url: ${SYSTEM_API_{{API_NAME}}_URL:http://localhost:8081}
    connect-timeout: 5s
    read-timeout: 10s

# If using Feign, add:
feign:
  client:
    config:
      {{api-name}}-system-api:
        connectTimeout: 5000
        readTimeout: 10000
        loggerLevel: basic
```

---

## Usage by Other Modules

### mod-code-017-persistence-systemapi

When persistence uses System API, it depends on this module for the REST client:

```
mod-017 (persistence)
    └── uses mod-018 (integration) for REST client
```

The adapter in mod-017 wraps the client from mod-018 with resilience patterns.

---

## Related

- **Source ERI:** [ERI-CODE-013](../../../ERIs/eri-code-013-api-integration-rest-java-spring/ERI.md)
- **ADR:** [ADR-012](../../../ADRs/adr-012-api-integration-patterns/ADR.md)
- **Used by:** mod-code-017-persistence-systemapi, skill-code-020
- **Capability:** integration.api.rest
