---
id: mod-code-018-api-integration-rest-java-spring
title: "MOD-018: API Integration REST - Java/Spring"
version: 1.0
framework: java / spring boot 3.2+
used_by:
  - skill-020-microservice-java-spring
  - mod-code-017-persistence-systemapi
---

# MOD-018: API Integration REST - Java/Spring

## Purpose

Reusable templates for REST API integration in Java/Spring Boot applications. Supports three functionally equivalent client implementations.

## Client Options

| Client | Recommended For |
|--------|-----------------|
| RestClient | Spring Boot 3.2+ (default) |
| Feign | Declarative approach |
| RestTemplate | Legacy implementations |

## Directory Structure

```
{{basePackagePath}}/adapter/integration/
├── {{apiName}}Client.java
├── {{apiName}}ClientConfig.java
└── exception/
    └── {{apiName}}Exception.java
```

## RestClient Template (Recommended)

```java
package {{basePackage}}.adapter.integration;

@Component
@RequiredArgsConstructor
public class {{apiName}}Client {

    private final RestClient restClient;
    private static final String CORRELATION_HEADER = "X-Correlation-ID";

    public {{responseType}} get(String id) {
        return restClient.get()
            .uri("/{{resourcePath}}/{id}", id)
            .header(CORRELATION_HEADER, MDC.get("correlationId"))
            .retrieve()
            .body({{responseType}}.class);
    }

    public {{responseType}} create({{requestType}} request) {
        return restClient.post()
            .uri("/{{resourcePath}}")
            .header(CORRELATION_HEADER, MDC.get("correlationId"))
            .contentType(MediaType.APPLICATION_JSON)
            .body(request)
            .retrieve()
            .body({{responseType}}.class);
    }

    public {{responseType}} update(String id, {{requestType}} request) {
        return restClient.put()
            .uri("/{{resourcePath}}/{id}", id)
            .header(CORRELATION_HEADER, MDC.get("correlationId"))
            .contentType(MediaType.APPLICATION_JSON)
            .body(request)
            .retrieve()
            .body({{responseType}}.class);
    }

    public void delete(String id) {
        restClient.delete()
            .uri("/{{resourcePath}}/{id}", id)
            .header(CORRELATION_HEADER, MDC.get("correlationId"))
            .retrieve()
            .toBodilessEntity();
    }
}
```

## Feign Client Template

```java
package {{basePackage}}.adapter.integration;

@FeignClient(name = "{{apiName}}", url = "${{{apiConfigPath}}.base-url}")
public interface {{apiName}}FeignClient {

    @GetMapping("/{{resourcePath}}/{id}")
    {{responseType}} getById(@PathVariable String id);

    @PostMapping("/{{resourcePath}}")
    {{responseType}} create(@RequestBody {{requestType}} request);

    @PutMapping("/{{resourcePath}}/{id}")
    {{responseType}} update(@PathVariable String id, @RequestBody {{requestType}} request);

    @DeleteMapping("/{{resourcePath}}/{id}")
    void delete(@PathVariable String id);
}
```

## Configuration Template

```java
package {{basePackage}}.adapter.integration;

@Configuration
public class {{apiName}}ClientConfig {

    @Bean
    public RestClient {{apiNameLower}}RestClient(
            @Value("${{{apiConfigPath}}.base-url}") String baseUrl,
            @Value("${{{apiConfigPath}}.timeout:30s}") Duration timeout) {
        return RestClient.builder()
            .baseUrl(baseUrl)
            .requestFactory(new JdkClientHttpRequestFactory(
                HttpClient.newBuilder()
                    .connectTimeout(timeout)
                    .build()))
            .build();
    }
}
```

## Application Properties

```yaml
{{apiConfigPath}}:
  base-url: ${API_BASE_URL:http://localhost:8080}
  timeout: ${API_TIMEOUT:30s}
```

## Dependencies

### RestClient (Spring Boot 3.2+)
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

### Feign
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

## Validation Checklist

- [ ] Correlation headers propagated
- [ ] Base URL externalized to configuration
- [ ] Error handling implemented
- [ ] Timeout configured
