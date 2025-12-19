---
id: mod-code-017-persistence-systemapi
title: "MOD-017: System API Persistence"
version: 1.2
framework: java / spring boot
used_by:
  - skill-020-microservice-java-spring
requires:
  - mod-code-018-api-integration-rest-java-spring
---

# MOD-017: System API Persistence

## Purpose

Reusable templates for implementing persistence through System API delegation, where domain APIs delegate to REST clients calling mainframe transactions.

**Use When**: Service delegates data ownership to a System of Record (e.g., mainframe).

## Directory Structure

```
{{basePackagePath}}/adapter/systemapi/
├── dto/
│   ├── {{entityName}}SystemApiRequest.java
│   └── {{entityName}}SystemApiResponse.java
├── {{entityName}}SystemApiMapper.java
├── {{entityName}}SystemApiClient.java
└── {{entityName}}SystemApiAdapter.java
```

## DTO Templates

```java
package {{basePackage}}.adapter.systemapi.dto;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class {{entityName}}SystemApiRequest {
    {{requestFields}}
}

@Data
@NoArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class {{entityName}}SystemApiResponse {
    {{responseFields}}
}
```

## Mapper Template

```java
package {{basePackage}}.adapter.systemapi;

@Component
public class {{entityName}}SystemApiMapper {

    public {{entityName}}SystemApiRequest toRequest({{entityName}} domain) {
        return new {{entityName}}SystemApiRequest(
            {{domainToRequestMappings}}
        );
    }

    public {{entityName}} toDomain({{entityName}}SystemApiResponse response) {
        return new {{entityName}}(
            {{entityName}}Id.of({{idTransformation}}),
            {{responseToDomainMappings}}
            LocalDateTime.now(),
            null
        );
    }
}
```

## Adapter Template with Resilience

```java
package {{basePackage}}.adapter.systemapi;

@Component
@RequiredArgsConstructor
@Slf4j
public class {{entityName}}SystemApiAdapter implements {{entityName}}Repository {

    private final {{entityName}}SystemApiClient client;
    private final {{entityName}}SystemApiMapper mapper;

    @Override
    @CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "saveFallback")
    @Retry(name = "{{retryName}}")
    public {{entityName}} save({{entityName}} entity) {
        var request = mapper.toRequest(entity);
        var response = client.create(request);
        return mapper.toDomain(response);
    }

    private {{entityName}} saveFallback({{entityName}} entity, Throwable t) {
        log.error("Failed to save {} via System API: {}", entity.getId(), t.getMessage());
        throw new SystemApiUnavailableException("System API unavailable", t);
    }

    @Override
    @CircuitBreaker(name = "{{circuitBreakerName}}", fallbackMethod = "findByIdFallback")
    @Retry(name = "{{retryName}}")
    public Optional<{{entityName}}> findById({{entityName}}Id id) {
        try {
            var response = client.getById(id.value());
            return Optional.of(mapper.toDomain(response));
        } catch (NotFoundException e) {
            return Optional.empty();
        }
    }

    private Optional<{{entityName}}> findByIdFallback({{entityName}}Id id, Throwable t) {
        log.error("Failed to fetch {} from System API: {}", id, t.getMessage());
        throw new SystemApiUnavailableException("System API unavailable", t);
    }
}
```

## Field Transformation Rules (mapping.json)

```json
{
  "transformations": [
    {"domain": "id", "api": "entity_id", "type": "uuid_to_string"},
    {"domain": "name", "api": "entity_name", "type": "identity"},
    {"domain": "status", "api": "status_code", "type": "enum_to_code"},
    {"domain": "createdAt", "api": "create_timestamp", "type": "datetime_to_iso"}
  ]
}
```

## Configuration Template

```yaml
systemapi:
  {{entityNameLower}}:
    base-url: ${SYSTEM_API_BASE_URL}
    timeout: ${SYSTEM_API_TIMEOUT:30s}

resilience4j:
  circuitbreaker:
    instances:
      {{circuitBreakerName}}:
        failure-rate-threshold: 50
        wait-duration-in-open-state: 60s
  retry:
    instances:
      {{retryName}}:
        max-attempts: 3
        wait-duration: 500ms
```
