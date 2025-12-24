---
id: mod-code-015-hexagonal-base-java-spring
title: "MOD-015: Hexagonal Base - Java/Spring Boot"
version: 1.2
date: 2025-12-01
updated: 2025-12-22
status: Active
derived_from: eri-code-001-hexagonal-light-java-spring
domain: code
tags:
  - java
  - spring-boot
  - hexagonal
  - architecture
  - ddd
used_by:
  - skill-code-020-generate-microservice-java-spring
---

# MOD-015: Hexagonal Base - Java/Spring Boot

**Module ID:** mod-code-015-hexagonal-base-java-spring  
**Version:** 1.2  
**Source ERI:** eri-code-001-hexagonal-light-java-spring  
**Framework:** Java 17+ / Spring Boot 3.2.x  
**Used by:** skill-code-020-generate-microservice-java-spring

---

## Purpose

Provides reusable code templates for generating Hexagonal Light microservices in Java/Spring Boot. Templates use `{{placeholder}}` variables that are replaced dynamically during code generation.

---

## Template Structure

```
templates/
├── Application.java.tpl              # Main application class
├── domain/
│   ├── Entity.java.tpl               # Domain entity (pure POJO)
│   ├── EntityId.java.tpl             # Value object for ID
│   ├── Enum.java.tpl                 # Domain enum (pure Java enum)
│   ├── Repository.java.tpl           # Port interface
│   ├── DomainService.java.tpl        # Domain service (pure POJO)
│   └── NotFoundException.java.tpl    # Domain exception
├── application/
│   ├── ApplicationService.java.tpl   # Application service (@Service)
│   └── dto/
│       ├── CreateRequest.java.tpl    # Create request DTO
│       ├── UpdateRequest.java.tpl    # Update request DTO
│       └── Response.java.tpl         # Response DTO
├── adapter/
│   └── RestController.java.tpl       # Inbound REST adapter
├── infrastructure/
│   ├── ApplicationConfig.java.tpl    # @Bean configuration
│   ├── GlobalExceptionHandler.java.tpl  # RFC 7807 exception handler
│   └── CorrelationIdFilter.java.tpl  # Request correlation tracking
├── config/
│   ├── pom.xml.tpl                   # Maven configuration
│   └── application.yml.tpl           # Spring configuration
└── test/
    └── DomainServiceTest.java.tpl    # Domain layer tests
```

---

## Template Variables

### Service-Level Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{serviceName}}` | Service name (kebab-case) | `customer-service` |
| `{{serviceNamePascal}}` | Service name (PascalCase) | `CustomerService` |
| `{{groupId}}` | Maven group ID | `com.company` |
| `{{artifactId}}` | Maven artifact ID | `customer-service` |
| `{{basePackage}}` | Java base package | `com.company.customer` |
| `{{basePackagePath}}` | Package as path | `com/company/customer` |
| `{{javaVersion}}` | Java version | `17` |
| `{{springBootVersion}}` | Spring Boot version | `3.2.0` |

### Entity-Level Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{entityName}}` | Entity name (PascalCase) | `Customer` |
| `{{entityNameLower}}` | Entity name (camelCase) | `customer` |
| `{{entityNamePlural}}` | Entity plural (lowercase) | `customers` |
| `{{entityFields}}` | List of field definitions | See templates |
| `{{entityImports}}` | Required imports for entity | `java.time.LocalDateTime` |

---

## Templates

### 1. Project Structure

#### Template: pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>{{springBootVersion}}</version>
        <relativePath/>
    </parent>
    
    <groupId>{{groupId}}</groupId>
    <artifactId>{{artifactId}}</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <name>{{serviceNamePascal}}</name>
    <description>{{serviceNamePascal}} - Hexagonal Light Architecture</description>
    
    <properties>
        <java.version>{{javaVersion}}</java.version>
        <mapstruct.version>1.5.5.Final</mapstruct.version>
        <testcontainers.version>1.19.3</testcontainers.version>
    </properties>
    
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
            <version>${mapstruct.version}</version>
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
        
        {{#features.resilience.circuit_breaker.enabled}}
        <!-- Resilience4j (Circuit Breaker) -->
        <dependency>
            <groupId>io.github.resilience4j</groupId>
            <artifactId>resilience4j-spring-boot3</artifactId>
        </dependency>
        {{/features.resilience.circuit_breaker.enabled}}
    </dependencies>
    
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.testcontainers</groupId>
                <artifactId>testcontainers-bom</artifactId>
                <version>${testcontainers.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.mapstruct</groupId>
                            <artifactId>mapstruct-processor</artifactId>
                            <version>${mapstruct.version}</version>
                        </path>
                    </annotationProcessorPaths>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

#### Template: Application.java

```java
package {{basePackage}};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class {{serviceNamePascal}}Application {
    
    public static void main(String[] args) {
        SpringApplication.run({{serviceNamePascal}}Application.class, args);
    }
}
```

#### Template: application.yml

```yaml
spring:
  application:
    name: {{serviceName}}
  
  datasource:
    url: jdbc:postgresql://localhost:5432/{{entityNamePlural}}
    username: ${DB_USERNAME:postgres}
    password: ${DB_PASSWORD:postgres}
    driver-class-name: org.postgresql.Driver
  
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
    open-in-view: false

server:
  port: 8080

management:
  server:
    port: 8081
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  endpoint:
    health:
      show-details: always
      probes:
        enabled: true

logging:
  level:
    {{basePackage}}: INFO
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} [%X{correlationId}] %-5level %logger{36} - %msg%n"

{{#features.resilience.circuit_breaker.enabled}}
# Circuit Breaker Configuration
resilience4j:
  circuitbreaker:
    instances:
      default:
        slidingWindowSize: 100
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        permittedNumberOfCallsInHalfOpenState: 10
{{/features.resilience.circuit_breaker.enabled}}
```

---

### 2. Domain Layer Templates

#### Template: Domain Entity

```java
package {{basePackage}}.domain.model;

{{entityImports}}
import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Domain entity representing a {{entityName}}.
 * Pure POJO - NO framework annotations.
 */
public class {{entityName}} {
    
    private final {{entityName}}Id id;
{{#entityFields}}
    private {{fieldType}} {{fieldName}};
{{/entityFields}}
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // Factory method for creation
    public static {{entityName}} create({{entityName}}Registration registration) {
        return new {{entityName}}(
            {{entityName}}Id.generate(),
{{#entityFields}}
            registration.{{fieldName}}(),
{{/entityFields}}
            LocalDateTime.now()
        );
    }
    
    // Constructor
    public {{entityName}}({{entityName}}Id id, {{constructorParams}}, LocalDateTime createdAt) {
        this.id = Objects.requireNonNull(id, "id must not be null");
{{#entityFields}}
{{#fieldRequired}}
        this.{{fieldName}} = Objects.requireNonNull({{fieldName}}, "{{fieldName}} must not be null");
{{/fieldRequired}}
{{^fieldRequired}}
        this.{{fieldName}} = {{fieldName}};
{{/fieldRequired}}
{{/entityFields}}
        this.createdAt = createdAt;
        this.updatedAt = createdAt;
    }
    
    // Getters
    public {{entityName}}Id getId() { return id; }
{{#entityFields}}
    public {{fieldType}} get{{fieldNamePascal}}() { return {{fieldName}}; }
{{/entityFields}}
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    
    // Update method
    public void update({{updateParams}}) {
{{#entityFields}}
{{#fieldUpdatable}}
        this.{{fieldName}} = {{fieldName}};
{{/fieldUpdatable}}
{{/entityFields}}
        this.updatedAt = LocalDateTime.now();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        {{entityName}} that = ({{entityName}}) o;
        return Objects.equals(id, that.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
```

#### Template: Value Object (ID)

```java
package {{basePackage}}.domain.model;

import java.util.Objects;
import java.util.UUID;

/**
 * Value object for {{entityName}} ID.
 * Immutable and self-validating.
 */
public record {{entityName}}Id(String value) {
    
    public {{entityName}}Id {
        Objects.requireNonNull(value, "{{entityName}}Id value must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("{{entityName}}Id value must not be blank");
        }
    }
    
    public static {{entityName}}Id generate() {
        return new {{entityName}}Id(UUID.randomUUID().toString());
    }
    
    public static {{entityName}}Id of(String value) {
        return new {{entityName}}Id(value);
    }
    
    @Override
    public String toString() {
        return value;
    }
}
```

#### Template: Domain Registration Command

```java
package {{basePackage}}.domain.model;

/**
 * Domain command for {{entityNameLower}} registration.
 * Represents intent to create a {{entityNameLower}}.
 */
public record {{entityName}}Registration(
{{#entityFields}}
    {{fieldType}} {{fieldName}}{{^last}},{{/last}}
{{/entityFields}}
) {
    public {{entityName}}Registration {
{{#entityFields}}
{{#fieldRequired}}
        if ({{fieldName}} == null{{#isString}} || {{fieldName}}.isBlank(){{/isString}}) {
            throw new IllegalArgumentException("{{fieldName}} must not be {{#isString}}blank{{/isString}}{{^isString}}null{{/isString}}");
        }
{{/fieldRequired}}
{{/entityFields}}
    }
}
```

#### Template: Domain Service

```java
package {{basePackage}}.domain.service;

import {{basePackage}}.domain.model.*;
import {{basePackage}}.domain.repository.{{entityName}}Repository;
import {{basePackage}}.domain.exception.*;

/**
 * Domain service containing business logic.
 * Pure POJO - NO Spring annotations.
 */
public class {{entityName}}DomainService {
    
    private final {{entityName}}Repository repository;
    
    public {{entityName}}DomainService({{entityName}}Repository repository) {
        this.repository = repository;
    }
    
    /**
     * Register a new {{entityNameLower}}.
     */
    public {{entityName}} register{{entityName}}({{entityName}}Registration registration) {
        // TODO: Add business validations
        
        {{entityName}} entity = {{entityName}}.create(registration);
        return repository.save(entity);
    }
    
    /**
     * Get {{entityNameLower}} by ID.
     */
    public {{entityName}} get{{entityName}}({{entityName}}Id id) {
        return repository.findById(id)
            .orElseThrow(() -> new {{entityName}}NotFoundException(id));
    }
    
    /**
     * Update {{entityNameLower}}.
     */
    public {{entityName}} update{{entityName}}({{entityName}}Id id, {{updateParams}}) {
        {{entityName}} entity = get{{entityName}}(id);
        entity.update({{updateArgs}});
        return repository.save(entity);
    }
    
    /**
     * Delete {{entityNameLower}}.
     */
    public void delete{{entityName}}({{entityName}}Id id) {
        get{{entityName}}(id); // Verify exists
        repository.deleteById(id);
    }
}
```

#### Template: Repository Interface (Port)

```java
package {{basePackage}}.domain.repository;

import {{basePackage}}.domain.model.{{entityName}};
import {{basePackage}}.domain.model.{{entityName}}Id;

import java.util.Optional;

/**
 * Repository interface (port) defined in domain layer.
 * Implementation provided by adapter layer.
 */
public interface {{entityName}}Repository {
    
    {{entityName}} save({{entityName}} entity);
    
    Optional<{{entityName}}> findById({{entityName}}Id id);
    
    void deleteById({{entityName}}Id id);
    
    boolean existsById({{entityName}}Id id);
}
```

#### Template: Domain Exception - Not Found

```java
package {{basePackage}}.domain.exception;

import {{basePackage}}.domain.model.{{entityName}}Id;

public class {{entityName}}NotFoundException extends RuntimeException {
    
    private final {{entityName}}Id {{entityNameLower}}Id;
    
    public {{entityName}}NotFoundException({{entityName}}Id id) {
        super("{{entityName}} not found: " + id.value());
        this.{{entityNameLower}}Id = id;
    }
    
    public {{entityName}}Id get{{entityName}}Id() {
        return {{entityNameLower}}Id;
    }
}
```

---

### 3. Application Layer Templates

#### Template: Application Service

```java
package {{basePackage}}.application.service;

import {{basePackage}}.adapter.rest.dto.*;
import {{basePackage}}.adapter.rest.mapper.{{entityName}}DtoMapper;
import {{basePackage}}.domain.model.*;
import {{basePackage}}.domain.service.{{entityName}}DomainService;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Application service - thin orchestration layer.
 * Spring annotations live HERE, not in domain.
 */
@Service
@Transactional
public class {{entityName}}ApplicationService {
    
    private final {{entityName}}DomainService domainService;
    private final {{entityName}}DtoMapper mapper;
    
    public {{entityName}}ApplicationService(
            {{entityName}}DomainService domainService,
            {{entityName}}DtoMapper mapper) {
        this.domainService = domainService;
        this.mapper = mapper;
    }
    
    public {{entityName}}DTO create{{entityName}}(Create{{entityName}}Request request) {
        {{entityName}}Registration registration = mapper.toRegistration(request);
        {{entityName}} entity = domainService.register{{entityName}}(registration);
        return mapper.toDTO(entity);
    }
    
    @Transactional(readOnly = true)
    public {{entityName}}DTO get{{entityName}}(String id) {
        {{entityName}} entity = domainService.get{{entityName}}({{entityName}}Id.of(id));
        return mapper.toDTO(entity);
    }
    
    public {{entityName}}DTO update{{entityName}}(String id, Update{{entityName}}Request request) {
        {{entityName}} entity = domainService.update{{entityName}}(
            {{entityName}}Id.of(id),
            {{updateArgsFromRequest}}
        );
        return mapper.toDTO(entity);
    }
    
    public void delete{{entityName}}(String id) {
        domainService.delete{{entityName}}({{entityName}}Id.of(id));
    }
}
```

---

### 4. Adapter Layer Templates

#### Template: REST Controller

```java
package {{basePackage}}.adapter.rest.controller;

import {{basePackage}}.adapter.rest.dto.*;
import {{basePackage}}.application.service.{{entityName}}ApplicationService;

import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/{{entityNamePlural}}")
public class {{entityName}}Controller {
    
    private final {{entityName}}ApplicationService applicationService;
    
    public {{entityName}}Controller({{entityName}}ApplicationService applicationService) {
        this.applicationService = applicationService;
    }
    
    @PostMapping
    public ResponseEntity<{{entityName}}DTO> create{{entityName}}(
            @Valid @RequestBody Create{{entityName}}Request request) {
        {{entityName}}DTO result = applicationService.create{{entityName}}(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(result);
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<{{entityName}}DTO> get{{entityName}}(@PathVariable String id) {
        {{entityName}}DTO result = applicationService.get{{entityName}}(id);
        return ResponseEntity.ok(result);
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<{{entityName}}DTO> update{{entityName}}(
            @PathVariable String id,
            @Valid @RequestBody Update{{entityName}}Request request) {
        {{entityName}}DTO result = applicationService.update{{entityName}}(id, request);
        return ResponseEntity.ok(result);
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete{{entityName}}(@PathVariable String id) {
        applicationService.delete{{entityName}}(id);
        return ResponseEntity.noContent().build();
    }
}
```

#### Template: DTO Response

```java
package {{basePackage}}.adapter.rest.dto;

import java.time.LocalDateTime;

public record {{entityName}}DTO(
    String id,
{{#entityFields}}
    {{fieldType}} {{fieldName}},
{{/entityFields}}
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {}
```

#### Template: Create Request DTO

```java
package {{basePackage}}.adapter.rest.dto;

import jakarta.validation.constraints.*;

public record Create{{entityName}}Request(
{{#entityFields}}
{{#fieldValidations}}
    {{validationAnnotation}}
{{/fieldValidations}}
    {{fieldType}} {{fieldName}}{{^last}},{{/last}}
{{/entityFields}}
) {}
```

---

### 5. Infrastructure Templates

#### Template: Application Config

```java
package {{basePackage}}.infrastructure.config;

import {{basePackage}}.domain.repository.{{entityName}}Repository;
import {{basePackage}}.domain.service.{{entityName}}DomainService;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration for wiring domain layer beans.
 */
@Configuration
public class ApplicationConfig {
    
    @Bean
    public {{entityName}}DomainService {{entityNameLower}}DomainService({{entityName}}Repository repository) {
        return new {{entityName}}DomainService(repository);
    }
}
```

#### Template: Global Exception Handler

```java
package {{basePackage}}.infrastructure.exception;

import {{basePackage}}.domain.exception.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.time.LocalDateTime;
import java.util.List;

@RestControllerAdvice
public class GlobalExceptionHandler {
    
    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);
    
    @ExceptionHandler({{entityName}}NotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound({{entityName}}NotFoundException ex) {
        log.warn("Entity not found: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
            .body(new ErrorResponse(
                LocalDateTime.now(),
                HttpStatus.NOT_FOUND.value(),
                "Not Found",
                ex.getMessage(),
                null
            ));
    }
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException ex) {
        List<ErrorResponse.FieldError> fieldErrors = ex.getBindingResult()
            .getFieldErrors()
            .stream()
            .map(fe -> new ErrorResponse.FieldError(fe.getField(), fe.getDefaultMessage()))
            .toList();
        
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
            .body(new ErrorResponse(
                LocalDateTime.now(),
                HttpStatus.BAD_REQUEST.value(),
                "Validation Failed",
                "Request validation failed",
                fieldErrors
            ));
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneric(Exception ex) {
        log.error("Unexpected error", ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(new ErrorResponse(
                LocalDateTime.now(),
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "Internal Server Error",
                "An unexpected error occurred",
                null
            ));
    }
}
```

---

### 6. Test Templates

#### Template: Domain Service Unit Test

```java
package {{basePackage}}.domain.service;

import {{basePackage}}.domain.model.*;
import {{basePackage}}.domain.repository.{{entityName}}Repository;
import {{basePackage}}.domain.exception.*;

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
 */
@ExtendWith(MockitoExtension.class)
class {{entityName}}DomainServiceTest {
    
    @Mock
    private {{entityName}}Repository repository;
    
    private {{entityName}}DomainService domainService;
    
    @BeforeEach
    void setUp() {
        domainService = new {{entityName}}DomainService(repository);
    }
    
    @Test
    void register{{entityName}}_WithValidData_Creates{{entityName}}() {
        // Given
        var registration = new {{entityName}}Registration({{testRegistrationArgs}});
        when(repository.save(any({{entityName}}.class))).thenAnswer(inv -> inv.getArgument(0));
        
        // When
        {{entityName}} result = domainService.register{{entityName}}(registration);
        
        // Then
        assertThat(result).isNotNull();
        assertThat(result.getId()).isNotNull();
        verify(repository).save(any({{entityName}}.class));
    }
    
    @Test
    void get{{entityName}}_WhenNotFound_ThrowsException() {
        // Given
        var id = {{entityName}}Id.of("non-existent");
        when(repository.findById(id)).thenReturn(Optional.empty());
        
        // When/Then
        assertThatThrownBy(() -> domainService.get{{entityName}}(id))
            .isInstanceOf({{entityName}}NotFoundException.class);
    }
}
```

---

## Best Practices

1. **Domain Purity is Non-Negotiable**
   - Domain layer must have ZERO Spring/JPA imports
   - Use constructor injection with interfaces, not `@Autowired`
   - Domain services are POJOs instantiated via `@Bean` in infrastructure

2. **Layer Direction**
   - Dependencies flow INWARD: Adapter → Application → Domain
   - Never import from outer layers into inner layers

3. **Annotation Placement**
   - `@Service`, `@Transactional` → Application layer ONLY
   - `@Entity`, `@Repository` → Adapter/Persistence layer ONLY
   - `@RestController`, `@RequestMapping` → Adapter/REST layer ONLY
   - Domain layer: NO Spring annotations whatsoever

4. **Testing Strategy**
   - Domain tests: Pure unit tests with Mockito (NO Spring context)
   - Application tests: `@SpringBootTest` only when testing transactions
   - Adapter tests: Use `@WebMvcTest`, `@DataJpaTest`, Testcontainers

5. **Value Objects for IDs**
   - Use `record` types for entity IDs (e.g., `CustomerId`, `OrderId`)
   - Provides type safety and prevents primitive obsession

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| `@Entity` in domain | JPA leaks into domain, breaks purity | Entity classes go in `adapter/persistence/entity/`, domain uses POJOs |
| `@Repository` on domain interface | Spring annotation in domain | Domain defines plain interface; adapter implements with `@Repository` |
| `@Service` on domain service | Spring dependency in domain | Use `@Bean` factory in `infrastructure/config/` |
| `@Transactional` in domain | Spring TX management in domain | Move to Application Service methods |
| Direct repository injection | Couples domain to persistence | Inject domain repository interface, implement in adapter |
| `@SpringBootTest` for domain tests | Slow tests, unnecessary context | Use `@ExtendWith(MockitoExtension.class)` for pure unit tests |
| Returning entities from adapters | JPA entities leak to domain | Map to domain models at adapter boundary |

---

## Usage Notes

1. **Variable Replacement:** All `{{variable}}` placeholders are replaced by the skill during generation
2. **Conditional Blocks:** `{{#condition}}...{{/condition}}` are included only if condition is true
3. **Iteration:** `{{#list}}...{{/list}}` repeats for each item in the list
4. **Features:** Templates conditionally include code based on enabled features

---

## Related

- **Source ERI:** eri-code-001-hexagonal-light-java-spring
- **Used by:** skill-code-020-generate-microservice-java-spring
- **Feature modules:** mod-code-001-circuit-breaker-java-resilience4j (for resilience feature)

---

## Persistence Implementation Note

This module defines the **structural foundation** for Hexagonal Light, including the `domain/repository/` interface (port). However, **this module does NOT include persistence adapter implementations**.

The Repository interface defined in `domain/repository/{{entityName}}Repository.java` is a **port** that must be implemented by an adapter. Two options are available:

| Persistence Type | Adapter Location | Module | Use When |
|------------------|------------------|--------|----------|
| **JPA** (local DB) | `adapter/persistence/` | [mod-code-016-persistence-jpa-spring](../mod-code-016-persistence-jpa-spring/) | Service owns its data |
| **System API** (mainframe) | `adapter/systemapi/` | [mod-code-017-persistence-systemapi](../mod-code-017-persistence-systemapi/) | Service delegates to mainframe |

**Architecture:**

```
domain/repository/CustomerRepository.java  ← Interface (Port) - defined here
        │
        │ implements
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
adapter/persistence/              adapter/systemapi/
CustomerPersistenceAdapter.java   CustomerSystemApiAdapter.java
(mod-016)                         (mod-017)
```

For complete persistence implementation patterns, see:
- **ERI-CODE-012:** Persistence Patterns (reference implementation)
- **ADR-011:** Persistence Patterns (decision record)

---

## Determinism (v1.2)

This section defines mandatory patterns for consistent code generation.

### Mandatory Patterns

| Element | Required Pattern | Rationale |
|---------|-----------------|-----------|
| Entity ID | `record` with `UUID` | Immutability, type safety |
| Request DTO | `record` | Immutability |
| Response DTO (no HATEOAS) | `record` | Immutability |
| Response DTO (HATEOAS) | `class extends RepresentationModel` | Framework requirement |
| Domain Entity | `class` | Mutable lifecycle |
| Domain Enum | Simple (no attributes) | Mapping in Mapper |

### Configurable Elements (Variants)

| Element | Options | Default | Selection Criteria |
|---------|---------|---------|-------------------|
| Response DTO | `record`, `hateoas` | `record` | HATEOAS enabled → `hateoas` |

### Required Annotations

All generated files MUST include:

```java
/**
 * @generated {skill-id} v{version}
 * @module mod-code-015-hexagonal-base-java-spring
 */
```

### Forbidden Patterns

| Pattern | Reason | Alternative |
|---------|--------|-------------|
| Lombok `@Data` | Records cleaner | Java `record` |
| `String` for IDs | Type safety | `UUID` in record |
| Enum with code attribute | Coupling | Mapper class |

---

**Module Version:** 1.2  
**Last Updated:** 2025-12-22
