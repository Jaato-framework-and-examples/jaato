---
id: mod-code-016-persistence-jpa-spring
title: "MOD-016: JPA Persistence - Spring Data JPA"
version: 1.0
date: 2025-12-01
status: Active
derived_from: eri-code-012-persistence-patterns-java-spring
domain: code
tags:
  - java
  - spring-boot
  - jpa
  - persistence
  - spring-data
used_by:
  - skill-code-020-generate-microservice-java-spring
---

# MOD-016: JPA Persistence - Spring Data JPA

## Overview

Reusable templates for implementing JPA persistence following Hexagonal Architecture. Domain entities remain pure; JPA entities live in the adapter layer.

**Source ERI:** [ERI-CODE-012](../../../ERIs/eri-code-012-persistence-patterns-java-spring/ERI.md)

**Use when:** Service owns its data (is System of Record)

---

## Template Catalog

This catalog defines the templates provided by this module and their output paths.

### Entity Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `entity/JpaEntity.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/persistence/entity/{{Entity}}JpaEntity.java` | JPA entity (adapter layer) |

### Repository Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `repository/JpaRepository.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/persistence/repository/{{Entity}}JpaRepository.java` | Spring Data JPA repository |

### Adapter Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `adapter/PersistenceAdapter.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/persistence/{{Entity}}PersistenceAdapter.java` | Repository port implementation |

### Mapper Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `mapper/PersistenceMapper.java.tpl` | `src/main/java/{{basePackagePath}}/adapter/persistence/mapper/{{Entity}}PersistenceMapper.java` | Domain ↔ JPA entity mapper |

### Configuration Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `config/application-jpa.yml.tpl` | Merge into `src/main/resources/application.yml` | JPA/Datasource configuration |
| `config/pom-jpa.xml.tpl` | Merge into `pom.xml` | Maven dependencies |

### Test Templates

| Template | Output Path | Description |
|----------|-------------|-------------|
| `test/PersistenceAdapterTest.java.tpl` | `src/test/java/{{basePackagePath}}/adapter/persistence/{{Entity}}PersistenceAdapterTest.java` | Integration tests |

---

## Template: JPA Entity

```java
// File: {basePackage}/adapter/persistence/entity/{Entity}JpaEntity.java

package {basePackage}.adapter.persistence.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "{tableName}")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class {Entity}JpaEntity {
    
    @Id
    @Column(name = "{idColumn}", length = 36)
    private String id;
    
    // Add entity fields here
    
    @Version
    private Long version;
    
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
```

---

## Template: Spring Data Repository

```java
// File: {basePackage}/adapter/persistence/repository/{Entity}JpaRepository.java

package {basePackage}.adapter.persistence.repository;

import {basePackage}.adapter.persistence.entity.{Entity}JpaEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface {Entity}JpaRepository extends JpaRepository<{Entity}JpaEntity, String> {
    
    // Custom query methods
    // List<{Entity}JpaEntity> findBy{Field}(String value);
}
```

---

## Template: Persistence Mapper

```java
// File: {basePackage}/adapter/persistence/mapper/{Entity}PersistenceMapper.java

package {basePackage}.adapter.persistence.mapper;

import {basePackage}.adapter.persistence.entity.{Entity}JpaEntity;
import {basePackage}.domain.model.{Entity};
import org.springframework.stereotype.Component;

@Component
public class {Entity}PersistenceMapper {
    
    public {Entity} toDomain({Entity}JpaEntity entity) {
        if (entity == null) return null;
        
        return {Entity}.builder()
            .id(entity.getId())
            // Map other fields
            .build();
    }
    
    public {Entity}JpaEntity toEntity({Entity} domain) {
        if (domain == null) return null;
        
        return {Entity}JpaEntity.builder()
            .id(domain.getId())
            // Map other fields
            .build();
    }
}
```

---

## Template: Persistence Adapter

```java
// File: {basePackage}/adapter/persistence/{Entity}PersistenceAdapter.java

package {basePackage}.adapter.persistence;

import {basePackage}.adapter.persistence.entity.{Entity}JpaEntity;
import {basePackage}.adapter.persistence.mapper.{Entity}PersistenceMapper;
import {basePackage}.adapter.persistence.repository.{Entity}JpaRepository;
import {basePackage}.domain.model.{Entity};
import {basePackage}.domain.repository.{Entity}Repository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

@Component
@RequiredArgsConstructor
public class {Entity}PersistenceAdapter implements {Entity}Repository {
    
    private final {Entity}JpaRepository jpaRepository;
    private final {Entity}PersistenceMapper mapper;
    
    @Override
    public Optional<{Entity}> findById(String id) {
        return jpaRepository.findById(id)
            .map(mapper::toDomain);
    }
    
    @Override
    public {Entity} save({Entity} entity) {
        {Entity}JpaEntity jpaEntity = mapper.toEntity(entity);
        {Entity}JpaEntity saved = jpaRepository.save(jpaEntity);
        return mapper.toDomain(saved);
    }
    
    @Override
    public void deleteById(String id) {
        jpaRepository.deleteById(id);
    }
}
```

---

## Template: Configuration

### application.yml

```yaml
spring:
  datasource:
    url: jdbc:postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME:{serviceName}db}
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
    hikari:
      maximum-pool-size: ${DB_POOL_SIZE:10}
      minimum-idle: 5
      connection-timeout: 30000
      
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        format_sql: true
        jdbc:
          batch_size: 50
    show-sql: false
    open-in-view: false
```

---

## Template: Dependencies

### pom.xml

```xml
<!-- Spring Data JPA -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<!-- PostgreSQL Driver -->
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
    <scope>runtime</scope>
</dependency>

<!-- H2 for testing -->
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>test</scope>
</dependency>

<!-- Testcontainers for integration tests -->
<dependency>
    <groupId>org.testcontainers</groupId>
    <artifactId>postgresql</artifactId>
    <scope>test</scope>
</dependency>
```

---

## Template: Unit Test

```java
// File: {basePackage}/adapter/persistence/{Entity}PersistenceAdapterTest.java

package {basePackage}.adapter.persistence;

import {basePackage}.adapter.persistence.entity.{Entity}JpaEntity;
import {basePackage}.adapter.persistence.mapper.{Entity}PersistenceMapper;
import {basePackage}.adapter.persistence.repository.{Entity}JpaRepository;
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
class {Entity}PersistenceAdapterTest {
    
    @Mock
    private {Entity}JpaRepository jpaRepository;
    
    @Mock
    private {Entity}PersistenceMapper mapper;
    
    @InjectMocks
    private {Entity}PersistenceAdapter adapter;
    
    @Test
    void findById_existingEntity_returnsEntity() {
        // Arrange
        String id = "test-id";
        {Entity}JpaEntity jpaEntity = new {Entity}JpaEntity();
        {Entity} expected = {Entity}.builder().id(id).build();
        
        when(jpaRepository.findById(id)).thenReturn(Optional.of(jpaEntity));
        when(mapper.toDomain(jpaEntity)).thenReturn(expected);
        
        // Act
        Optional<{Entity}> result = adapter.findById(id);
        
        // Assert
        assertThat(result).isPresent();
        assertThat(result.get().getId()).isEqualTo(id);
    }
    
    @Test
    void findById_nonExistingEntity_returnsEmpty() {
        // Arrange
        when(jpaRepository.findById("unknown")).thenReturn(Optional.empty());
        
        // Act
        Optional<{Entity}> result = adapter.findById("unknown");
        
        // Assert
        assertThat(result).isEmpty();
    }
}
```

---

## Parameter Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{basePackage}` | Base Java package | `com.company.customer` |
| `{Entity}` | Entity name (PascalCase) | `Customer` |
| `{tableName}` | Database table name | `customers` |
| `{idColumn}` | Primary key column | `customer_id` |
| `{serviceName}` | Service name (lowercase) | `customer` |

---

## Validation

See [validation/README.md](validation/README.md) for validation script details.

---

## Related

- **Source ERI:** [ERI-CODE-012](../../../ERIs/eri-code-012-persistence-patterns-java-spring/ERI.md)
- **Alternative:** mod-code-017-persistence-systemapi (for System API delegation)
