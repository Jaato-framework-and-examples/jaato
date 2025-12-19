---
id: mod-code-016-persistence-jpa-spring
title: "MOD-016: JPA Persistence - Spring Data JPA"
version: 1.0
framework: java / spring data jpa
used_by:
  - skill-020-microservice-java-spring
---

# MOD-016: JPA Persistence - Spring Data JPA

## Purpose

Reusable templates for implementing JPA persistence using Hexagonal Architecture. Separates domain entities (pure business logic) from JPA entities (adapter layer).

**Use When**: Service owns its data as the System of Record.

## Directory Structure

```
{{basePackagePath}}/adapter/persistence/
├── {{entityName}}JpaEntity.java
├── {{entityName}}JpaRepository.java
├── {{entityName}}PersistenceMapper.java
└── {{entityName}}PersistenceAdapter.java
```

## JPA Entity Template

```java
package {{basePackage}}.adapter.persistence;

@Entity
@Table(name = "{{tableName}}")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class {{entityName}}JpaEntity {

    @Id
    private String id;

    {{entityFields}}

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
```

## Spring Data Repository Template

```java
package {{basePackage}}.adapter.persistence;

@Repository
public interface {{entityName}}JpaRepository extends JpaRepository<{{entityName}}JpaEntity, String> {
    {{customQueryMethods}}
}
```

## Persistence Mapper Template

```java
package {{basePackage}}.adapter.persistence;

@Component
public class {{entityName}}PersistenceMapper {

    public {{entityName}}JpaEntity toJpaEntity({{entityName}} domain) {
        return new {{entityName}}JpaEntity(
            domain.getId().value(),
            {{domainToJpaFieldMappings}}
            domain.getCreatedAt(),
            domain.getUpdatedAt()
        );
    }

    public {{entityName}} toDomain({{entityName}}JpaEntity entity) {
        return new {{entityName}}(
            {{entityName}}Id.of(entity.getId()),
            {{jpaToDomainFieldMappings}}
            entity.getCreatedAt(),
            entity.getUpdatedAt()
        );
    }
}
```

## Persistence Adapter Template

```java
package {{basePackage}}.adapter.persistence;

@Component
@RequiredArgsConstructor
public class {{entityName}}PersistenceAdapter implements {{entityName}}Repository {

    private final {{entityName}}JpaRepository jpaRepository;
    private final {{entityName}}PersistenceMapper mapper;

    @Override
    public {{entityName}} save({{entityName}} entity) {
        var jpaEntity = mapper.toJpaEntity(entity);
        var saved = jpaRepository.save(jpaEntity);
        return mapper.toDomain(saved);
    }

    @Override
    public Optional<{{entityName}}> findById({{entityName}}Id id) {
        return jpaRepository.findById(id.value())
            .map(mapper::toDomain);
    }

    @Override
    public void deleteById({{entityName}}Id id) {
        jpaRepository.deleteById(id.value());
    }

    @Override
    public boolean existsById({{entityName}}Id id) {
        return jpaRepository.existsById(id.value());
    }
}
```

## Configuration Template

```yaml
spring:
  datasource:
    url: jdbc:postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME}
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
    hikari:
      maximum-pool-size: ${DB_POOL_SIZE:10}
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        jdbc:
          batch_size: 50
```

## Dependencies

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
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
```
