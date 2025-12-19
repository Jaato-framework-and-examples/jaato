---
id: mod-code-015-hexagonal-base-java-spring
title: "MOD-015: Hexagonal Base - Java/Spring Boot"
version: 1.1
framework: java 17+ / spring boot 3.2.x
used_by:
  - skill-020-microservice-java-spring
---

# MOD-015: Hexagonal Base - Java/Spring Boot

## Purpose

Reusable templates for generating Hexagonal Light microservices in Java/Spring Boot.

## Directory Structure

```
{{basePackagePath}}/
├── Application.java
├── domain/
│   ├── {{entityName}}.java
│   ├── {{entityName}}Id.java
│   ├── {{entityName}}Service.java
│   ├── {{entityName}}Repository.java (port)
│   └── Register{{entityName}}Command.java
├── application/
│   ├── {{entityName}}ApplicationService.java
│   └── dto/
│       ├── {{entityName}}Response.java
│       └── Create{{entityName}}Request.java
├── adapter/
│   └── rest/
│       └── {{entityName}}Controller.java
└── infrastructure/
    ├── config/
    │   └── ApplicationConfig.java
    └── exception/
        └── GlobalExceptionHandler.java
```

## Template Variables

| Variable | Description |
|----------|-------------|
| `{{serviceName}}` | kebab-case service identifier |
| `{{serviceNamePascal}}` | PascalCase variant |
| `{{groupId}}` | Maven organization |
| `{{artifactId}}` | Maven project identifier |
| `{{basePackage}}` | Java package root |
| `{{basePackagePath}}` | Package as filesystem path |
| `{{entityName}}` | Domain entity class name |
| `{{entityNameLower}}` | Lowercase variant |
| `{{entityNamePlural}}` | Plural form for collections |

## Domain Entity Template

```java
package {{basePackage}}.domain;

public class {{entityName}} {
    private final {{entityName}}Id id;
    {{entityFields}}
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static {{entityName}} create({{constructorParameters}}) {
        return new {{entityName}}({{entityName}}Id.generate(), {{constructorArguments}}, LocalDateTime.now(), null);
    }

    public void update({{updateParameters}}) {
        {{updateLogic}}
        this.updatedAt = LocalDateTime.now();
    }
}
```

## Repository Port Template

```java
package {{basePackage}}.domain;

public interface {{entityName}}Repository {
    {{entityName}} save({{entityName}} entity);
    Optional<{{entityName}}> findById({{entityName}}Id id);
    void deleteById({{entityName}}Id id);
    boolean existsById({{entityName}}Id id);
}
```

## REST Controller Template

```java
package {{basePackage}}.adapter.rest;

@RestController
@RequestMapping("/api/v1/{{entityNamePlural}}")
@RequiredArgsConstructor
public class {{entityName}}Controller {

    private final {{entityName}}ApplicationService service;

    @PostMapping
    public ResponseEntity<{{entityName}}Response> create(@Valid @RequestBody Create{{entityName}}Request request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(service.create(request));
    }

    @GetMapping("/{id}")
    public ResponseEntity<{{entityName}}Response> getById(@PathVariable String id) {
        return ResponseEntity.ok(service.getById(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<{{entityName}}Response> update(@PathVariable String id, @Valid @RequestBody Update{{entityName}}Request request) {
        return ResponseEntity.ok(service.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable String id) {
        service.delete(id);
        return ResponseEntity.noContent().build();
    }
}
```

## Annotation Rules

| Layer | Allowed Annotations |
|-------|---------------------|
| Domain | None |
| Application | `@Service`, `@Transactional` |
| Adapter/REST | `@RestController`, `@RequestMapping` |
| Adapter/Persistence | `@Entity`, `@Repository` |

## Dependencies (pom.xml)

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
<dependency>
    <groupId>org.mapstruct</groupId>
    <artifactId>mapstruct</artifactId>
    <version>1.5.5.Final</version>
</dependency>
```
