# Example 01: Customer Domain API

## Description

Generates a complete Domain API microservice for customer management with:
- Hexagonal Light architecture
- Single entity (Customer)
- Circuit breaker enabled
- PostgreSQL persistence
- Docker support

## Input

See [input.json](./input.json)

## Expected Output Structure

```
customer-service/
├── pom.xml
├── README.md
├── Dockerfile
├── .gitignore
│
├── src/main/java/com/company/customer/
│   ├── CustomerServiceApplication.java
│   │
│   ├── domain/
│   │   ├── model/
│   │   │   ├── Customer.java              # Pure POJO
│   │   │   ├── CustomerId.java            # Value Object
│   │   │   └── CustomerRegistration.java  # Command
│   │   ├── service/
│   │   │   └── CustomerDomainService.java # NO @Service
│   │   ├── repository/
│   │   │   └── CustomerRepository.java    # Interface (port)
│   │   └── exception/
│   │       └── CustomerNotFoundException.java
│   │
│   ├── application/
│   │   └── service/
│   │       └── CustomerApplicationService.java  # @Service here
│   │
│   ├── adapter/
│   │   ├── rest/
│   │   │   ├── controller/
│   │   │   │   └── CustomerController.java
│   │   │   ├── dto/
│   │   │   │   ├── CustomerDTO.java
│   │   │   │   ├── CreateCustomerRequest.java
│   │   │   │   └── UpdateCustomerRequest.java
│   │   │   └── mapper/
│   │   │       └── CustomerDtoMapper.java
│   │   │
│   │   └── persistence/
│   │       ├── entity/
│   │       │   └── CustomerEntity.java    # @Entity here
│   │       ├── repository/
│   │       │   └── CustomerJpaRepository.java
│   │       ├── adapter/
│   │       │   └── CustomerRepositoryAdapter.java
│   │       └── mapper/
│   │           └── CustomerEntityMapper.java
│   │
│   └── infrastructure/
│       ├── config/
│       │   └── ApplicationConfig.java
│       └── exception/
│           ├── GlobalExceptionHandler.java
│           └── ErrorResponse.java
│
├── src/main/resources/
│   ├── application.yml
│   ├── application-dev.yml
│   ├── application-prod.yml
│   └── openapi.yaml
│
└── src/test/java/com/company/customer/
    ├── domain/service/
    │   └── CustomerDomainServiceTest.java  # No Spring context!
    └── adapter/rest/controller/
        └── CustomerControllerIntegrationTest.java
```

## Validation Criteria

1. **Domain Layer Purity**
   - `domain/` has NO Spring annotations
   - `domain/model/` has NO JPA annotations
   - `CustomerRepository` is an interface

2. **Proper Layer Separation**
   - `@Service` only in `application/`
   - `@Entity` only in `adapter/persistence/`
   - `@RestController` only in `adapter/rest/`

3. **Tests**
   - Domain tests use `@ExtendWith(MockitoExtension.class)`
   - Domain tests do NOT use `@SpringBootTest`

4. **Compilation**
   - `mvn clean compile` succeeds
   - `mvn test` passes

## File Count

| Category | Count |
|----------|-------|
| Java files | ~25 |
| Test files | ~5 |
| Config files | ~8 |
| **Total** | ~38 |
