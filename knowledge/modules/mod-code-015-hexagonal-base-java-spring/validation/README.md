# Hexagonal Architecture Validation

## Purpose

Validates that Hexagonal Light architecture is correctly implemented according to ADR-009.

## Module

**Module:** mod-code-015-hexagonal-base-java-spring  
**Pattern:** Hexagonal Light Architecture  
**Framework:** Spring Boot  
**Implements:** ADR-009 (Service Architecture Patterns)

## What This Validates

### 1. Layer Structure (CRITICAL)
- ✅ Domain layer exists
- ✅ Application layer exists
- ✅ Adapter layer exists
- ⚠️ Infrastructure layer exists (optional)

### 2. Domain Purity (CRITICAL - ADR-009)
- ✅ **Zero Spring annotations** in domain (`@Service`, `@Component`, `@Autowired`)
- ✅ **Zero JPA annotations** in domain/model (`@Entity`, `@Table`, `@Column`)
- ✅ Repository is **interface** (not implementation)
- ✅ Repository has **no @Repository** annotation (Spring-free)

### 3. Application Layer
- ✅ Application services have `@Service` annotation
- ✅ Application services use `@Transactional` where appropriate

### 4. Adapter Layer
- ✅ Repository **adapter implements** domain interface
- ✅ JPA entities (`@Entity`) are in adapter layer
- ✅ `@Repository` annotations in adapter layer (not domain)

### 5. Tests
- ✅ Domain tests **don't use @SpringBootTest**
- ✅ Domain tests use Mockito (pure unit tests)

## Usage

```bash
# With base package path
./hexagonal-structure-check.sh /path/to/service com/company/customer

# Auto-detect structure
./hexagonal-structure-check.sh /path/to/service
```

## Arguments

1. `SERVICE_DIR` - Path to service root directory
2. `BASE_PACKAGE_PATH` (optional) - Package path (e.g., `com/company/customer`)

If BASE_PACKAGE_PATH not provided, script will try to auto-detect hexagonal structure.

## Checks Performed

| Check | Severity | ADR-009 | Description |
|-------|----------|---------|-------------|
| Domain layer exists | FAIL | ✓ | Directory structure |
| Application layer exists | FAIL | ✓ | Directory structure |
| Adapter layer exists | FAIL | ✓ | Directory structure |
| Domain has NO Spring | FAIL | ✓ | Domain purity |
| Domain model has NO JPA | FAIL | ✓ | Domain purity |
| Repository is interface | FAIL | ✓ | Domain interface |
| Repository NO @Repository | FAIL | ✓ | Domain purity |
| Application has @Service | WARN | ✓ | Application layer |
| Adapter implements interface | WARN | ✓ | Adapter pattern |
| JPA in adapter layer | WARN | ✓ | Separation of concerns |
| @Repository in adapter | INFO | ✓ | Adapter annotations |
| Domain tests NO Spring | FAIL | ✓ | Test purity |
| Domain tests use Mockito | WARN | ✓ | Test best practice |

## Exit Codes

- `0`: All critical checks passed (ADR-009 compliant)
- `1`: One or more critical checks failed (ADR-009 violations)

Warnings do not cause failure.

## Example Output

**Success (ADR-009 Compliant):**
```
✅ PASS: Domain layer exists
✅ PASS: Application layer exists
✅ PASS: Adapter layer exists
⚠️  WARN: Infrastructure layer not found (optional)
✅ PASS: Domain layer is Spring-free (ADR-009 compliant)
✅ PASS: Domain model is JPA-free (ADR-009 compliant)
✅ PASS: Repository is interface in domain layer
✅ PASS: Repository interface has no Spring annotations
✅ PASS: Application services have @Service annotation
✅ PASS: Repository adapter implements domain interface
✅ PASS: JPA @Entity annotations in adapter layer (correct)
✅ PASS: @Repository annotations in adapter layer (correct)
✅ PASS: Domain tests don't use Spring context (ADR-009 compliant)
✅ PASS: Domain tests use Mockito (recommended)

✅ Hexagonal Architecture: VALIDATED (ADR-009 Compliant)
```

**Failure (ADR-009 Violations):**
```
✅ PASS: Domain layer exists
✅ PASS: Application layer exists
✅ PASS: Adapter layer exists
❌ FAIL: Spring annotations found in domain layer (ADR-009 violation)
domain/service/CustomerDomainService.java:@Service
domain/repository/CustomerRepository.java:@Repository
❌ FAIL: JPA annotations found in domain model (ADR-009 violation)
domain/model/Customer.java:@Entity
domain/model/Customer.java:@Table

❌ Hexagonal Architecture: VALIDATION FAILED
   Errors: 2 (ADR-009 violations)
```

## ADR-009 Key Principles

### Domain Layer (Pure, Framework-Free)

**✅ CORRECT:**
```java
// domain/service/CustomerDomainService.java
package com.company.customer.domain.service;

// NO Spring imports
// NO JPA imports

public class CustomerDomainService {  // NO @Service
    private final CustomerRepository repository;
    
    public CustomerDomainService(CustomerRepository repository) {
        this.repository = repository;
    }
    
    public Customer registerCustomer(CustomerRegistration registration) {
        // Pure business logic
    }
}
```

**❌ WRONG:**
```java
// domain/service/CustomerDomainService.java
import org.springframework.stereotype.Service;  // ❌ Spring import

@Service  // ❌ Spring annotation in domain
public class CustomerDomainService {
    @Autowired  // ❌ Spring DI in domain
    private CustomerRepository repository;
}
```

### Domain Model (Pure POJOs)

**✅ CORRECT:**
```java
// domain/model/Customer.java
package com.company.customer.domain.model;

// NO JPA imports

public class Customer {  // NO @Entity
    private CustomerId id;
    private String name;
    private String email;
    
    // Constructor, getters, business methods
}
```

**❌ WRONG:**
```java
// domain/model/Customer.java
import jakarta.persistence.*;  // ❌ JPA import

@Entity  // ❌ JPA in domain
@Table(name = "customers")  // ❌ JPA in domain
public class Customer {
    @Id  // ❌ JPA in domain
    private String id;
}
```

### Domain Repository (Interface Only)

**✅ CORRECT:**
```java
// domain/repository/CustomerRepository.java
package com.company.customer.domain.repository;

// NO @Repository annotation
public interface CustomerRepository {  
    Customer save(Customer customer);
    Optional<Customer> findById(CustomerId id);
}
```

**❌ WRONG:**
```java
// domain/repository/CustomerRepository.java
import org.springframework.stereotype.Repository;  // ❌ Spring

@Repository  // ❌ Should be in adapter, not domain
public interface CustomerRepository {
    // ...
}
```

### Application Layer (Spring Orchestration)

**✅ CORRECT:**
```java
// application/service/CustomerApplicationService.java
import org.springframework.stereotype.Service;  // ✓ OK in application
import org.springframework.transaction.annotation.Transactional;  // ✓ OK

@Service  // ✓ OK in application layer
@Transactional  // ✓ OK in application layer
public class CustomerApplicationService {
    
    private final CustomerDomainService domainService;
    
    public CustomerApplicationService(CustomerDomainService domainService) {
        this.domainService = domainService;
    }
}
```

### Adapter Layer (JPA Implementation)

**✅ CORRECT:**
```java
// adapter/persistence/entity/CustomerJpaEntity.java
import jakarta.persistence.*;  // ✓ OK in adapter

@Entity  // ✓ OK in adapter
@Table(name = "customers")
public class CustomerJpaEntity {
    @Id
    private String id;
    // ...
}
```

```java
// adapter/persistence/adapter/CustomerJpaAdapter.java
import org.springframework.stereotype.Repository;  // ✓ OK in adapter

@Repository  // ✓ OK in adapter
public class CustomerJpaAdapter implements CustomerRepository {  // ✓ Implements domain interface
    
    private final CustomerJpaRepository jpaRepository;
    
    @Override
    public Customer save(Customer customer) {
        // Map domain → JPA entity → save → domain
    }
}
```

### Domain Tests (Pure Unit Tests)

**✅ CORRECT:**
```java
// test/domain/service/CustomerDomainServiceTest.java
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;  // ✓ Mockito, not Spring

@ExtendWith(MockitoExtension.class)  // ✓ Pure unit test
class CustomerDomainServiceTest {
    
    @Mock
    private CustomerRepository repository;
    
    private CustomerDomainService service;
    
    @BeforeEach
    void setUp() {
        service = new CustomerDomainService(repository);  // ✓ POJO construction
    }
}
```

**❌ WRONG:**
```java
// test/domain/service/CustomerDomainServiceTest.java
import org.springframework.boot.test.context.SpringBootTest;  // ❌ Spring in domain test

@SpringBootTest  // ❌ Domain tests should NOT use Spring
class CustomerDomainServiceTest {
    @Autowired  // ❌ Should use Mockito
    private CustomerDomainService service;
}
```

## When This Runs

This validation runs when:
- A skill generates code using mod-code-015-hexagonal-base
- **Always** for projects using Hexagonal Light architecture

**Skills that use this:**
- skill-code-020-generate-microservice (always)
- Any skill generating hexagonal services

## Related

- **Module:** mod-code-015-hexagonal-base-java-spring.md
- **ADR:** adr-009-service-architecture-patterns
- **Skill:** skill-code-020-generate-microservice-java-spring

---

**Validation Script Version:** 1.0  
**Last Updated:** 2025-11-25
