# Examples - skill-020-generate-microservice-java-spring

## Examples Index

| Example | Type | Description | Entities | Features |
|---------|------|-------------|----------|----------|
| [example-01-customer-domain-api](./example-01-customer-domain-api/) | domain_api | Basic Domain API | 1 (Customer) | circuit_breaker, persistence, docker |
| example-02-order-composable-api | composable_api | Orchestration API | 1 (OrderRequest) | circuit_breaker, NO persistence |
| example-03-product-multi-entity | domain_api | Multiple entities | 2 (Product, Category) | persistence |

---

## Example 1: Domain API - Customer Service

**Directory:** `example-01-customer-domain-api/`

**Files:**
- `input.json` - Input configuration
- `README.md` - Example description
- `expected/STRUCTURE.md` - Expected output with key file contents

### Input Config

```json
{
  "serviceName": "customer-service",
  "groupId": "com.company",
  "artifactId": "customer-service",
  "basePackage": "com.company.customer",
  "javaVersion": "17",
  "springBootVersion": "3.2.0",
  
  "apiType": "domain_api",
  
  "entities": [
    {
      "name": "Customer",
      "isAggregateRoot": true,
      "fields": [
        { "name": "name", "type": "String", "required": true, "minLength": 2, "maxLength": 100 },
        { "name": "email", "type": "String", "required": true, "format": "email", "unique": true },
        { "name": "age", "type": "int", "required": true, "min": 18, "max": 120 },
        { "name": "tier", "type": "String", "required": false }
      ]
    }
  ],
  
  "features": {
    "resilience": {
      "circuit_breaker": { "enabled": true, "pattern": "basic_fallback" }
    },
    "persistence": { "enabled": true, "database": "postgresql" },
    "health_checks": { "enabled": true },
    "structured_logging": { "enabled": true, "format": "json" },
    "docker": { "enabled": true }
  }
}
```

### Expected Output Structure

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
│   │   │   ├── Customer.java
│   │   │   ├── CustomerId.java
│   │   │   └── CustomerRegistration.java
│   │   ├── service/
│   │   │   └── CustomerDomainService.java
│   │   ├── repository/
│   │   │   └── CustomerRepository.java
│   │   └── exception/
│   │       └── CustomerNotFoundException.java
│   │
│   ├── application/
│   │   └── service/
│   │       └── CustomerApplicationService.java
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
│   │       │   └── CustomerEntity.java
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
    │   └── CustomerDomainServiceTest.java
    └── adapter/rest/controller/
        └── CustomerControllerIntegrationTest.java
```

### Generated OpenAPI (openapi.yaml)

```yaml
openapi: 3.0.3
info:
  title: Customer Service API
  description: Domain API for Customer Management
  version: 1.0.0
  
servers:
  - url: http://localhost:8080
    description: Local development

paths:
  /api/v1/customers:
    post:
      summary: Create a new customer
      operationId: createCustomer
      tags:
        - customers
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateCustomerRequest'
      responses:
        '201':
          description: Customer created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CustomerDTO'
        '400':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
    
  /api/v1/customers/{id}:
    get:
      summary: Get customer by ID
      operationId: getCustomer
      tags:
        - customers
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Customer found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CustomerDTO'
        '404':
          description: Customer not found
          
    put:
      summary: Update customer
      operationId: updateCustomer
      tags:
        - customers
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateCustomerRequest'
      responses:
        '200':
          description: Customer updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CustomerDTO'
                
    delete:
      summary: Delete customer
      operationId: deleteCustomer
      tags:
        - customers
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: Customer deleted

components:
  schemas:
    CustomerDTO:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:
          type: string
          format: email
        age:
          type: integer
        tier:
          type: string
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time
          
    CreateCustomerRequest:
      type: object
      required:
        - name
        - email
        - age
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 100
        email:
          type: string
          format: email
        age:
          type: integer
          minimum: 18
          maximum: 120
          
    UpdateCustomerRequest:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 100
        email:
          type: string
          format: email
          
    ErrorResponse:
      type: object
      properties:
        timestamp:
          type: string
          format: date-time
        status:
          type: integer
        error:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            $ref: '#/components/schemas/FieldError'
            
    FieldError:
      type: object
      properties:
        field:
          type: string
        message:
          type: string
```

---

## Example 2: Composable API - Order Orchestration

### Input Config

```json
{
  "serviceName": "order-orchestration-api",
  "groupId": "com.company",
  "basePackage": "com.company.order.orchestration",
  "javaVersion": "17",
  "springBootVersion": "3.2.0",
  
  "apiType": "composable_api",
  
  "entities": [
    {
      "name": "OrderRequest",
      "isAggregateRoot": true,
      "fields": [
        { "name": "customerId", "type": "String", "required": true },
        { "name": "items", "type": "List", "required": true },
        { "name": "totalAmount", "type": "BigDecimal", "required": true }
      ]
    }
  ],
  
  "features": {
    "resilience": {
      "circuit_breaker": { "enabled": true, "pattern": "multiple_fallbacks" }
    },
    "persistence": { "enabled": false },
    "health_checks": { "enabled": true }
  }
}
```

**Note:** `persistence.enabled: false` because Composable APIs are stateless orchestrators.

---

## Example 3: Multiple Entities

### Input Config

```json
{
  "serviceName": "product-catalog-service",
  "groupId": "com.company",
  "basePackage": "com.company.catalog",
  "javaVersion": "17",
  "springBootVersion": "3.2.0",
  
  "apiType": "domain_api",
  
  "entities": [
    {
      "name": "Product",
      "isAggregateRoot": true,
      "fields": [
        { "name": "name", "type": "String", "required": true },
        { "name": "description", "type": "String", "required": false },
        { "name": "price", "type": "BigDecimal", "required": true },
        { "name": "categoryId", "type": "String", "required": true }
      ]
    },
    {
      "name": "Category",
      "isAggregateRoot": false,
      "fields": [
        { "name": "name", "type": "String", "required": true },
        { "name": "parentCategoryId", "type": "String", "required": false }
      ]
    }
  ],
  
  "features": {
    "persistence": { "enabled": true, "database": "postgresql" },
    "health_checks": { "enabled": true }
  }
}
```

**Note:** Multiple entities generate separate domain/application/adapter classes for each.
