# Skill: Fusion REST API Generator (4-Layer Model)

**Skill ID:** skill-021-api-rest-java-spring  
**Extends:** skill-020-microservice-java-spring  
**Domain:** code  
**Layer:** soi  
**Type:** GENERATE  
**Version:** 2.2.0  
**Status:** Active
**Last Updated:** 2025-12-22

---

## Overview

Extends `skill-020-microservice-java-spring` to generate REST APIs following the **Fusion API Model** (4-layer: Experience, Composable, Domain, System) defined in ADR-001.

> **IMPORTANT:** This skill applies ONLY when the request explicitly references a **Fusion API**.
> See ADR-001 "Fusion API Identification" section for inference rules.
> If the request does not mention "Fusion" with an API layer name, either ask for clarification
> or use skill-020 for internal microservices.

**This skill inherits ALL capabilities from skill-020** and adds Fusion API-specific patterns:
- Pagination (PageResponse, filtering, sorting)
- HATEOAS (hypermedia links)
- Compensation (SAGA participation for Domain APIs)

### Inheritance Model

```
skill-020-microservice-java-spring (inherited)
├── mod-015: Hexagonal Light base
├── mod-001-004: Resilience patterns
├── mod-016/017: Persistence patterns
├── mod-018: API Integration
├── [future: observability, caching, etc.]
│
└── skill-021-api-rest-java-spring (this skill - delta only)
    ├── mod-019: API Public Exposure (pagination, HATEOAS)
    └── mod-020: Compensation (Domain layer only)
```

---

## Pre-conditions (Activation Rules)

This skill should ONLY be activated when the request explicitly references a **Fusion API**. 
Follow the inference rules defined in `runtime/discovery/skill-index.yaml` (section `fusion_api_rules`).

### When to Use This Skill

| Prompt Contains | Action | Skill |
|-----------------|--------|-------|
| "Fusion" + API layer (Domain/System/BFF/Experience/Composable) | ✅ Apply directly | skill-021 |
| API layer WITHOUT "Fusion" (e.g., "Domain API") | ⚠️ ASK for clarification | - |
| "microservicio", "servicio interno", "API interna" | ❌ Use base skill | skill-020 |

### Examples

**Use skill-021:**
- "Genera una Fusion Domain API para Customer"
- "Implementar la API de Sistema Fusion para Parties"
- "Create a Fusion BFF for mobile channel"

**ASK for clarification:**
- "Genera una Domain API para Customer" → Ask: "¿Te refieres a una API del modelo Fusion?"
- "Create a System API for Parties" → Ask: "Is this a Fusion System API?"

**Use skill-020 (NOT this skill):**
- "Genera un microservicio para Customer"
- "Implementar un servicio interno de notificaciones"
- "Create an internal API for event processing"

---

## Extension Declaration

```yaml
extends: skill-020-microservice-java-spring

# This skill INHERITS:
#   - All modules from skill-020
#   - All parameters from skill-020
#   - All validation from skill-020
#   - Execution flow from skill-020

# This skill ADDS:
#   - Additional modules (see below)
#   - Additional parameters (apiLayer)
#   - Additional validation (Tier 3 for mod-019, mod-020)
```

---

## Knowledge Dependencies (Delta Only)

> Inherited dependencies from skill-020 are not repeated here.

| Type | Asset | Purpose |
|------|-------|---------|
| ADR | adr-001-api-design | API model, REST standards, pagination, HATEOAS |
| ADR | adr-013-distributed-transactions | Compensation patterns for Domain APIs |
| ERI | eri-code-014-api-public-exposure-java-spring | Pagination, HATEOAS reference |
| ERI | eri-code-015-distributed-transactions-java-spring | Compensation reference |
| Module | mod-code-019-api-public-exposure-java-spring | Pagination, HATEOAS templates |
| Module | mod-code-020-compensation-java-spring | Compensation templates |

---

## Parameters (Delta Only)

> Inherited parameters from skill-020 (serviceName, basePackage, entities, features.*) are not repeated.

### Additional Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `apiLayer` | enum | API layer type | `domain` |

### API Layer Values

| Value | Description | Modules Added |
|-------|-------------|---------------|
| `experience` | BFF for UI channels | mod-019 (full HATEOAS) |
| `composable` | Multi-domain orchestration | mod-019 (pagination only) |
| `domain` | Business capabilities | mod-019 (full) + mod-020 (if compensation enabled) |
| `system` | SoR integration | mod-019 (pagination only) |

### Additional Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pagination.defaultSize` | int | 20 | Default page size |
| `pagination.maxSize` | int | 100 | Maximum page size |
| `features.compensation.enabled` | boolean | false | Enable SAGA compensation (Domain APIs only) |

---

## Module Resolution (Delta Only)

> Base modules are resolved by skill-020. This section defines ADDITIONAL modules.

### Additional Modules

| Condition | Module | Purpose |
|-----------|--------|---------|
| always | mod-code-019-api-public-exposure-java-spring | Pagination, filtering, HATEOAS |
| `apiLayer = domain` AND `features.compensation.enabled = true` | mod-code-020-compensation-java-spring | SAGA compensation interface |

> **NOTE:** mod-020 (compensation) is **opt-in**. Even for Domain APIs, compensation is only 
> generated when explicitly requested via `features.compensation.enabled = true`. This is because
> not all Domain APIs participate in distributed transactions (SAGAs).

### Layer-Based Feature Matrix

| Feature | Experience | Composable | Domain | System |
|---------|------------|------------|--------|--------|
| Pagination | ✅ | ✅ | ✅ | ✅ |
| HATEOAS | ✅ | ❌ | ✅ | ❌ |
| Compensation | ❌ | ❌ | ✅ | ❌ |

### Module Selection Logic

```python
# Pseudocode for module resolution

# 1. Inherit all modules from skill-020
modules = skill_020.resolve_modules(input)

# 2. Add mod-019 (always for REST APIs)
modules.add("mod-code-019-api-public-exposure-java-spring")

# 3. Add mod-020 (Domain layer AND compensation explicitly enabled)
# NOTE: compensation is opt-in, not automatic for Domain APIs
if input.apiLayer == "domain" and input.features.compensation.enabled == True:
    modules.add("mod-code-020-compensation-java-spring")

# 4. Configure HATEOAS based on layer
if input.apiLayer in ["experience", "domain"]:
    modules.configure("mod-019", hateoas=True)
else:
    modules.configure("mod-019", hateoas=False)

return modules
```

---

## Output Specification (Delta Only)

> Base output structure is defined by skill-020. This section defines ADDITIONAL artifacts.

### Additional Generated Artifacts

| Artifact | Path | Condition |
|----------|------|-----------|
| PageResponse DTO | `adapter/in/rest/dto/PageResponse.java` | Always |
| Filter DTOs | `adapter/in/rest/dto/{Entity}Filter.java` | Always |
| Pagination config | `infrastructure/web/PageableConfig.java` | Always |
| HATEOAS assemblers | `adapter/in/rest/assembler/` | Experience, Domain |
| Compensation interfaces | `domain/transaction/` | Domain + `features.compensation.enabled` |
| /compensate endpoint | Controller method | Domain + `features.compensation.enabled` |

### Extended Output Structure

```
{serviceName}/
├── [inherited from skill-020]
│   ├── domain/model/
│   ├── domain/repository/
│   ├── application/service/
│   ├── adapter/in/rest/{Entity}Controller.java
│   └── infrastructure/
│
└── [added by skill-021]
    ├── domain/transaction/           # Domain only
    │   ├── Compensable.java
    │   ├── CompensationRequest.java
    │   ├── CompensationResult.java
    │   ├── CompensationStatus.java
    │   └── TransactionLog.java
    ├── adapter/in/rest/
    │   ├── dto/
    │   │   ├── PageResponse.java
    │   │   └── {Entity}Filter.java
    │   └── assembler/                # Experience, Domain
    │       └── {Entity}ModelAssembler.java
    └── infrastructure/web/
        └── PageableConfig.java
```

---

## Validation (Delta Only)

> Tier 1 and Tier 2 validation is inherited from skill-020.

### Additional Tier 3 Validation

| Validator | Module | Condition |
|-----------|--------|-----------|
| pagination-check.sh | mod-019 | Always |
| config-check.sh | mod-019 | Always |
| hateoas-check.sh | mod-019 | apiLayer IN [experience, domain] |
| compensation-interface-check.sh | mod-020 | apiLayer = domain |
| compensation-endpoint-check.sh | mod-020 | apiLayer = domain |
| transaction-log-check.sh | mod-020 | apiLayer = domain |

### Validation Orchestration

```bash
#!/bin/bash
# validate.sh for skill-021

SERVICE_DIR="$1"
API_LAYER="$2"

# 1. Run inherited validation from skill-020
source skill-020/validation/validate.sh "$SERVICE_DIR"
ERRORS=$?

# 2. Run additional Tier 3 validation (mod-019)
source mod-019/validation/pagination-check.sh "$SERVICE_DIR"
ERRORS=$((ERRORS + $?))

source mod-019/validation/config-check.sh "$SERVICE_DIR"
ERRORS=$((ERRORS + $?))

if [[ "$API_LAYER" == "experience" || "$API_LAYER" == "domain" ]]; then
    source mod-019/validation/hateoas-check.sh "$SERVICE_DIR"
    ERRORS=$((ERRORS + $?))
fi

# 3. Run additional Tier 3 validation (mod-020) - Domain only
if [[ "$API_LAYER" == "domain" ]]; then
    source mod-020/validation/compensation-interface-check.sh "$SERVICE_DIR"
    ERRORS=$((ERRORS + $?))
    
    source mod-020/validation/compensation-endpoint-check.sh "$SERVICE_DIR"
    ERRORS=$((ERRORS + $?))
fi

exit $ERRORS
```

---

## Execution Flow

1. **Parse input** and validate `apiLayer` parameter
2. **Delegate to skill-020** for base generation
3. **Apply mod-019** templates (pagination, HATEOAS if applicable)
4. **Apply mod-020** templates (Domain layer only)
5. **Merge configurations** (application.yml additions)
6. **Run validation** (inherited + additional)
7. **Generate manifest** with full traceability

---

## Input Schema (Extension)

> Extends skill-020 input schema with additional properties.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "allOf": [
    { "$ref": "skill-020-input-schema.json" },
    {
      "type": "object",
      "required": ["apiLayer"],
      "properties": {
        "apiLayer": {
          "type": "string",
          "enum": ["experience", "composable", "domain", "system"],
          "description": "API layer per ADR-001"
        },
        "pagination": {
          "type": "object",
          "properties": {
            "defaultSize": { "type": "integer", "default": 20 },
            "maxSize": { "type": "integer", "default": 100 }
          }
        }
      }
    }
  ]
}
```

---

## Example Input

```json
{
  "serviceName": "customer-management-api",
  "basePackage": "com.bank.customer",
  "apiLayer": "domain",
  "entities": [
    {
      "name": "Customer",
      "fields": [
        { "name": "firstName", "type": "String", "required": true },
        { "name": "lastName", "type": "String", "required": true },
        { "name": "email", "type": "String", "required": true }
      ]
    }
  ],
  "features": {
    "resilience": { "enabled": true },
    "persistence": { "type": "system_api" }
  }
}
```

---

## Related Skills

| Skill | Relationship |
|-------|--------------|
| skill-020-microservice-java-spring | **Parent** - provides base microservice generation |
| skill-022-api-grpc-java-spring | **Sibling** - extends skill-020 for gRPC (planned) |
| skill-023-api-async-java-spring | **Sibling** - extends skill-020 for Async (planned) |

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-19 | 2.0.0 | Redesigned as extension of skill-020 | C4E Team |
| 2025-12-19 | 1.0.0 | Initial version (standalone) | C4E Team |
